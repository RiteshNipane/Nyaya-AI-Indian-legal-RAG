"""
pdf_utils.py

Shared helpers for scraping large judgment archives (Supreme Court + High Courts).

What changed versus the original single-file approach, and why it matters at
50,000+ judgments:

* Streams the tar from S3 (tarfile mode "r|*") instead of buffering a whole
  multi-GB archive in memory.
* Extracts PDF text in a process pool (PyPDF2 is CPU-bound).
* Keeps an append-only "seen" manifest per partition, so an interrupted run
  resumes without reopening thousands of JSON files and without reprocessing
  PDFs that had no extractable text.
* Gives each output file a stable, hash-based name so re-runs never duplicate.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, Optional

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

MIN_TEXT_CHARS = 200


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Plain text from PDF bytes ('' if unreadable or image-only)."""
    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception:
        return ""


def stable_filename(prefix: str, key: str) -> str:
    """e.g. sc_2023_3f9a1c0b7d2e.json — deterministic for a given tar member."""
    return f"{prefix}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}.json"


class SeenManifest:
    """Append-only set of processed tar-member basenames, persisted to a text file."""

    def __init__(self, path: str, bootstrap: Optional[Callable[[], Iterable[str]]] = None):
        self.path = path
        self.seen: set = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.seen = {line.rstrip("\n") for line in f if line.strip()}
        elif bootstrap is not None:
            # one-time import of files saved by an older version of the scraper
            self.seen = set(bootstrap())
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(name + "\n" for name in sorted(self.seen))

    def __contains__(self, name: str) -> bool:
        return name in self.seen

    def add(self, name: str) -> None:
        if name in self.seen:
            return
        self.seen.add(name)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(name + "\n")


def _open_http_stream(url: str, headers: dict, timeout=(30, 300)):
    import requests

    r = requests.get(url, headers=headers, stream=True, timeout=timeout)
    r.raise_for_status()
    r.raw.decode_content = True
    return r


def process_tar_stream(
    url: str,
    seen: SeenManifest,
    on_text: Callable[[str, str], bool],
    should_stop: Callable[[], bool] = lambda: False,
    workers: int = 4,
    headers: Optional[dict] = None,
    extractor: Callable[[bytes], str] = extract_pdf_text,
    opener: Optional[Callable[[str], object]] = None,
    retries: int = 3,
    log: Callable[[str], None] = print,
) -> dict:
    """Stream a tar of PDFs, extract text in parallel, hand each result to `on_text`.

    on_text(basename, text) -> True when a record was saved. Members whose text is
    shorter than MIN_TEXT_CHARS are skipped (but recorded as seen). `should_stop()`
    is polled between members so callers can cap the number of saved records.

    `opener(url)` may return an object with a `.raw` file-like attribute and
    `.close()`; it exists so tests can feed an in-memory tar.
    """
    headers = headers or HEADERS
    stats = {"saved": 0, "skipped_short": 0, "skipped_seen": 0, "errors": 0}
    last_err: Optional[Exception] = None

    for attempt in range(retries):
        try:
            _stream_once(url, seen, on_text, should_stop, workers, headers,
                         extractor, opener, stats)
            return stats
        except Exception as e:  # network drop mid-stream, truncated tar, ...
            last_err = e
            wait = 2 ** attempt * 2
            log(f"    ⚠ stream interrupted ({e}); retry {attempt + 1}/{retries} in {wait}s "
                f"(progress is kept in the manifest)")
            time.sleep(wait)

    log(f"    ✗ giving up on {url}: {last_err}")
    stats["failed"] = True
    return stats


def _stream_once(url, seen, on_text, should_stop, workers, headers, extractor,
                 opener, stats):
    resp = (opener or (lambda u: _open_http_stream(u, headers)))(url)
    tf = tarfile.open(fileobj=resp.raw, mode="r|*")
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    pending: list = []  # (basename, future-or-text)
    max_pending = workers * 4 if pool else 1

    def drain(limit: int) -> None:
        while len(pending) > limit:
            if should_stop():
                # cap reached: discard queued work; it was never recorded as seen,
                # so a later run (with a higher cap) picks it up
                pending.clear()
                return
            name, item = pending.pop(0)
            try:
                text = item.result() if pool else item
            except Exception:
                stats["errors"] += 1
                continue
            if len(text) < MIN_TEXT_CHARS:
                stats["skipped_short"] += 1
                seen.add(name)
                continue
            try:
                if on_text(name, text):
                    stats["saved"] += 1
                seen.add(name)
            except Exception:
                stats["errors"] += 1

    try:
        for member in tf:
            if should_stop():
                break
            if not member.isfile() or not member.name.lower().endswith(".pdf"):
                continue
            name = os.path.basename(member.name)
            if name in seen:
                stats["skipped_seen"] += 1
                continue
            data = tf.extractfile(member).read()
            pending.append((name, pool.submit(extractor, data) if pool else extractor(data)))
            drain(max_pending)
            if should_stop():
                break
        drain(0)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
        tf.close()
        try:
            resp.close()
        except Exception:
            pass
