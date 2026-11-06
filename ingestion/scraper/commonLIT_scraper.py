"""
commonLIT_scraper.py — Supreme Court judgments from AWS Open Data (1950–2025)

Scaled for the full archive:
  * streams each year's tar instead of loading it into memory
  * extracts PDF text in a process pool
  * resumes from an append-only manifest (no re-reading thousands of JSON files)
  * `max_judgments=None` means "everything in the year"

Usage
-----
    python ingestion/scraper/commonLIT_scraper.py --years 2018-2024 --max-per-year 200
    python ingestion/scraper/commonLIT_scraper.py --years 1950-2025 --workers 6   # full archive
"""

import argparse
import json
import os
import time
from typing import Optional

import requests

try:  # imported as a package module (run_all_scraper) or run directly
    from scraper.pdf_utils import (
        HEADERS, SeenManifest, extract_pdf_text, process_tar_stream, stable_filename)
except ImportError:
    from pdf_utils import (
        HEADERS, SeenManifest, extract_pdf_text, process_tar_stream, stable_filename)

S3_BASE = "https://indian-supreme-court-judgments.s3.amazonaws.com"

# Kept for backwards compatibility with older imports.
pdf_bytes_to_text = extract_pdf_text


def _year_files(year: int, output_dir: str) -> list:
    if not os.path.exists(output_dir):
        return []
    return [f for f in os.listdir(output_dir)
            if f.startswith(f"sc_{year}_") and f.endswith(".json")]


def already_downloaded(year: int, output_dir: str, max_judgments: Optional[int]) -> bool:
    """True when we already hold >= 90% of the requested cap for this year.

    With no cap (None) we never skip: the manifest makes a re-run cheap and picks up
    judgments added by the bi-monthly dataset updates.
    """
    count = len(_year_files(year, output_dir))
    if max_judgments is not None and count >= max_judgments * 0.9:
        print(f"    ✓ Already have {count} judgments for {year} — skipping download")
        return True
    if count > 0:
        print(f"    Found {count} existing judgments for {year} — resuming")
    return False


def get_existing_filenames(year: int, output_dir: str) -> set:
    """Original tar-member names of files saved by earlier runs (reads each JSON once)."""
    existing = set()
    for f in _year_files(year, output_dir):
        try:
            with open(os.path.join(output_dir, f), encoding="utf-8") as fp:
                name = json.load(fp).get("filename")
                if name:
                    existing.add(name)
        except Exception:
            continue
    return existing


def get_metadata_index(year: int) -> dict:
    """Download metadata index. Returns dict: { filename -> metadata }"""
    url = f"{S3_BASE}/data/tar/year={year}/english/english.index.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            index_list = r.json()
            return {
                (item.get("filename") or item.get("file_name") or ""): item
                for item in index_list
            }
        print(f"    Index HTTP {r.status_code} — proceeding without metadata")
        return {}
    except Exception as e:
        print(f"    Index error: {e}")
        return {}


def download_year(year: int, output_dir: str, max_judgments: Optional[int] = 100,
                  workers: int = 4) -> int:
    """Stream the year's tar, save each judgment as JSON. Returns files on disk for the year."""
    os.makedirs(output_dir, exist_ok=True)

    if already_downloaded(year, output_dir, max_judgments):
        return len(_year_files(year, output_dir))

    manifest = SeenManifest(
        os.path.join(output_dir, f".seen_sc_{year}.txt"),
        bootstrap=lambda: get_existing_filenames(year, output_dir),
    )
    existing = len(_year_files(year, output_dir))
    budget = None if max_judgments is None else max_judgments - existing
    if budget is not None and budget <= 0:
        print(f"    Already have {existing} — nothing to do")
        return existing

    print("    Fetching metadata index...")
    meta_lookup = get_metadata_index(year)

    saved_now = {"n": 0}

    def on_text(base_name: str, text: str) -> bool:
        meta = meta_lookup.get(base_name, {})
        data = {
            "title":    meta.get("title") or meta.get("case_title") or base_name,
            "year":     year,
            "date":     meta.get("date") or meta.get("decision_date") or str(year),
            "text":     text,
            "court":    "Supreme Court of India",
            "source":   "aws-open-data/indian-supreme-court-judgments",
            "doc_type": "judgment",
            "filename": base_name,
            "metadata": meta,
        }
        path = os.path.join(output_dir, stable_filename(f"sc_{year}", base_name))
        with open(path, "w", encoding="utf-8") as out:
            json.dump(data, out, ensure_ascii=False, indent=2)
        saved_now["n"] += 1
        if saved_now["n"] % 100 == 0:
            print(f"    … {saved_now['n']} saved for {year}")
        return True

    def should_stop() -> bool:
        return budget is not None and saved_now["n"] >= budget

    tar_url = f"{S3_BASE}/data/tar/year={year}/english/english.tar"
    print(f"    Streaming: {tar_url}")
    stats = process_tar_stream(tar_url, manifest, on_text, should_stop, workers=workers)
    print(f"    saved {stats['saved']} | skipped (no text) {stats['skipped_short']} | "
          f"already had {stats['skipped_seen']} | errors {stats['errors']}")

    return len(_year_files(year, output_dir))


def scrape_commonlii(
    courts: list = None,
    years: list = None,
    output_dir: str = "data/raw/judgments",
    max_per_court_year: Optional[int] = 100,
    workers: int = 4,
) -> list:
    """Main entry point. `courts` is ignored — this dataset is Supreme Court only.

    max_per_court_year=None downloads every judgment for each year.
    """
    if years is None:
        years = list(range(2018, 2025))

    os.makedirs(output_dir, exist_ok=True)
    total = 0
    cap = "all" if max_per_court_year is None else max_per_court_year

    print(f"\n{'='*55}")
    print("AWS Open Data: Indian Supreme Court Judgments")
    print("License: CC BY 4.0  |  Coverage: 1950-2025")
    print(f"Years: {years[0]}–{years[-1]} ({len(years)})  |  Max per year: {cap}  |  Workers: {workers}")
    print(f"{'='*55}")

    for year in years:
        print(f"\n── Year {year} ──")
        count = download_year(year, output_dir, max_per_court_year, workers)
        total += count
        print(f"✓ {count} judgments for {year}")
        time.sleep(1)

    print(f"\n{'='*55}")
    print(f"TOTAL: {total} judgments")
    print(f"Path:  {os.path.abspath(output_dir)}")
    print(f"{'='*55}")
    return []


def parse_years(spec: str) -> list:
    """'2018-2024' -> [2018..2024]; '2019,2021' -> [2019, 2021]."""
    years: list = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            years.extend(range(int(a), int(b) + 1))
        elif part:
            years.append(int(part))
    return sorted(set(years))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download Supreme Court judgments (AWS Open Data)")
    ap.add_argument("--years", default="2022-2024", help="e.g. 2018-2024 or 1950-2025")
    ap.add_argument("--max-per-year", type=int, default=50,
                    help="cap per year; 0 = no cap (everything)")
    ap.add_argument("--workers", type=int, default=4, help="PDF extraction processes")
    ap.add_argument("--out", default="data/raw/judgments")
    a = ap.parse_args()
    scrape_commonlii(years=parse_years(a.years), output_dir=a.out,
                     max_per_court_year=(a.max_per_year or None), workers=a.workers)
