"""
commonLIT_scraper.py — with skip-if-already-downloaded logic
"""

import requests
import json
import os
import time
import tarfile
import io
import PyPDF2
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

S3_BASE = "https://indian-supreme-court-judgments.s3.amazonaws.com"


def already_downloaded(year: int, output_dir: str, max_judgments: int) -> bool:
    """
    Check if judgments for this year are already downloaded.
    Returns True if we already have >= 90% of max_judgments files for this year.
    """
    if not os.path.exists(output_dir):
        return False

    existing = [
        f for f in os.listdir(output_dir)
        if f.startswith(f"sc_{year}_") and f.endswith(".json")
    ]
    count = len(existing)

    if count >= max_judgments * 0.9:
        print(
            f"    ✓ Already have {count} judgments for {year} — skipping download")
        return True

    if count > 0:
        print(
            f"    Found {count} existing judgments for {year} — resuming from where left off")

    return False


def get_existing_filenames(year: int, output_dir: str) -> set:
    """Return set of already-saved filenames for a year (for resume support)."""
    existing = set()
    if not os.path.exists(output_dir):
        return existing

    for f in os.listdir(output_dir):
        if f.startswith(f"sc_{year}_") and f.endswith(".json"):
            try:
                with open(os.path.join(output_dir, f), encoding="utf-8") as fp:
                    data = json.load(fp)
                    if "filename" in data:
                        existing.add(data["filename"])
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


def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes."""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception:
        return ""


def download_year(year: int, output_dir: str, max_judgments: int = 100) -> int:
    """Download tar for a year, extract PDFs, save as JSON. Returns count saved."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Skip check ───────────────────────────────────────────
    if already_downloaded(year, output_dir, max_judgments):
        return len([
            f for f in os.listdir(output_dir)
            if f.startswith(f"sc_{year}_") and f.endswith(".json")
        ])

    # ── Resume: get already-saved filenames ──────────────────
    existing_filenames = get_existing_filenames(year, output_dir)
    existing_count = len([
        f for f in os.listdir(output_dir)
        if f.startswith(f"sc_{year}_") and f.endswith(".json")
    ])
    remaining = max_judgments - existing_count

    if remaining <= 0:
        print(f"    Already have {existing_count} — nothing to do")
        return existing_count

    # ── Metadata index ────────────────────────────────────────
    print(f"    Fetching metadata index...")
    meta_lookup = get_metadata_index(year)

    # ── Download tar ──────────────────────────────────────────
    tar_url = f"{S3_BASE}/data/tar/year={year}/english/english.tar"
    print(f"    Downloading: {tar_url}")

    try:
        r = requests.get(tar_url, headers=HEADERS, timeout=300, stream=True)
        if r.status_code != 200:
            print(f"    ✗ HTTP {r.status_code}")
            return existing_count

        total_size = int(r.headers.get("content-length", 0))
        content = b""
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"    {year} tar") as pbar:
            for chunk in r.iter_content(chunk_size=65536):
                content += chunk
                pbar.update(len(chunk))

    except Exception as e:
        print(f"    ✗ Download failed: {e}")
        return existing_count

    # ── Open tar ──────────────────────────────────────────────
    try:
        tf = tarfile.open(fileobj=io.BytesIO(content))
    except Exception as e:
        print(f"    ✗ Cannot open tar: {e}")
        return existing_count

    all_members = tf.getmembers()
    pdf_members = [m for m in all_members if m.name.lower().endswith(".pdf")]

    print(f"    Files in tar: {len(all_members)} | PDFs: {len(pdf_members)}")

    if not pdf_members:
        extensions = set(os.path.splitext(m.name)[1] for m in all_members[:20])
        print(f"    ✗ No PDFs found. Extensions in tar: {extensions}")
        tf.close()
        return existing_count

    # ── Filter out already-saved files ───────────────────────
    to_process = [
        m for m in pdf_members
        if os.path.basename(m.name) not in existing_filenames
    ][:remaining]

    print(
        f"    Already saved: {existing_count} | To process: {len(to_process)}")

    # ── Extract + save ────────────────────────────────────────
    saved = existing_count

    for i, member in enumerate(tqdm(to_process, desc=f"    Extracting {year}")):
        fname_out = f"sc_{year}_{saved:04d}.json"
        fpath_out = os.path.join(output_dir, fname_out)

        try:
            f = tf.extractfile(member)
            if not f:
                continue

            pdf_bytes = f.read()
            text = pdf_bytes_to_text(pdf_bytes)

            if len(text) < 200:
                continue

            base_name = os.path.basename(member.name)
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
                "metadata": meta
            }

            with open(fpath_out, "w", encoding="utf-8") as out:
                json.dump(data, out, ensure_ascii=False, indent=2)

            saved += 1

        except Exception:
            continue

    tf.close()
    return saved


def scrape_commonlii(
    courts: list = None,
    years: list = None,
    output_dir: str = "data/raw/judgments",
    max_per_court_year: int = 100
) -> list:
    """Main entry point. courts param ignored — SC only dataset."""

    if years is None:
        years = list(range(2018, 2025))

    os.makedirs(output_dir, exist_ok=True)
    total = 0

    print(f"\n{'='*55}")
    print("AWS Open Data: Indian Supreme Court Judgments")
    print("License: CC BY 4.0  |  Coverage: 1950-2025")
    print(f"Years: {years}  |  Max per year: {max_per_court_year}")
    print(f"{'='*55}")

    for year in years:
        print(f"\n── Year {year} ──")
        count = download_year(year, output_dir, max_per_court_year)
        total += count
        print(f"✓ {count} judgments for {year}")
        time.sleep(1)

    print(f"\n{'='*55}")
    print(f"TOTAL: {total} judgments")
    print(f"Path:  {os.path.abspath(output_dir)}")
    print(f"{'='*55}")
    return []


if __name__ == "__main__":
    scrape_commonlii(years=[2022, 2023, 2024], max_per_court_year=50)
