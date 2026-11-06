"""
highcourt_scraper.py — High Court judgments from AWS Open Data

Dataset : s3://indian-high-court-judgments (region ap-south-1, CC-BY-4.0, 25 courts / 45 benches)
Layout  : data/tar/year=YYYY/court=<code>/bench=<bench>/<part>.tar   (court code with '~' -> '_')
          Court/bench codes: opendata/docs/high_courts.csv in github.com/vanga/indian-high-court-judgments

Output  : data/raw/hc_judgments/hc_<court>_<year>_<hash>.json  (doc_type "judgment", court = the
          High Court's name), which embeddings/batch_ingestion.py already ingests.

Notes
-----
* The full archive is very large. Start with a sample:
      python ingestion/scraper/highcourt_scraper.py --courts 27_1 7_26 --years 2023-2024 --max-per-court-year 300
* Benches are discovered from the bucket listing, so new benches are picked up automatically.
* Some PDFs are scans without a text layer; those are skipped (recorded in the manifest).
* Titles are a best-effort guess from the first lines of the judgment ("X vs Y"); the file
  name is the fallback.
"""

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Optional

import requests

try:
    from scraper.pdf_utils import HEADERS, SeenManifest, process_tar_stream, stable_filename
except ImportError:
    from pdf_utils import HEADERS, SeenManifest, process_tar_stream, stable_filename

S3_HOST = "https://indian-high-court-judgments.s3.ap-south-1.amazonaws.com"
_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# court code -> name (from high_courts.csv, as of Nov 2026)
HIGH_COURTS = {
    "9_13": "Allahabad High Court",
    "27_1": "Bombay High Court",
    "19_16": "Calcutta High Court",
    "18_6": "Gauhati High Court",
    "36_29": "High Court for State of Telangana",
    "28_2": "High Court of Andhra Pradesh",
    "22_18": "High Court of Chhattisgarh",
    "7_26": "High Court of Delhi",
    "24_17": "High Court of Gujarat",
    "2_5": "High Court of Himachal Pradesh",
    "1_12": "High Court of Jammu and Kashmir",
    "20_7": "High Court of Jharkhand",
    "29_3": "High Court of Karnataka",
    "32_4": "High Court of Kerala",
    "23_23": "High Court of Madhya Pradesh",
    "14_25": "High Court of Manipur",
    "17_21": "High Court of Meghalaya",
    "21_11": "High Court of Orissa",
    "3_22": "High Court of Punjab and Haryana",
    "8_9": "High Court of Rajasthan",
    "11_24": "High Court of Sikkim",
    "16_20": "High Court of Tripura",
    "5_15": "High Court of Uttarakhand",
    "33_10": "Madras High Court",
    "10_8": "Patna High Court",
}


# ─────────────────────────────────────────────────────────
# S3 listing (anonymous ListObjectsV2 over HTTPS)
# ─────────────────────────────────────────────────────────

def parse_listing(xml_text: str) -> tuple:
    """Parse a ListObjectsV2 response -> (common_prefixes, keys, next_token)."""
    root = ET.fromstring(xml_text)
    prefixes = [e.text for e in root.findall("s3:CommonPrefixes/s3:Prefix", _NS)]
    keys = [e.text for e in root.findall("s3:Contents/s3:Key", _NS)]
    truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=_NS) == "true")
    token = root.findtext("s3:NextContinuationToken", namespaces=_NS) if truncated else None
    return prefixes, keys, token


def list_s3(prefix: str, delimiter: Optional[str] = "/", session=None) -> tuple:
    """All (common_prefixes, keys) under `prefix`, following pagination."""
    http = session or requests
    prefixes: list = []
    keys: list = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        r = http.get(S3_HOST + "/", params=params, headers=HEADERS, timeout=60)
        r.raise_for_status()
        p, k, token = parse_listing(r.text)
        prefixes += p
        keys += k
        if not token:
            return prefixes, keys


def _last_segment(prefix: str, name: str) -> str:
    """'data/tar/year=2024/court=27_1/' -> '27_1' for name='court'."""
    m = re.search(rf"{name}=([^/]+)/?$", prefix)
    return m.group(1) if m else ""


def list_benches(year: int, court: str, session=None) -> list:
    prefixes, _ = list_s3(f"data/tar/year={year}/court={court}/", session=session)
    return sorted(b for b in (_last_segment(p, "bench") for p in prefixes) if b)


def list_tar_parts(year: int, court: str, bench: str, session=None) -> list:
    _, keys = list_s3(f"data/tar/year={year}/court={court}/bench={bench}/",
                      delimiter=None, session=session)
    return sorted(k for k in keys if k.endswith(".tar"))


# ─────────────────────────────────────────────────────────
# Record building
# ─────────────────────────────────────────────────────────

_PARTY_LINE = re.compile(r"\b(?:vs?\.?|versus)\b", re.IGNORECASE)


def guess_title(text: str, fallback: str) -> str:
    """Best-effort case title: the first short line in the header containing 'v/vs/versus'."""
    for line in text[:1500].splitlines():
        line = " ".join(line.split())
        if 8 <= len(line) <= 160 and _PARTY_LINE.search(line):
            return line
    return fallback


def build_record(court: str, bench: str, year: int, base_name: str, text: str,
                 part_key: str) -> dict:
    court_name = HIGH_COURTS.get(court, f"High Court ({court})")
    return {
        "title": guess_title(text, base_name),
        "year": year,
        "date": str(year),
        "text": text,
        "court": court_name,
        "court_code": court,
        "court_level": "high",
        "bench": bench,
        "source": "aws-open-data/indian-high-court-judgments",
        "doc_type": "judgment",
        "filename": base_name,
        "tar_part": part_key,
    }


# ─────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────

def download_court_year(court: str, year: int, output_dir: str,
                        max_judgments: Optional[int] = None, workers: int = 4) -> int:
    """Download one court's judgments for a year across all benches. Returns files saved now."""
    os.makedirs(output_dir, exist_ok=True)
    manifest = SeenManifest(os.path.join(output_dir, f".seen_hc_{court}_{year}.txt"))
    saved_now = {"n": 0}

    try:
        benches = list_benches(year, court)
    except Exception as e:
        print(f"    ✗ cannot list benches for {court}/{year}: {e}")
        return 0
    if not benches:
        print(f"    (no data for {HIGH_COURTS.get(court, court)} in {year})")
        return 0

    for bench in benches:
        if max_judgments is not None and saved_now["n"] >= max_judgments:
            break
        try:
            parts = list_tar_parts(year, court, bench)
        except Exception as e:
            print(f"    ✗ cannot list parts for bench {bench}: {e}")
            continue

        for part_key in parts:
            if max_judgments is not None and saved_now["n"] >= max_judgments:
                break

            def on_text(base_name: str, text: str, _bench=bench, _part=part_key) -> bool:
                rec = build_record(court, _bench, year, base_name, text, _part)
                fname = stable_filename(f"hc_{court}_{year}", f"{_bench}/{base_name}")
                with open(os.path.join(output_dir, fname), "w", encoding="utf-8") as out:
                    json.dump(rec, out, ensure_ascii=False, indent=2)
                saved_now["n"] += 1
                return True

            def should_stop() -> bool:
                return max_judgments is not None and saved_now["n"] >= max_judgments

            print(f"    {bench}: {os.path.basename(part_key)}")
            stats = process_tar_stream(f"{S3_HOST}/{part_key}", manifest, on_text,
                                       should_stop, workers=workers)
            print(f"      saved {stats['saved']} | no text {stats['skipped_short']} | "
                  f"already had {stats['skipped_seen']}")

    return saved_now["n"]


def scrape_high_courts(courts: Optional[list] = None, years: Optional[list] = None,
                       output_dir: str = "data/raw/hc_judgments",
                       max_per_court_year: Optional[int] = 200, workers: int = 4) -> int:
    courts = courts or list(HIGH_COURTS)
    years = years or [2023, 2024]
    unknown = [c for c in courts if c not in HIGH_COURTS]
    if unknown:
        raise ValueError(f"Unknown court codes: {unknown}. Known: {sorted(HIGH_COURTS)}")

    cap = "all" if max_per_court_year is None else max_per_court_year
    print(f"\n{'='*55}\nAWS Open Data: Indian High Court Judgments (CC BY 4.0)")
    print(f"Courts: {len(courts)} | Years: {years[0]}–{years[-1]} | Max per court-year: {cap}\n{'='*55}")

    total = 0
    for court in courts:
        for year in years:
            print(f"\n── {HIGH_COURTS[court]} ({court}) · {year} ──")
            total += download_court_year(court, year, output_dir, max_per_court_year, workers)
    print(f"\nTOTAL saved this run: {total}\nPath: {os.path.abspath(output_dir)}")
    return total


def _years(spec: str) -> list:
    out: list = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return sorted(set(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download High Court judgments (AWS Open Data)")
    ap.add_argument("--courts", nargs="+", default=["all"],
                    help="court codes, e.g. 27_1 7_26 (Bombay, Delhi), or 'all'")
    ap.add_argument("--years", default="2023-2024")
    ap.add_argument("--max-per-court-year", type=int, default=200, help="0 = no cap")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="data/raw/hc_judgments")
    ap.add_argument("--list-courts", action="store_true")
    a = ap.parse_args()

    if a.list_courts:
        for code, name in sorted(HIGH_COURTS.items(), key=lambda kv: kv[1]):
            print(f"{code:7s} {name}")
    else:
        scrape_high_courts(
            None if a.courts == ["all"] else a.courts, _years(a.years), a.out,
            a.max_per_court_year or None, a.workers)
