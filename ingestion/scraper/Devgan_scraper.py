"""
Devgan_scraper.py — FIXED VERSION

Key fix: Devgan.in uses a specific URL pattern:
  - Index page: https://devgan.in/all_sections_bns.php
  - Section page: https://devgan.in/bns/section/1/

We generate section URLs directly from known section counts
instead of crawling index pages (which was failing before).
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,*/*",
}

# Exact section counts for each act (verified)
ACT_CONFIG = {
    "bns":          {"prefix": "bns",          "total_sections": 358,  "name": "Bharatiya Nyaya Sanhita 2023"},
    "bnss":         {"prefix": "bnss",         "total_sections": 531,  "name": "Bharatiya Nagarik Suraksha Sanhita 2023"},
    "bsa":          {"prefix": "bsa",          "total_sections": 170,  "name": "Bharatiya Sakshya Adhiniyam 2023"},
    "ipc":          {"prefix": "ipc",          "total_sections": 511,  "name": "Indian Penal Code 1860"},
    "crpc":         {"prefix": "crpc",         "total_sections": 484,  "name": "Code of Criminal Procedure 1973"},
    "evidence":     {"prefix": "evidence",     "total_sections": 167,  "name": "Indian Evidence Act 1872"},
}

# URL pattern: https://devgan.in/{prefix}/section/{n}/
SECTION_URL = "https://devgan.in/{prefix}/section/{n}/"


def scrape_section(url: str) -> str:
    """Scrape text from a single Devgan section page."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "lxml")

        # Remove nav/footer/header junk
        for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside"]):
            tag.decompose()

        # Devgan content is in <div class="entry-content"> or similar
        content = (
            soup.find("div", {"class": "entry-content"}) or
            soup.find("div", {"class": "content"}) or
            soup.find("article") or
            soup.find("main") or
            soup.find("body")
        )
        return content.get_text(separator="\n", strip=True) if content else ""
    except Exception:
        return ""


def scrape_act(act_key: str, output_dir: str = "data/raw/acts", delay: float = 0.4) -> list:
    """
    Scrape all sections of an act by iterating section numbers directly.
    Much more reliable than crawling index pages.
    """
    config = ACT_CONFIG.get(act_key)
    if not config:
        print(f"Unknown act: {act_key}. Available: {list(ACT_CONFIG.keys())}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{act_key}_devgan.json")

    # Resume support — load already scraped sections
    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            existing = json.load(f)
        if len(existing) >= config["total_sections"] * 0.9:  # 90% done = skip
            print(f"  Already complete: {act_key} ({len(existing)} sections)")
            return existing
        print(f"  Resuming {act_key}: {len(existing)} sections already done")
        done_sections = {s["section_number"] for s in existing}
    else:
        existing = []
        done_sections = set()

    prefix = config["prefix"]
    total = config["total_sections"]
    act_name = config["name"]

    print(f"\n  Scraping: {act_name}")
    print(f"  Sections: 1 to {total}")
    print(f"  URL pattern: https://devgan.in/{prefix}/section/N/")

    sections = list(existing)
    failed = []

    for n in tqdm(range(1, total + 1), desc=f"  {act_key}"):
        if n in done_sections:
            continue

        url = SECTION_URL.format(prefix=prefix, n=n)
        text = scrape_section(url)

        if len(text) > 50:
            sections.append({
                "act":            act_key,
                "act_name":       act_name,
                "section_number": n,
                "url":            url,
                "text":           text,
                "doc_type":       "act",
                "source":         "devgan.in"
            })
        else:
            failed.append(n)

        # Save every 50 sections so we don't lose progress
        if n % 50 == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)

        time.sleep(delay)

    # Final save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Saved {len(sections)} sections → {output_file}")
    if failed:
        print(
            f"  ✗ Failed sections (likely don't exist): {failed[:10]}{'...' if len(failed) > 10 else ''}")

    return sections


def scrape_all_acts(output_dir: str = "data/raw/acts", acts: list = None):
    """Scrape all or specified acts from Devgan.in"""

    if acts is None:
        acts = list(ACT_CONFIG.keys())

    os.makedirs(output_dir, exist_ok=True)
    summary = {}

    print("\n" + "="*55)
    print("Devgan.in Scraper — Section-by-section")
    print("="*55)

    for act_key in acts:
        result = scrape_act(act_key, output_dir)
        summary[act_key] = len(result)
        time.sleep(2)  # Pause between acts

    print("\n" + "="*55)
    print("DEVGAN SCRAPE SUMMARY")
    print("="*55)
    for act, count in summary.items():
        expected = ACT_CONFIG[act]["total_sections"]
        status = "✓" if count >= expected * 0.9 else "⚠"
        print(f"  {status} {act:12s}: {count}/{expected} sections")

    return summary


if __name__ == "__main__":
    # Scrape just the new laws first (fastest, most relevant for RAG)
    scrape_all_acts(acts=["bns", "bnss", "bsa"])

    # Then legacy acts (still cited in old SC judgments)
    # scrape_all_acts(acts=["ipc", "crpc", "evidence"])
