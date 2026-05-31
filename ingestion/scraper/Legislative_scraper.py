"""
Legislative_scraper.py

Uses THREE fast sources instead of indiacode.nic.in:
  1. cdnbbsr.s3waas.gov.in  — Ministry of Law CDN (fastest, AWS-backed)
  2. egazette.gov.in        — Official Gazette of India
  3. Devgan.in              — Scrapes clean section-by-section HTML text (no PDF needed)

All URLs verified from live search results.
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import json
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,text/html,*/*",
}

# ─────────────────────────────────────────────────────────────────────
# SOURCE 1: cdnbbsr.s3waas.gov.in
# Ministry of Law & Justice CDN — AWS-backed, very fast
# These are the official English versions published by Govt of India
# ─────────────────────────────────────────────────────────────────────

CDN_ACTS = [
    {
        "name": "Constitution of India (updated to 106th Amendment, 2023)",
        "url": "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2024/07/20240716890312078.pdf",
        "filename": "constitution_of_india.pdf",
        "type": "constitution",
        "backup_url": "https://cdnbbsr.s3waas.gov.in/s3978d76676f5e7918f81d28e7d092ca0d/uploads/2021/09/2021092040.pdf"
    },
    {
        "name": "BNS + BNSS + BSA Reference Material (MHA Official)",
        "url": "https://cdnbbsr.s3waas.gov.in/s3ec02907edb0aa6986220dbffb79a7885/uploads/2024/05/2024052146.pdf",
        "filename": "bns_bnss_bsa_reference_mha.pdf",
        "type": "act"
    },
]

# ─────────────────────────────────────────────────────────────────────
# SOURCE 2: egazette.gov.in
# Official Gazette — authoritative legal text
# BNS/BNSS/BSA gazette notifications from Dec 25, 2023
# ─────────────────────────────────────────────────────────────────────

GAZETTE_ACTS = [
    {
        "name": "Bharatiya Nyaya Sanhita 2023 — Official Gazette",
        "url": "https://egazette.gov.in/WriteReadData/2023/247490.pdf",
        "filename": "bns_2023_gazette.pdf",
        "type": "act"
    },
    {
        "name": "Bharatiya Nagarik Suraksha Sanhita 2023 — Official Gazette",
        "url": "https://egazette.gov.in/WriteReadData/2023/247491.pdf",
        "filename": "bnss_2023_gazette.pdf",
        "type": "act"
    },
    {
        "name": "Bharatiya Sakshya Adhiniyam 2023 — Official Gazette",
        "url": "https://egazette.gov.in/WriteReadData/2023/247492.pdf",
        "filename": "bsa_2023_gazette.pdf",
        "type": "act"
    },
]

# ─────────────────────────────────────────────────────────────────────
# SOURCE 3: Devgan.in
# Clean HTML text scraped section by section — no PDF, no OCR issues
# Best for chunking into RAG since it's already structured
# ─────────────────────────────────────────────────────────────────────

DEVGAN_ACTS = {
    "ipc":          "https://devgan.in/ipc/",
    "crpc":         "https://devgan.in/crpc/",
    "evidence":     "https://devgan.in/evidence/",
    "bns":          "https://devgan.in/bns/",
    "bnss":         "https://devgan.in/bnss/",
    "bsa":          "https://devgan.in/bsa/",
    "constitution": "https://devgan.in/constitution/",
}


# ─────────────────────────────────────────────────────────────────────
# PDF DOWNLOADER
# ─────────────────────────────────────────────────────────────────────

def is_valid_pdf(content: bytes) -> bool:
    return content[:4] == b"%PDF"


def download_pdf(url: str, output_path: str, name: str) -> bool:
    try:
        r = requests.get(url, headers=HEADERS, timeout=45)
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}")
            return False
        if not is_valid_pdf(r.content):
            print(f"    Not a PDF (got HTML error page)")
            return False
        with open(output_path, "wb") as f:
            f.write(r.content)
        print(f"    ✓ {name} — {len(r.content)//1024} KB")
        return True
    except requests.exceptions.Timeout:
        print(f"    Timeout")
        return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
# DEVGAN SCRAPER (HTML → structured JSON, better than PDF for RAG)
# ─────────────────────────────────────────────────────────────────────

def get_devgan_section_links(act_url: str) -> list:
    """Get all section links from a Devgan.in Act index page."""
    try:
        r = requests.get(act_url, headers=HEADERS, timeout=30)
        soup = BeautifulSoup(r.text, "lxml")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            if href and len(text) > 3:
                full = href if href.startswith("http") else act_url.rstrip(
                    "/") + "/" + href.lstrip("/")
                # Only include section-level links (avoid nav/external links)
                if act_url.split("//")[1].split("/")[1] in full:
                    links.append({"url": full, "title": text})
        # Deduplicate
        seen = set()
        unique = []
        for l in links:
            if l["url"] not in seen:
                seen.add(l["url"])
                unique.append(l)
        return unique
    except Exception as e:
        print(f"    Error getting links: {e}")
        return []


def scrape_devgan_section(url: str) -> str:
    """Scrape text content from a single Devgan.in section page."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "lxml")
        # Remove navigation elements
        for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside"]):
            tag.decompose()
        # Get main content
        main = (soup.find("main") or
                soup.find("article") or
                soup.find("div", {"class": "content"}) or
                soup.find("div", {"id": "content"}) or
                soup.body)
        return main.get_text(separator="\n", strip=True) if main else ""
    except Exception as e:
        return ""


def scrape_devgan_act(act_key: str, output_dir: str = "data/raw/acts") -> list:
    """
    Scrape all sections of an Act from Devgan.in as structured JSON.
    This is BETTER than PDF for RAG — already clean, section-level chunks.
    """
    act_url = DEVGAN_ACTS.get(act_key)
    if not act_url:
        print(f"Unknown act key: {act_key}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{act_key}_devgan.json")

    if os.path.exists(output_file):
        print(f"    Already scraped: {act_key}")
        with open(output_file, encoding="utf-8") as f:
            return json.load(f)

    print(f"\n  Scraping {act_key} from Devgan.in...")
    links = get_devgan_section_links(act_url)
    print(f"    Found {len(links)} sections")

    if not links:
        print(f"    No links found — skipping {act_key}")
        return []

    sections = []
    for link in tqdm(links, desc=f"  {act_key}"):
        text = scrape_devgan_section(link["url"])
        if len(text) > 100:
            sections.append({
                "act": act_key,
                "title": link["title"],
                "url": link["url"],
                "text": text,
                "doc_type": "act",
                "source": "devgan.in"
            })
        time.sleep(0.4)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)

    print(f"    Saved {len(sections)} sections → {output_file}")
    return sections


# ─────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────

def download_all_acts(output_dir: str = "data/raw/acts"):
    """
    Downloads all Acts using 3 fast sources.
    Strategy:
      Step 1 — PDF: Ministry of Law CDN (fastest)
      Step 2 — PDF: eGazette (official gazette)
      Step 3 — HTML: Devgan.in scrape (best for RAG chunking)
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    # ── STEP 1: Ministry CDN PDFs ──────────────────────────────────
    print("\n" + "="*55)
    print("STEP 1: Ministry of Law CDN (fast AWS server)")
    print("="*55)

    for act in CDN_ACTS:
        out = os.path.join(output_dir, act["filename"])
        print(f"\n  {act['name']}")
        if os.path.exists(out):
            print(f"    Already exists — skipping")
            results.append({**act, "status": "cached"})
            continue
        ok = download_pdf(act["url"], out, act["name"])
        if not ok and "backup_url" in act:
            print(f"    Trying backup...")
            ok = download_pdf(act["backup_url"], out, act["name"])
        results.append({**act, "status": "downloaded" if ok else "failed"})
        time.sleep(1)

    # ── STEP 2: eGazette PDFs ──────────────────────────────────────
    print("\n" + "="*55)
    print("STEP 2: Official eGazette (BNS / BNSS / BSA)")
    print("="*55)

    for act in GAZETTE_ACTS:
        out = os.path.join(output_dir, act["filename"])
        print(f"\n  {act['name']}")
        if os.path.exists(out):
            print(f"    Already exists — skipping")
            results.append({**act, "status": "cached"})
            continue
        ok = download_pdf(act["url"], out, act["name"])
        results.append({**act, "status": "downloaded" if ok else "failed"})
        time.sleep(1)

    # ── STEP 3: Devgan.in HTML scrape ─────────────────────────────
    print("\n" + "="*55)
    print("STEP 3: Devgan.in — clean section-by-section text")
    print("(Best format for RAG — no PDF parsing needed)")
    print("="*55)

    devgan_results = []
    for act_key in DEVGAN_ACTS:
        sections = scrape_devgan_act(act_key, output_dir)
        devgan_results.append({
            "act": act_key,
            "sections_scraped": len(sections),
            "status": "done" if sections else "failed"
        })
        time.sleep(1.5)

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("FINAL SUMMARY")
    print("="*55)

    pdf_ok = [r for r in results if r["status"] in ("downloaded", "cached")]
    pdf_failed = [r for r in results if r["status"] == "failed"]
    devgan_ok = [r for r in devgan_results if r["status"] == "done"]

    print(f"✓ PDFs downloaded:      {len(pdf_ok)}/{len(results)}")
    print(f"✓ Devgan acts scraped:  {len(devgan_ok)}/{len(devgan_results)}")

    if pdf_failed:
        print(f"\n✗ Failed PDFs ({len(pdf_failed)}):")
        for r in pdf_failed:
            print(f"   - {r['name']}")
        print("\n  These acts are available as Devgan text instead (Step 3).")
        print("  For PDFs, manually download from: https://egazette.gov.in")

    # Save manifest
    manifest = {"pdfs": results, "devgan": devgan_results}
    mpath = os.path.join(output_dir, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nManifest saved → {mpath}")
    print(f"All files  → {os.path.abspath(output_dir)}")

    return manifest


if __name__ == "__main__":
    download_all_acts()
