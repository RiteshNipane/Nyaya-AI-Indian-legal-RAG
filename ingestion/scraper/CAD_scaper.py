import requests
from bs4 import BeautifulSoup
import json
import time
import os
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

BASE = "https://www.constitutionofindia.net"


def get_existing_sessions(output_dir: str) -> set:
    """Return set of already-scraped session filenames."""
    if not os.path.exists(output_dir):
        return set()
    return {f for f in os.listdir(output_dir) if f.endswith(".json")}


def already_downloaded(output_dir: str, max_sessions: int = None) -> bool:
    """
    Check if CAD is already fully downloaded.
    - Full scrape: checks if we have >= 100 sessions (all 12 volumes)
    - Limited scrape: checks if we have >= max_sessions * 12 * 0.9
    """
    existing = get_existing_sessions(output_dir)
    count = len(existing)

    if count == 0:
        return False

    if max_sessions:
        expected = max_sessions * 12  # 12 volumes
        if count >= expected * 0.9:
            print(f"✓ CAD already downloaded ({count} sessions) — skipping")
            return True
        print(f"  Found {count}/{expected} CAD sessions — resuming")
        return False
    else:
        if count >= 100:  # reasonable threshold for full scrape
            print(f"✓ CAD already downloaded ({count} sessions) — skipping")
            return True
        print(f"  Found {count} CAD sessions — resuming")
        return False


def get_cad_volume_links() -> list:
    """Get links to all CAD volumes."""
    url = f"{BASE}/constitution_assembly_debates/"
    response = requests.get(url, headers=HEADERS, timeout=30)
    soup = BeautifulSoup(response.text, "lxml")

    volume_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        if "volume" in text.lower() or "vol" in href.lower():
            full_url = BASE + href if href.startswith("/") else href
            volume_links.append({"url": full_url, "title": text})

    return volume_links


def get_session_links(volume_url: str) -> list:
    """Get individual session/day links from a volume page."""
    response = requests.get(volume_url, headers=HEADERS, timeout=30)
    soup = BeautifulSoup(response.text, "lxml")

    sessions = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        if any(char.isdigit() for char in text) and len(text) > 5:
            full_url = BASE + href if href.startswith("/") else href
            sessions.append({"url": full_url, "title": text})

    return sessions


def scrape_session(session_url: str) -> dict:
    """Scrape a single CAD session."""
    response = requests.get(session_url, headers=HEADERS, timeout=30)
    soup = BeautifulSoup(response.text, "lxml")

    for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()

    title = soup.find("h1")
    date = title.get_text(strip=True) if title else ""

    content = soup.find("main") or soup.find(
        "div", {"class": "debate"}) or soup.body
    text = content.get_text(separator="\n", strip=True) if content else ""

    return {
        "url":      session_url,
        "date":     date,
        "text":     text,
        "doc_type": "cad",
        "source":   "constitutionofindia.net"
    }


def scrape_cad(output_dir: str = "data/raw/cad", max_sessions: int = None):
    """Scrape all CAD volumes and sessions."""

    os.makedirs(output_dir, exist_ok=True)

    # ── Skip if already downloaded ────────────────────────────
    if already_downloaded(output_dir, max_sessions):
        return []
    # ─────────────────────────────────────────────────────────

    existing_files = get_existing_sessions(output_dir)

    print("Getting CAD volume links...")
    volumes = get_cad_volume_links()
    print(f"Found {len(volumes)} volumes")

    all_sessions = []

    for vol_num, volume in enumerate(volumes, 1):
        print(f"\nVolume {vol_num}: {volume['title']}")
        sessions = get_session_links(volume["url"])
        print(f"  {len(sessions)} sessions")

        if max_sessions:
            sessions = sessions[:max_sessions]

        for session in tqdm(sessions):
            try:
                # Build filename the same way as before
                filename = f"cad_vol{vol_num:02d}_{session['title'][:30]}.json"
                filename = "".join(
                    c for c in filename if c.isalnum() or c in "._-")

                # ── Skip already scraped sessions ─────────────
                if filename in existing_files:
                    continue
                # ─────────────────────────────────────────────

                data = scrape_session(session["url"])
                data["volume"] = vol_num

                with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                all_sessions.append(data)
                time.sleep(0.8)

            except Exception as e:
                print(f"  Error: {e}")
                continue

    print(f"\nTotal CAD sessions scraped: {len(all_sessions)}")
    return all_sessions


if __name__ == "__main__":
    scrape_cad(max_sessions=3)
