"""
run_all_scraper.py
Runs all scrapers in sequence.
judis_scraper.py has been removed — commonLIT handles SC judgments directly.
"""

from scraper.commonLIT_scraper import scrape_commonlii
from scraper.highcourt_scraper import scrape_high_courts
from scraper.Legislative_scraper import download_all_acts
from scraper.Devgan_scraper import scrape_devgan_act
from scraper.CAD_scaper import scrape_cad
from cleaner.PDF_text_extracter import extract_all_pdfs


def run_mvp_scrape():
    """
    MVP scrape — enough data to test the full pipeline.
    Estimated time: 30-60 mins depending on internet speed.
    """

    # ── PHASE 1: Acts from legislative.gov.in / indiacode ────
    print("\n" + "="*60)
    print("PHASE 1: Download Acts (Legislative.gov.in)")
    print("="*60)
    download_all_acts()
    extract_all_pdfs()

    # ── PHASE 2: Act text from Devgan.in ─────────────────────
    print("\n" + "="*60)
    print("PHASE 2: Scrape Acts text (Devgan.in)")
    print("="*60)
    scrape_devgan_act("bns")     # Bharatiya Nyaya Sanhita
    scrape_devgan_act("bnss")    # Bharatiya Nagarik Suraksha Sanhita
    scrape_devgan_act("bsa")     # Bharatiya Sakshya Adhiniyam
    scrape_devgan_act("ipc")     # IPC (cited in old judgments)
    scrape_devgan_act("crpc")    # CrPC (cited in old judgments)

    # ── PHASE 3: SC Judgments from AWS Open Data ─────────────
    print("\n" + "="*60)
    print("PHASE 3: Scrape SC Judgments (AWS Open Data)")
    print("="*60)
    scrape_commonlii(
        years=[2022, 2023, 2024],
        max_per_court_year=50
    )

    # ── PHASE 4: CAD from constitutionofindia.net ─────────────
    print("\n" + "="*60)
    print("PHASE 4: Scrape CAD (constitutionofindia.net)")
    print("="*60)
    scrape_cad(max_sessions=5)

    print("\n" + "="*60)
    print("✅ MVP scrape complete!")
    print("Next step: run embeddings/batch_ingest.py")
    print("="*60)


def run_full_scrape():
    """
    Full scrape — run after MVP works fine.
    Warning: SC judgments alone can be several GBs.
    """
    download_all_acts()
    extract_all_pdfs()

    for act in ["bns", "bnss", "bsa", "ipc", "crpc", "evidence"]:
        scrape_devgan_act(act)

    scrape_commonlii(
        years=list(range(2000, 2025)),
        max_per_court_year=200
    )

    scrape_cad()


def run_scale_scrape():
    """
    Scale-up scrape: every SC judgment 1950-2025 (50,000+) plus High Courts.
    Downloads are resumable. Expect many GB and several hours; ingest with
    QDRANT_ON_DISK=1 QDRANT_QUANTIZE=1 (see README "Scaling").
    """
    scrape_commonlii(years=list(range(1950, 2026)),
                     max_per_court_year=None, workers=4)
    scrape_high_courts(years=list(range(2018, 2026)),
                       max_per_court_year=None, workers=4)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "mvp"
    {"mvp": run_mvp_scrape, "full": run_full_scrape, "scale": run_scale_scrape}[mode]()
