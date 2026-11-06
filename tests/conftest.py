import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# scrapers live in ingestion/scraper and import each other as `scraper.pdf_utils`
for _p in (os.path.join(ROOT, "ingestion"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
