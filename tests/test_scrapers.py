import io
import json
import os
import tarfile

import pytest

from scraper import commonLIT_scraper as sc
from scraper import highcourt_scraper as hc
from scraper import pdf_utils as pu

LONG = "x" * 300


def make_tar(members: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeResp:
    def __init__(self, data: bytes):
        self.raw = io.BytesIO(data)
        self.closed = False

    def close(self):
        self.closed = True


def opener_for(members):
    data = make_tar(members)
    return lambda url: FakeResp(data)


def run(members, tmp_path, workers=1, should_stop=lambda: False, seen=None, extractor=bytes.decode):
    seen = seen or pu.SeenManifest(str(tmp_path / "seen.txt"))
    got = []
    stats = pu.process_tar_stream(
        "http://x/t.tar", seen, lambda n, t: got.append(n) or True, should_stop,
        workers=workers, extractor=extractor, opener=opener_for(members), log=lambda m: None)
    return stats, got, seen


def test_stream_saves_pdfs_only_and_records_seen(tmp_path):
    stats, got, seen = run({"d/a.pdf": LONG.encode(), "d/b.PDF": LONG.encode(),
                            "d/notes.txt": LONG.encode()}, tmp_path)
    assert got == ["a.pdf", "b.PDF"] and stats["saved"] == 2
    assert "a.pdf" in seen and "notes.txt" not in seen


def test_short_text_is_skipped_but_remembered(tmp_path):
    stats, got, seen = run({"a.pdf": b"tiny", "b.pdf": LONG.encode()}, tmp_path)
    assert got == ["b.pdf"] and stats["skipped_short"] == 1
    assert "a.pdf" in seen  # not retried next run


def test_resume_skips_seen_members(tmp_path):
    seen = pu.SeenManifest(str(tmp_path / "seen.txt"))
    seen.add("a.pdf")
    stats, got, _ = run({"a.pdf": LONG.encode(), "b.pdf": LONG.encode()}, tmp_path, seen=seen)
    assert got == ["b.pdf"] and stats["skipped_seen"] == 1


def test_manifest_persists_across_instances(tmp_path):
    p = str(tmp_path / "m.txt")
    m = pu.SeenManifest(p)
    m.add("one.pdf"); m.add("one.pdf"); m.add("two.pdf")
    assert open(p).read().splitlines() == ["one.pdf", "two.pdf"]
    assert "two.pdf" in pu.SeenManifest(p)


def test_manifest_bootstrap_only_when_file_missing(tmp_path):
    p = str(tmp_path / "m.txt")
    m = pu.SeenManifest(p, bootstrap=lambda: ["old1.pdf", "old2.pdf"])
    assert "old1.pdf" in m
    again = pu.SeenManifest(p, bootstrap=lambda: ["should-not-be-used.pdf"])
    assert "should-not-be-used.pdf" not in again and "old2.pdf" in again


def test_should_stop_caps_saved_records(tmp_path):
    members = {f"{i}.pdf": LONG.encode() for i in range(10)}
    count = {"n": 0}
    seen = pu.SeenManifest(str(tmp_path / "s.txt"))

    def on_text(n, t):
        count["n"] += 1
        return True

    stats = pu.process_tar_stream("u", seen, on_text, lambda: count["n"] >= 3, workers=1,
                                  extractor=bytes.decode, opener=opener_for(members),
                                  log=lambda m: None)
    assert count["n"] == 3 and stats["saved"] == 3


def test_process_pool_path_matches_inline(tmp_path):
    members = {f"{i}.pdf": LONG.encode() for i in range(6)}
    stats, got, _ = run(members, tmp_path, workers=2)
    assert sorted(got) == sorted(members) and stats["saved"] == 6


def test_stream_failure_retries_then_reports_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(pu.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def bad_opener(url):
        calls["n"] += 1
        raise ConnectionError("reset")

    stats = pu.process_tar_stream("u", pu.SeenManifest(str(tmp_path / "s.txt")),
                                  lambda n, t: True, opener=bad_opener, retries=3,
                                  log=lambda m: None)
    assert calls["n"] == 3 and stats.get("failed") is True


def test_stable_filename_is_deterministic_and_prefixed():
    a = pu.stable_filename("sc_2023", "x.pdf")
    assert a == pu.stable_filename("sc_2023", "x.pdf")
    assert a != pu.stable_filename("sc_2023", "y.pdf")
    assert a.startswith("sc_2023_") and a.endswith(".json")


# ── Supreme Court scraper ────────────────────────────────

def test_parse_years():
    assert sc.parse_years("2018-2020") == [2018, 2019, 2020]
    assert sc.parse_years("2019, 2021,2019") == [2019, 2021]


def fake_stream(items):
    """Stand-in for process_tar_stream: feeds `items` and honours seen/should_stop."""
    captured = {}

    def _fake(url, seen, on_text, should_stop=lambda: False, **kw):
        captured["seen"] = seen
        stats = {"saved": 0, "skipped_short": 0, "skipped_seen": 0, "errors": 0}
        for name, text in items:
            if should_stop():
                break
            if name in seen:
                stats["skipped_seen"] += 1
                continue
            if on_text(name, text):
                stats["saved"] += 1
            seen.add(name)
        return stats

    return _fake, captured


def test_sc_download_year_unlimited_and_idempotent(tmp_path, monkeypatch):
    out = str(tmp_path)
    items = [(f"j{i}.pdf", LONG) for i in range(5)]
    fake, _ = fake_stream(items)
    monkeypatch.setattr(sc, "process_tar_stream", fake)
    monkeypatch.setattr(sc, "get_metadata_index", lambda y: {"j0.pdf": {"title": "A v. B"}})

    assert sc.download_year(2023, out, max_judgments=None) == 5
    first = sorted(f for f in os.listdir(out) if f.endswith(".json"))
    rec = json.load(open(os.path.join(out, [f for f in first if True][0])))
    assert rec["doc_type"] == "judgment" and rec["court"] == "Supreme Court of India"

    assert sc.download_year(2023, out, max_judgments=None) == 5     # re-run: no duplicates
    assert sorted(f for f in os.listdir(out) if f.endswith(".json")) == first
    titles = {json.load(open(os.path.join(out, f)))["title"] for f in first}
    assert "A v. B" in titles


def test_sc_cap_counts_existing_files_and_bootstraps_manifest(tmp_path, monkeypatch):
    out = str(tmp_path)
    old = {"title": "old", "filename": "old.pdf", "text": LONG}
    json.dump(old, open(os.path.join(out, "sc_2023_0000.json"), "w"))
    fake, captured = fake_stream([(f"n{i}.pdf", LONG) for i in range(10)])
    monkeypatch.setattr(sc, "process_tar_stream", fake)
    monkeypatch.setattr(sc, "get_metadata_index", lambda y: {})

    total = sc.download_year(2023, out, max_judgments=4)   # 1 existing -> budget of 3
    assert total == 4
    assert "old.pdf" in captured["seen"]


def test_sc_already_downloaded_threshold(tmp_path):
    for i in range(9):
        open(tmp_path / f"sc_2020_{i:04d}.json", "w").write("{}")
    assert sc.already_downloaded(2020, str(tmp_path), 10) is True
    assert sc.already_downloaded(2020, str(tmp_path), 100) is False
    assert sc.already_downloaded(2020, str(tmp_path), None) is False


# ── High Court scraper ───────────────────────────────────

LISTING = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Prefix>data/tar/year=2024/court=27_1/</Prefix>
  <IsTruncated>{trunc}</IsTruncated>
  {token}
  <CommonPrefixes><Prefix>data/tar/year=2024/court=27_1/bench={b}/</Prefix></CommonPrefixes>
  <Contents><Key>data/tar/year=2024/court=27_1/bench={b}/part-0.tar</Key></Contents>
</ListBucketResult>"""


def test_parse_listing():
    xml = LISTING.format(trunc="true", token="<NextContinuationToken>T1</NextContinuationToken>", b="hcaurdb")
    prefixes, keys, token = hc.parse_listing(xml)
    assert prefixes == ["data/tar/year=2024/court=27_1/bench=hcaurdb/"]
    assert keys == ["data/tar/year=2024/court=27_1/bench=hcaurdb/part-0.tar"]
    assert token == "T1"
    assert hc.parse_listing(LISTING.format(trunc="false", token="", b="x"))[2] is None


class FakeHttp:
    def __init__(self, pages):
        self.pages, self.calls = list(pages), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(dict(params))

        class R:
            text = self.pages.pop(0)

            def raise_for_status(self):
                pass

        return R()


def test_list_s3_follows_pagination():
    http = FakeHttp([
        LISTING.format(trunc="true", token="<NextContinuationToken>T1</NextContinuationToken>", b="a"),
        LISTING.format(trunc="false", token="", b="b"),
    ])
    prefixes, keys = hc.list_s3("data/tar/year=2024/court=27_1/", session=http)
    assert len(prefixes) == 2 and len(keys) == 2
    assert http.calls[1]["continuation-token"] == "T1" and http.calls[0]["delimiter"] == "/"


def test_list_benches_extracts_bench_codes():
    http = FakeHttp([LISTING.format(trunc="false", token="", b="hcaurdb")])
    assert hc.list_benches(2024, "27_1", session=http) == ["hcaurdb"]


def test_court_table_covers_25_courts_and_known_codes():
    assert len(hc.HIGH_COURTS) == 25
    assert hc.HIGH_COURTS["27_1"] == "Bombay High Court"
    assert hc.HIGH_COURTS["7_26"] == "High Court of Delhi"


def test_guess_title():
    text = "IN THE HIGH COURT\n\nRAMESH KUMAR  vs.  STATE OF MAHARASHTRA\n\nJudgment..."
    assert hc.guess_title(text, "f.pdf") == "RAMESH KUMAR vs. STATE OF MAHARASHTRA"
    assert hc.guess_title("no parties here", "f.pdf") == "f.pdf"


def test_build_record_schema_matches_ingestion_expectations():
    rec = hc.build_record("27_1", "hcaurdb", 2024, "j.pdf", "A vs. B\n" + LONG, "k.tar")
    assert rec["doc_type"] == "judgment" and rec["court"] == "Bombay High Court"
    assert rec["court_level"] == "high" and rec["year"] == 2024 and rec["text"]


def test_download_court_year_respects_cap_across_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "list_benches", lambda y, c: ["b1", "b2"])
    monkeypatch.setattr(hc, "list_tar_parts", lambda y, c, b: [f"data/tar/{b}/p0.tar", f"data/tar/{b}/p1.tar"])
    fake, _ = fake_stream([(f"j{i}.pdf", "A vs. B " + LONG) for i in range(4)])
    monkeypatch.setattr(hc, "process_tar_stream", fake)

    saved = hc.download_court_year("27_1", 2024, str(tmp_path), max_judgments=3)
    assert saved == 3
    files = [f for f in os.listdir(tmp_path) if f.endswith(".json")]
    assert len(files) == 3 and all(f.startswith("hc_27_1_2024_") for f in files)


def test_download_court_year_handles_empty_listing(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "list_benches", lambda y, c: [])
    assert hc.download_court_year("27_1", 1999, str(tmp_path)) == 0


def test_scrape_high_courts_rejects_unknown_code(tmp_path):
    with pytest.raises(ValueError, match="Unknown court codes"):
        hc.scrape_high_courts(courts=["99_99"], output_dir=str(tmp_path))
