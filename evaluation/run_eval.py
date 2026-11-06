"""
evaluation/run_eval.py

Evaluate the NyayaAI retrieval pipeline against a golden set and compare
pipeline variants (an ablation of reranker and query routing).

Examples
--------
# Offline: check query-classifier routing accuracy (no Qdrant / models needed)
python -m evaluation.run_eval --classifier-only

# Full retrieval ablation (needs Qdrant running + embeddings ingested)
python -m evaluation.run_eval --ks 1 3 5 10 --candidate-k 30

# Include answer quality (citations + keyword coverage), and LLM-judge scores
python -m evaluation.run_eval --generation --judge

Variants: baseline (dense only) | rerank | route | rerank+route
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.metrics import (  # noqa: E402
    aggregate, is_relevant, percentile, query_metrics, relevance_flags,
)
from retrieval.query_classifier import QueryClassifier  # noqa: E402

VARIANTS = {
    "baseline":     {"rerank": False, "route": False},
    "rerank":       {"rerank": True,  "route": False},
    "route":        {"rerank": False, "route": True},
    "rerank+route": {"rerank": True,  "route": True},
}

DEFAULT_GOLDEN = os.path.join(ROOT, "evaluation", "golden_set.jsonl")


def load_golden(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON ({e})") from e
            for key in ("id", "query", "relevant"):
                if key not in row:
                    raise ValueError(f"{path}:{n}: missing '{key}'")
            rows.append(row)
    return rows


def query_text(entry: dict, use_english: bool) -> str:
    """The query sent to the pipeline. Multilingual rows carry an English gloss."""
    if use_english and entry.get("query_en"):
        return entry["query_en"]
    return entry["query"]


# ─────────────────────────────────────────────────────────
# Classifier accuracy (offline)
# ─────────────────────────────────────────────────────────

def classifier_report(golden: list, classifier: QueryClassifier = None) -> dict:
    classifier = classifier or QueryClassifier()
    rows, correct, total = [], 0, 0
    for e in golden:
        if "expected_doc_types" not in e:
            continue
        got = classifier.classify(query_text(e, use_english=True)).doc_types
        want = e["expected_doc_types"]
        ok = (sorted(got) if got else None) == (sorted(want) if want else None)
        total += 1
        correct += ok
        rows.append({"id": e["id"], "expected": want, "got": got, "correct": ok})
    return {
        "accuracy": (correct / total) if total else None,
        "correct": correct, "total": total,
        "mismatches": [r for r in rows if not r["correct"]],
    }


# ─────────────────────────────────────────────────────────
# Retrieval ablation
# ─────────────────────────────────────────────────────────

def evaluate_retrieval(pipeline, golden: list, variants: list, ks: list,
                       use_english: bool = False) -> dict:
    top_k = max(ks)

    # 1. run every variant on every query, keep the ranked lists and pools
    runs: dict = {v: {} for v in variants}
    for e in golden:
        q = query_text(e, use_english)
        for v in variants:
            flags = VARIANTS[v]
            runs[v][e["id"]] = pipeline.retrieve(
                q, top_k=top_k, rerank=flags["rerank"], route=flags["route"],
                language=e.get("lang") if not use_english else None,
            )

    # 2. pooled relevance: relevant chunk ids found by ANY variant for the query
    pooled: dict = {}
    for e in golden:
        ids: set = set()
        for v in variants:
            res = runs[v][e["id"]]
            for c in list(res.pool) + list(res.candidates):
                if is_relevant(c.payload, e["relevant"]):
                    ids.add(c.id)
        pooled[e["id"]] = len(ids)

    # 3. metrics per variant
    report: dict = {}
    for v in variants:
        per_query, latencies = [], []
        for e in golden:
            res = runs[v][e["id"]]
            rels = relevance_flags(res.candidates, e["relevant"])
            m = query_metrics(rels, ks, pooled[e["id"]])
            first = next((i for i, r in enumerate(rels, 1) if r), None)
            per_query.append({
                "id": e["id"], "category": e.get("category", "other"),
                "first_relevant_rank": first,
                "routed_doc_types": res.routed_doc_types,
                "widened": res.widened, "reranker": res.reranker,
                **m,
            })
            latencies.append(res.timings_ms.get("total", 0.0))

        by_cat: dict = {}
        for cat in sorted({q["category"] for q in per_query}):
            by_cat[cat] = aggregate([q for q in per_query if q["category"] == cat])

        report[v] = {
            "overall": aggregate(per_query),
            "by_category": by_cat,
            "latency_ms": {"p50": round(percentile(latencies, 50), 1),
                           "p95": round(percentile(latencies, 95), 1)},
            "per_query": per_query,
        }
    return report


def evaluate_generation(pipeline, llm_client, golden: list, variant: str, model: str,
                        judge_model: str = None, use_english: bool = False) -> dict:
    from evaluation.generation_eval import LLMJudge, evaluate_answer
    from retrieval.generation import generate_answer

    judge = LLMJudge(llm_client, judge_model) if judge_model else None
    flags = VARIANTS[variant]
    rows = []
    for e in golden:
        q = query_text(e, use_english)
        res = pipeline.retrieve(q, top_k=8, rerank=flags["rerank"], route=flags["route"],
                                language=e.get("lang") if not use_english else None)
        try:
            answer = generate_answer(llm_client, model, q, res.candidates, res.language)
        except Exception as err:
            rows.append({"id": e["id"], "error": str(err)})
            continue
        scores = evaluate_answer(q, answer, [c.text for c in res.candidates],
                                 e.get("expected_keywords"), judge)
        rows.append({"id": e["id"], "category": e.get("category", "other"), **scores})

    scored = [r for r in rows if "error" not in r]
    numeric = [{k: v for k, v in r.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)} for r in scored]
    return {"variant": variant, "model": model,
            "overall": aggregate(numeric), "errors": len(rows) - len(scored), "per_query": rows}


# ─────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────

def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def render_markdown(report: dict, ks: list) -> str:
    head = ks[len(ks) // 2] if len(ks) > 2 else ks[-1]
    lines = [f"# NyayaAI evaluation — {report['meta']['timestamp']}", ""]
    lines.append(f"Queries: {report['meta']['n_queries']} · k = {ks} · "
                 f"candidate pool = {report['meta'].get('candidate_k', 'n/a')}")
    lines.append("")

    if "classifier" in report:
        c = report["classifier"]
        lines += ["## Query classifier routing", "",
                  f"Accuracy: **{_fmt(c['accuracy'])}** ({c['correct']}/{c['total']})", ""]
        for m in c["mismatches"]:
            lines.append(f"- `{m['id']}`: expected {m['expected']}, got {m['got']}")
        lines.append("")

    variants = report.get("variants", {})
    if variants:
        lines += [f"## Retrieval (headline k = {head})", "",
                  f"| variant | pool hit | hit@{head} | P@{head} | R@{head} | MRR@{head} | nDCG@{head} | p50 ms | p95 ms |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, r in variants.items():
            o = r["overall"]
            lines.append(
                f"| {name} | {_fmt(o.get('pool_hit'))} | {_fmt(o.get(f'hit@{head}'))} | "
                f"{_fmt(o.get(f'precision@{head}'))} | {_fmt(o.get(f'recall@{head}'))} | "
                f"{_fmt(o.get(f'mrr@{head}'))} | {_fmt(o.get(f'ndcg@{head}'))} | "
                f"{r['latency_ms']['p50']} | {r['latency_ms']['p95']} |")
        lines.append("")

        lines += ["## nDCG and recall across k", "",
                  "| variant | " + " | ".join(f"nDCG@{k}" for k in ks) + " | "
                  + " | ".join(f"R@{k}" for k in ks) + " |",
                  "|---|" + "---|" * (2 * len(ks))]
        for name, r in variants.items():
            o = r["overall"]
            lines.append(f"| {name} | " + " | ".join(_fmt(o.get(f"ndcg@{k}")) for k in ks)
                         + " | " + " | ".join(_fmt(o.get(f"recall@{k}")) for k in ks) + " |")
        lines.append("")

        if "baseline" in variants and len(variants) > 1:
            base = variants["baseline"]["overall"]
            lines += [f"## Change vs baseline (k = {head})", "",
                      f"| variant | Δ hit@{head} | Δ MRR@{head} | Δ nDCG@{head} |", "|---|---|---|---|"]
            for name, r in variants.items():
                if name == "baseline":
                    continue
                o = r["overall"]
                d = lambda key: (None if o.get(key) is None or base.get(key) is None
                                 else o[key] - base[key])
                fmt = lambda x: "n/a" if x is None else f"{x:+.3f}"
                lines.append(f"| {name} | {fmt(d(f'hit@{head}'))} | "
                             f"{fmt(d(f'mrr@{head}'))} | {fmt(d(f'ndcg@{head}'))} |")
            lines.append("")

        lines += [f"## By category (nDCG@{head})", ""]
        cats = sorted({c for r in variants.values() for c in r["by_category"]})
        lines += ["| category | " + " | ".join(variants) + " |",
                  "|---|" + "---|" * len(variants)]
        for cat in cats:
            lines.append(f"| {cat} | " + " | ".join(
                _fmt(r["by_category"].get(cat, {}).get(f"ndcg@{head}")) for r in variants.values()) + " |")
        lines.append("")

    if "generation" in report:
        g = report["generation"]
        lines += [f"## Answer quality (variant `{g['variant']}`, model `{g['model']}`)", ""]
        for k, v in g["overall"].items():
            lines.append(f"- {k}: {_fmt(v)}")
        if g["errors"]:
            lines.append(f"- generation errors: {g['errors']}")
        lines.append("")

    lines += ["> Recall and nDCG are *pooled* (relevant chunks surfaced by any variant); "
              "see evaluation/metrics.py.", ""]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--golden", default=DEFAULT_GOLDEN)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--candidate-k", type=int, default=30)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--categories", nargs="+", help="only run these golden categories")
    p.add_argument("--limit", type=int, help="only the first N queries")
    p.add_argument("--classifier-only", action="store_true",
                   help="only measure routing accuracy (offline)")
    p.add_argument("--no-translate", action="store_true",
                   help="send the English gloss (query_en) instead of the original query")
    p.add_argument("--reranker", help="auto|cohere|cross-encoder|bm25 (overrides RERANKER)")
    p.add_argument("--generation", action="store_true", help="also evaluate generated answers")
    p.add_argument("--judge", action="store_true", help="add LLM-judge scores (implies --generation)")
    p.add_argument("--gen-variant", default="rerank+route", choices=list(VARIANTS))
    p.add_argument("--model", default=os.getenv("DEFAULT_MODEL", "anthropic/claude-3.5-sonnet"))
    p.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "anthropic/claude-3.5-sonnet"))
    p.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    args = p.parse_args(argv)

    golden = load_golden(args.golden)
    if args.categories:
        golden = [g for g in golden if g.get("category") in set(args.categories)]
    if args.limit:
        golden = golden[: args.limit]
    if not golden:
        print("No golden queries selected.")
        return 1

    report: dict = {"meta": {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_queries": len(golden), "ks": args.ks, "candidate_k": args.candidate_k,
        "variants": args.variants,
    }}
    report["classifier"] = classifier_report(golden)

    if not args.classifier_only:
        if args.reranker:
            os.environ["RERANKER"] = args.reranker
        from dotenv import load_dotenv
        load_dotenv()
        from retrieval.adapters import build_pipeline_from_env

        pipeline, _store, _embedder, llm_client = build_pipeline_from_env()
        pipeline.candidate_k = args.candidate_k
        if pipeline.reranker is None and any(VARIANTS[v]["rerank"] for v in args.variants):
            print("⚠ RERANKER=none: rerank variants will equal their non-rerank counterparts")

        use_english = args.no_translate or llm_client is None
        if use_english and not args.no_translate:
            print("⚠ OPENROUTER_API_KEY not set: multilingual queries use their English gloss")

        t = time.time()
        report["variants"] = evaluate_retrieval(
            pipeline, golden, args.variants, args.ks, use_english)
        print(f"Retrieval eval finished in {time.time() - t:.1f}s")

        if args.generation or args.judge:
            report["generation"] = evaluate_generation(
                pipeline, llm_client, golden, args.gen_variant, args.model,
                args.judge_model if args.judge else None, use_english)

    md = render_markdown(report, args.ks)
    print("\n" + md)

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(args.out, f"eval-{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, f"eval-{stamp}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
