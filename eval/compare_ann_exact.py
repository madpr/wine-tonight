"""Measure whether approximate vector retrieval changes what the user sees.

The recall@10 = 0.932 figure from ingest/build_vector_index.py measured a single
stage in isolation: raw HNSW vs exact neighbours, using corpus vectors as
queries, with no filters, no BM25, no fusion, no reranking. It does not answer
the question that matters -- whether the final ranked list differs.

It could go either way. RRF fuses with BM25, so a wine HNSW misses may be
recovered if BM25 already ranked it, and the cross-encoder reorders the top 50
anyway. Or the loss could amplify: a wine that would sit at rank ~45 under exact
search but slips past ~50 under ANN drops out of the reranker's window
entirely, and the reranker is what produces the final order.

This runs both paths over a frozen query set (see build_query_set.py -- filters
are pre-extracted because the LLM is non-deterministic) and reports divergence
at each stage. Exact is treated as ground truth; the question is how far
approximate deviates, not which is more relevant.
"""

import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "serve"))

from fusion import reciprocal_rank_fusion
from rerank import rerank
from retrieval import keyword_candidates, vector_candidates, vector_candidates_ann

QUERY_SET = pathlib.Path(__file__).resolve().parent / "query_set.json"

CANDIDATES_PER_PATH = 100
RERANK_TOP_N = 50
FINAL_K = 10


def overlap(a: list[int], b: list[int]) -> float:
    """Fraction of `a` also present in `b`. Undefined on empty a -> 1.0."""
    if not a:
        return 1.0
    return len(set(a) & set(b)) / len(a)


def run_query(record: dict) -> dict:
    query, filters = record["query"], record["filters"]
    semantic = filters["query"]

    # BM25 is identical for both paths -- only the vector stage differs.
    keyword_ids = keyword_candidates(filters, semantic, top_n=CANDIDATES_PER_PATH)

    t0 = time.perf_counter()
    exact_vec = vector_candidates(filters, semantic, top_n=CANDIDATES_PER_PATH)
    exact_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ann_vec = vector_candidates_ann(filters, semantic, top_n=CANDIDATES_PER_PATH)
    ann_ms = (time.perf_counter() - t0) * 1000

    exact_fused = [i for i, _ in reciprocal_rank_fusion(keyword_ids, exact_vec)]
    ann_fused = [i for i, _ in reciprocal_rank_fusion(keyword_ids, ann_vec)]

    exact_final = [i for i, _ in rerank(query, exact_fused, top_n=RERANK_TOP_N)][:FINAL_K]
    ann_final = [i for i, _ in rerank(query, ann_fused, top_n=RERANK_TOP_N)][:FINAL_K]

    return {
        "query": query,
        "eligible": record["eligible_count"],
        "n_keyword": len(keyword_ids),
        "n_exact_vec": len(exact_vec),
        "n_ann_vec": len(ann_vec),
        "vec_recall": overlap(exact_vec, ann_vec),
        "fused_recall_50": overlap(exact_fused[:RERANK_TOP_N], ann_fused[:RERANK_TOP_N]),
        "final_overlap": overlap(exact_final, ann_final),
        "top1_same": bool(exact_final and ann_final and exact_final[0] == ann_final[0]),
        "exact_ms": exact_ms,
        "ann_ms": ann_ms,
    }


def main() -> None:
    records = json.loads(QUERY_SET.read_text())
    results = [run_query(r) for r in records]

    print(f"{'eligible':>9} {'vecR@100':>9} {'fusedR@50':>10} {'finalR@10':>10} {'top1':>5} "
          f"{'exact':>8} {'ann':>8}  query")
    print("-" * 100)
    for r in sorted(results, key=lambda x: -x["eligible"]):
        print(f"{r['eligible']:>9,} {r['vec_recall']:>9.3f} {r['fused_recall_50']:>10.3f} "
              f"{r['final_overlap']:>10.3f} {'same' if r['top1_same'] else 'DIFF':>5} "
              f"{r['exact_ms']:>7.1f}m {r['ann_ms']:>7.1f}m  {r['query'][:38]}")

    print()
    print("=== aggregate ===")
    for label, key in [
        ("vector recall@100", "vec_recall"),
        ("fused recall@50", "fused_recall_50"),
        ("FINAL recall@10", "final_overlap"),
    ]:
        vals = [r[key] for r in results]
        print(f"{label:>18}: mean {statistics.mean(vals):.3f}  min {min(vals):.3f}  "
              f"perfect {sum(1 for v in vals if v == 1.0)}/{len(vals)}")

    same = sum(1 for r in results if r["top1_same"])
    print(f"{'top-1 unchanged':>18}: {same}/{len(results)}")
    print(f"{'mean latency':>18}: exact {statistics.mean([r['exact_ms'] for r in results]):.1f}ms  "
          f"ann {statistics.mean([r['ann_ms'] for r in results]):.1f}ms")

    # The hypothesis worth testing: does divergence track filter selectivity?
    print()
    print("=== final recall@10 by filter selectivity ===")
    buckets = [("narrow (<500)", 0, 500), ("mid (500-20k)", 500, 20_000),
               ("broad (>20k)", 20_000, 10**9)]
    for label, lo, hi in buckets:
        vals = [r["final_overlap"] for r in results if lo <= r["eligible"] < hi]
        if vals:
            print(f"{label:>16}: mean {statistics.mean(vals):.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
