"""Sweep HNSW build and search parameters against exact search.

Motivation: the first comparison ran the index as built -- M=32,
efConstruction=40, efSearch=16, all FAISS defaults -- and requested top_n=100
from a search whose candidate list only held 16 entries. That conflated a
misconfiguration (search budget 6x smaller than k) with an inherent property of
filtered ANN, so the resulting "filtered HNSW loses two thirds of candidates"
claim wasn't supported.

Three parameters, two of which need a rebuild:

  M              build   edges per node. Expected to matter most here: when a
                         filter excludes ~98% of nodes, the traversal strands
                         itself among ineligible neighbours, and more edges per
                         node means a better chance of an eligible hop.
  efConstruction build   candidate list while inserting; graph quality. Lifts
                         recall at every efSearch.
  efSearch       query   candidate list while searching. Free to change.

Rebuilds are cheap (2.5-7.5s) because embeddings.npy already exists -- no model
inference. Phase A below therefore sweeps widely on vector recall only; run the
full pipeline (eval/compare_ann_exact.py) on whatever configs look worth it.

Latency here times only the search call. The query embedding and the DuckDB
filter scan are common to exact and ANN, so including them would dilute the
comparison.
"""

import json
import pathlib
import statistics
import sys
import time

import duckdb
import faiss
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "serve"))

from retrieval import QUERY_PREFIX, _build_where_clause, _model

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "index" / "wine.duckdb"
EMBEDDINGS_PATH = ROOT / "index" / "embeddings.npy"
QUERY_SET = pathlib.Path(__file__).resolve().parent / "query_set.json"

TOP_N = 100
BUILD_CONFIGS = [(32, 40), (32, 200), (64, 200), (96, 400)]
EF_SEARCH_VALUES = [16, 64, 128, 256, 512, 1024]


def prepare(records: list[dict], embeddings: np.ndarray) -> list[dict]:
    """Per query: eligible ids, query vector, and the exact top-N ground truth."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    prepared = []
    for record in records:
        filters = record["filters"]
        where, params = _build_where_clause(filters)
        eligible = np.array(
            [r[0] for r in con.execute(f"SELECT id FROM wines WHERE {where}", params).fetchall()],
            dtype=np.int64,
        )
        query_emb = _model.encode(
            [QUERY_PREFIX + filters["query"]], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        # Ground truth: exact scan over the eligible subset.
        sims = embeddings[eligible] @ query_emb[0]
        k = min(TOP_N, len(eligible))
        top = np.argpartition(-sims, k - 1)[:k]
        exact_ids = set(eligible[top[np.argsort(-sims[top])]].tolist())

        prepared.append(
            {
                "query": record["query"],
                "eligible": eligible,
                "query_emb": query_emb,
                "exact_ids": exact_ids,
                "selectivity": len(eligible) / len(embeddings),
            }
        )
    con.close()
    return prepared


def main() -> None:
    embeddings = np.load(EMBEDDINGS_PATH)
    records = json.loads(QUERY_SET.read_text())
    prepared = prepare(records, embeddings)
    print(f"{len(prepared)} queries, ground truth = exact scan over the filtered subset\n")

    print(f"{'M':>4} {'efC':>5} {'efS':>6} {'vecR@100':>9} {'min':>6} {'zeros':>6} {'ms':>7} {'build':>7}")
    print("-" * 60)

    rows = []
    for M, ef_construction in BUILD_CONFIGS:
        t0 = time.perf_counter()
        index = faiss.IndexHNSWFlat(embeddings.shape[1], M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.add(embeddings)
        build_s = time.perf_counter() - t0

        for ef_search in EF_SEARCH_VALUES:
            recalls, timings = [], []
            for item in prepared:
                eligible = item["eligible"]
                # Selector holds a raw pointer into `eligible`; both must stay alive.
                selector = faiss.IDSelectorBatch(eligible.size, faiss.swig_ptr(eligible))
                params = faiss.SearchParametersHNSW(sel=selector, efSearch=ef_search)

                t = time.perf_counter()
                _, ids = index.search(item["query_emb"], TOP_N, params=params)
                timings.append((time.perf_counter() - t) * 1000)

                found = {int(i) for i in ids[0] if i != -1}
                expected = item["exact_ids"]
                recalls.append(len(found & expected) / len(expected) if expected else 1.0)

            zeros = sum(1 for r in recalls if r == 0.0)
            row = {
                "M": M, "efC": ef_construction, "efS": ef_search,
                "mean": statistics.mean(recalls), "min": min(recalls),
                "zeros": zeros, "ms": statistics.mean(timings), "build_s": build_s,
            }
            rows.append(row)
            print(f"{M:>4} {ef_construction:>5} {ef_search:>6} {row['mean']:>9.3f} "
                  f"{row['min']:>6.3f} {zeros:>6} {row['ms']:>7.2f} {build_s:>6.1f}s")

    best = max(rows, key=lambda r: r["mean"])
    print(f"\nbest vector recall: {best['mean']:.3f} at "
          f"M={best['M']}, efConstruction={best['efC']}, efSearch={best['efS']} "
          f"({best['ms']:.2f}ms/query)")

    baseline = next(r for r in rows if (r["M"], r["efC"], r["efS"]) == (32, 40, 16))
    print(f"original config:    {baseline['mean']:.3f} at M=32, efConstruction=40, efSearch=16 "
          f"({baseline['ms']:.2f}ms/query)")


if __name__ == "__main__":
    main()
