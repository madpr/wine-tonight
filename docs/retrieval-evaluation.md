# Exact vs approximate vector search, measured

**Conclusion: serving uses exact cosine similarity over the filtered subset. The FAISS HNSW index is not in the query path.** `faiss` is imported nowhere in `serve/` by default, and `faiss.index` is excluded from the deployed Space entirely.

That's a measured decision. This document is the measurement.

## Why the question comes up

`ingest/build_vector_index.py` builds an HNSW index and reports `recall@10 = 0.932` against exact search. If you stop there, it looks like the project built an ANN index and then didn't use it — cargo-culting.

The reason it isn't: at this scale exact search is both faster *and* exact, and we can show it.

## First, the 0.932 figure is unrepresentative

It measured **one stage in isolation**: raw HNSW vs exact neighbours, using 50 randomly-sampled **corpus vectors** as queries, with **no filters**, no BM25, no fusion, no reranking.

Under real natural-language queries *with filters*, the same index scores **0.357**. Two causes compound:

1. **Filtered ANN is a harder problem.** HNSW's graph spans the whole corpus. When an `IDSelector` excludes ~98% of nodes, the greedy traversal spends its budget on ineligible neighbours and can't reach the good ones.
2. **The index shipped misconfigured.** `efSearch` defaulted to 16 while the pipeline requested `top_n=100` — a search budget 6× smaller than the result count. Recall is structurally capped before filtering even applies.

## The parameter sweep

`eval/sweep_hnsw.py` sweeps all three HNSW parameters over a frozen 20-query set spanning filter selectivity from 129,971 eligible wines down to 46.

| Parameter | When | Tunable without rebuild? |
|---|---|---|
| `M` | build — edges per node | ❌ |
| `efConstruction` | build — graph quality | ❌ |
| `efSearch` | query — candidate list size | ✅ |

Rebuilds are cheap (2.5–15s) because `embeddings.npy` already exists — no model inference required.

| M | efConstruction | efSearch | vector recall@100 | ms |
|---|---|---|---|---|
| 32 | 40 | **16** *(as built)* | **0.357** | 0.12 |
| 32 | 40 | 1024 | 0.856 | 2.60 |
| 96 | 400 | 16 | 0.365 | 0.11 |
| 96 | 400 | **1024** *(best)* | **0.872** | 2.99 |

**`efSearch` dominates; rebuilding barely matters.** Raising `efSearch` 16 → 1024 gained **+0.50** recall. Rebuilding with M 32 → 96 and `efConstruction` 40 → 400 added **+0.016** on top of that.

That contradicts the intuitive prediction. The reasoning was that under selective filters, more edges per node means a better chance of an eligible hop — plausible, and real, but marginal at this scale. Worth recording because it's a case where the mechanism-based guess was directionally right and magnitudinally wrong.

## Head to head at matched effort

Search-only (excluding query embedding and the DuckDB filter scan, which are common to both), on identical filtered subsets:

| | Latency | Recall |
|---|---|---|
| **Exact numpy scan** | **0.41 ms** | **1.000** |
| HNSW, best config | 2.82 ms | 0.876 |

**Exact is faster on 20/20 queries.** Its worst case — scanning all 129,971 vectors on an unfiltered query — is 2.40 ms.

HNSW is strictly dominated here: slower *and* lossier. There is no tradeoff to negotiate.

### Why exact wins

A filter cuts exact search's work **proportionally** — it scans only eligible rows. The same filter makes HNSW **less** efficient, because the graph still spans the entire corpus and the traversal burns budget on nodes that will be discarded. Recovering that recall means raising `efSearch`, which costs the latency advantage that motivated ANN in the first place.

This is a known production pattern, not a quirk: filtered-vector systems estimate filter cardinality and **fall back to brute force when the filter is selective**.

## The pipeline absorbs retrieval loss anyway

`eval/compare_ann_exact.py` runs both paths through identical BM25, fusion and reranking:

| Stage | Agreement with exact |
|---|---|
| Vector candidates @100 | 0.357 |
| After RRF fusion with BM25 @50 | 0.705 |
| **After cross-encoder rerank @10** | **0.825** |
| **Top-1 result unchanged** | **18 / 20 queries** |

A 64% vector-retrieval loss becomes a 17.5% final-list difference and a 10% top-1 difference. BM25 independently surfaces wines HNSW missed, and the cross-encoder reorders survivors on true relevance rather than vector rank.

Divergence is **non-monotonic** in selectivity — narrow filters 0.900, mid 0.773, broad 0.880. The middle is worst: plausibly, narrow filters make both paths converge on a small pool where BM25 and reranking dominate, broad filters barely disrupt the graph, and mid-range gets a disrupted graph *plus* a pool large enough for the loss to show.

## So why keep the index?

Two reasons, both honest:

1. **It's the measurement instrument.** You cannot claim "exact search is the right call" without having built the alternative and measured it. The recall numbers above are what justify the decision.
2. **It's the scale-up lever.** Exact costs ~16 ns/vector, so ~10M vectors puts a full scan around 160 ms — which is where ANN starts earning its keep, and also roughly where holding the matrix in RAM stops being free.

`serve/retrieval.py:vector_candidates_ann` implements the filtered-ANN path (via `IDSelectorBatch` + `SearchParametersHNSW`) so both are runnable and comparable. `faiss` is imported lazily inside it, so the default exact path never loads the library or the 224MB index.

## Reproducing

```
python3 eval/build_query_set.py      # freeze 20 queries + extracted filters
python3 eval/compare_ann_exact.py    # stage-by-stage divergence
python3 eval/sweep_hnsw.py           # M / efConstruction / efSearch sweep
```

`build_query_set.py` is a separate step on purpose: `understand_query` is non-deterministic (the same input has produced three different residual query strings), so filters are extracted once and frozen. Calling the LLM inside the comparison would measure model variance instead of retrieval behavior.

## Caveats

- 20 queries is a small set.
- Exact is ground truth **by construction**, so this measures deviation, not relevance quality. It does not show exact search returns *better* wines — only that ANN returns different ones.
- No human relevance judgments exist for this corpus, so "is the ranking actually good?" remains unmeasured.
