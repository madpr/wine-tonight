# Known limitations and scale behavior

What this system doesn't do, where it breaks, and what it would take to fix — stated up front rather than discovered later. Every number here is measured; where something is unmeasured, that's said explicitly.

---

## 1. Relevance is unmeasured

**This is the biggest gap.** Every metric in this repo compares two configurations against each other; none is anchored to whether a human wanted the wine.

| Measurement | Compares | Proves |
|---|---|---|
| recall@10 = 0.932 | HNSW vs exact neighbours | index fidelity |
| filtered recall@100 = 0.357 → 0.876 | ANN vs exact under filters | index fidelity |
| final recall@10 = 0.825, top-1 same 18/20 | ANN pipeline vs exact pipeline | **self-consistency** |
| rules vs Haiku = 95% | one extractor vs another | **self-consistency** |

Every one of those numbers would be unchanged if the ranking were uniformly bad — as long as it were *consistently* bad. Exact search is "ground truth" only as the un-approximated version of our own method.

### What would answer it

**NDCG@k** is the right primary metric, for two reasons specific to this domain:

- **Relevance is graded, not binary.** For "cheap Italian red, 90+": an Italian red at $18/91pts is perfect, an Italian *white* is wrong-but-related, a French white is simply wrong.
- **Position matters.** DCG divides each grade by `log₂(rank+1)`, so a result at rank 1 counts ~3.5× the same result at rank 10. Normalizing by the ideal ordering keeps queries comparable.

**Precision@k structurally cannot evaluate a reranker.** A reranker doesn't change *which* wines return from the top-50, only their order — and precision is order-blind. Same five wines, different order:

| Ranking | Grades | NDCG@5 | Precision@5 |
|---|---|---|---|
| Well-ordered | `[3,3,1,2,0]` | **0.989** | 0.6 |
| Badly-ordered | `[1,0,3,3,2]` | **0.722** | 0.6 |

### Getting labels

1. **Hand-label 20–30 queries** with graded relevance over *pooled* top-10 from several variants. Pooling matters: judging only your current system's output means you can never detect what it failed to retrieve. This is the prerequisite — without human labels you cannot validate a judge.
2. **LLM-as-judge, calibrated against that set** (measure Cohen's κ), then scaled to hundreds of queries. **Use a different model family than the pipeline's** — judging Claude's extraction with Claude risks forgiving exactly the misreadings it makes.
3. **Online: interleaving**, not a cohort A/B. Blend two rankers into one list and attribute clicks per result; it controls for position bias and needs far less traffic.

### What this leaves unvalidated

| Choice | Known | Unknown |
|---|---|---|
| Cross-encoder reranking | reorders heavily — 8/10 of top-10 turn over | **whether it improves anything** |
| Hybrid vs vector-only vs BM25-only | both paths contribute candidates | whether fusing beats either alone |
| RRF k=60 | it's the standard default | whether it suits this data |
| `color` filter | removes whites from "red" queries | whether users prefer that to a broader list |

---

## 2. Scale: where the design breaks

The design is **filter in DuckDB, exact-scan the survivors** — fast only because filtered subsets are small. But the corpus already spans the full range of selectivity:

| Query | Eligible | % of corpus |
|---|---|---|
| "Bordeaux blend under $25, 92+" | 46 | 0.04% |
| "big tannic wine for steak" (→ `color=red`) | **78,663** | **61%** |
| "earthy leather and cherry" (no filters) | **129,971** | **100%** |

"Any red" barely filters. Scale that shape to 100M rows and the same query exact-scans ~60M vectors.

### Memory is the first wall, not latency

`retrieval.py` loads the whole matrix (`np.load`) because `_embeddings[eligible_ids]` is a fancy-index gather over scattered, filter-determined rows — random access, so disk residency would mean a seek per row.

At 384 dims × 4 bytes = **1,536 bytes/vector**, linear in corpus size:

| Vectors | float32 | Viable? |
|---|---|---|
| 130k (today) | 0.2 GB | trivially |
| 10M | 15 GB | needs a real box |
| **100M** | **154 GB** | **not one ordinary machine** |
| 1B | 1.5 TB | needs a redesign |

You hit "won't fit" well before "too slow." Note `bge-small` is only 384-dim; 1536-dim models hit this 4× sooner.

**Fixes, in order of leverage:**

- **Product quantization** — split the vector into 96 subvectors, k-means each subspace, store one centroid byte each: 1,536 → 96 bytes, **16×**. 154 GB → 9.6 GB, fits one machine.
- **Scalar quantization** — float32 → int8, 4×, ~1–2% recall loss.
- **Rerank against full-precision vectors on SSD** to recover quantization loss — structurally the same retrieve-then-rescore pattern this pipeline already uses, one level lower: PQ codes (100M) → full vectors (~1,000) → cross-encoder (~50).
- **DiskANN** — vectors on SSD, only PQ codes and graph in RAM; serves 1B vectors from one node.
- **Sharding** — simplest, linear machine cost, adds a network hop.

### Latency, from measured numbers

Exact scan measured at **~16 ns/vector**:

| Scanned | Time |
|---|---|
| 130k (full corpus today) | 2.1 ms |
| 10M | ~160 ms |
| 60M ("any red" at 100M scale) | **~1 s** |

Best-config HNSW measured **2.82 ms**, which exact search reaches at roughly **175k vectors** — barely above where we are. The correct claim isn't "exact wins," it's **"exact wins at 130k, and we're within ~1.3× of the crossover."**

### Why the obvious fix isn't a drop-in

Filtered ANN (IVF with payload filters, filtered-DiskANN) is the right family — but our own sweep found **filtered HNSW degrades hardest exactly where brute force is cheapest**:

| Filter selectivity | Exact scan | Filtered ANN |
|---|---|---|
| Narrow (163 rows) | trivial — µs | **degrades badly** |
| Broad (60M rows) | **~1 s** | works as designed |

They're complementary, not competing. The production answer is **cardinality-based routing**: estimate rows passing the filter (DuckDB already has the statistics), then brute-force small sets and hit the ANN index for large ones. **What exists today is that system with the router pinned to one branch**, because at 130k that branch always wins.

---

## 3. Reranking depth (K=50) is convention, not measurement

Measured: **112 ms for 50 candidates**, ~2.2 ms/pair, linear.

| K | Rerank cost | Share of a ~2400 ms request |
|---|---|---|
| 20 | ~45 ms | 1.9% |
| **50** | **112 ms** | **4.7%** |
| 200 | ~450 ms | ~17% |

**K is a hard recall ceiling** — a wine at fused rank 60 is unreachable at K=50 however relevant it is.

The latency budget is *not* the binding constraint: 200 would cost ~450 ms against a 1920 ms LLM call. **The reason not to raise K is that there's no evidence it would help.** Setting it properly means measuring recall@K of the fused list against labeled relevance and picking where the curve plateaus — which needs §1.

---

## 4. Duplicate reviews

The source CSV contains exact duplicates — ids **5574 and 68601** are byte-identical in `title` and `description` — surfacing as near-identical adjacent results.

Dedup belongs at **ingestion, as a content hash** over `(title, description)`: O(1) per record, cheap into the millions. Near-duplicates (one word changed) defeat hashing and need MinHash/SimHash or an embedding-similarity threshold.

**Why it's unimplemented:** dropping rows changes row count and `id` contiguity, and `build_vector_index.py` deliberately relies on `id == embeddings row index` to avoid a mapping file. Dedup therefore means re-running the entire offline pipeline.

**A gap inside the gap:** the duplicate *rate* has never been counted — examples are known, magnitude isn't. That's one SQL query and the obvious first step.

---

## 5. Freshness: no incremental path exists

The original architecture sketch called out freshness (reindex lag vs cost); `PLAN.md` dismissed it in one line as a static batch dataset. True for a Kaggle CSV, but the consequence is that nothing incremental was ever built:

- `load_to_duckdb.py` **drops and recreates** the table — no append path
- Embeddings are one batch; adding a wine means re-embedding everything, or appending and breaking the `id == row index` invariant
- The FTS index and color classification are both wholesale rebuilds

Making it incremental requires decoupling id from row position (a mapping file), append-only ingestion with hash dedup, and choosing a reindex cadence — the lag-vs-cost tradeoff the sketch named.

### Cold-start splits in two

- **System cold-start is real and measured.** The Space loads a 200 MB matrix and two models at boot; the first query measured 357 ms against 13–20 ms steady-state for vector search, on top of seconds of model loading.
- **Item cold-start doesn't apply.** Ranking is purely content-based, so a brand-new wine is instantly as rankable as any other. The flip side is the real limitation: **there is no popularity, rating-volume, or behavioral signal in the ranking at all**, because no interaction data exists to build one from.

---

## 6. Smaller known gaps

- **`price` is null for ~7% of wines**, so any price filter silently excludes them.
- **Long-tail varieties aren't filterable** — only the top 200 by frequency reach the query-understanding prompt; the other 507 fall through to free-text search.
- **The query-understanding system prompt is uncached.** ~2,340 stable tokens per call, making it a prompt-caching candidate that would cut both the dominant latency component and per-query cost.
- **`region_1`/`province` are unused as filters** despite being clean enough — `province` would catch "a wine from Tuscany", which `country` can't resolve.
