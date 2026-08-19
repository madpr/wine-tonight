# Architecture

Two halves: an offline pipeline that builds shared stores, and an online pipeline that serves queries against them.

```
OFFLINE (ingest/)                          ONLINE (serve/)

winemag CSV                                natural-language query
    │                                          │
    ├─→ DuckDB `wines` table  ←──────────────  ├─ query understanding (LLM)
    │     + BM25 FTS index                     │     → filters + residual query
    │     + derived `color`                    │
    │                                          ├─ keyword path: filter → BM25
    └─→ embeddings.npy ←─────────────────────  ├─ vector path:  filter → cosine
          (+ faiss.index, offline only)        │
                                               ├─ Reciprocal Rank Fusion
                                               ├─ cross-encoder rerank (top 50)
                                               └─→ ranked results
```

## Offline

**`load_to_duckdb.py`** — loads the CSV into a persistent DuckDB table with a primary-key `id`. Trims all VARCHAR columns: a few rows had trailing whitespace in `variety` (`'Tintilia '`), which silently breaks exact-match filters downstream.

**`build_vector_index.py`** — embeds every `description` with `BAAI/bge-small-en-v1.5` (384-dim, 33M params) into a row-aligned `embeddings.npy`, so array row `i` *is* wine `id` `i` — no id-mapping file needed.

One BGE-specific detail: the query side requires the instruction prefix `"Represent this sentence for searching relevant passages: "`; the passage side does not. Getting this backwards silently degrades recall.

Also builds a FAISS HNSW index. **This is not used at query time** — see [retrieval-evaluation.md](retrieval-evaluation.md).

**`build_fts_index.py`** — DuckDB's FTS extension over `description`, giving real BM25 scoring. Note that `INSTALL fts` is needed at query time too, not just `LOAD`: the index *data* lives inside the `.duckdb` file, but the extension *binary* is a platform-specific shared library that a fresh host has never fetched.

**`classify_wine_color.py`** — the dataset has no color column, so `claude-haiku-4-5` classifies each of the 707 distinct varieties once into `red`/`white`/`rose`/`sparkling`/`dessert`/`fortified`/`orange`/`other`, joined onto every wine as `wines.color`.

This exists because "red wine" had no hard filter to enforce it — "red" reached only BM25/vector scoring as free text, so white and rosé wines ranked into results for red-wine queries. Classifying 707 variety *names* once is cheap; classifying 130k rows would not be.

## Online

### 1. Query understanding

`claude-haiku-4-5` with a strict tool schema maps natural language onto `{country, variety, color, price_min/max, points_min/max, query}`. The known `country` and top-200 `variety` values are loaded from DuckDB and injected into the system prompt, so "Italian" resolves to the exact stored value `Italy` rather than a hallucinated near-miss.

Haiku rather than a frontier model because this is narrow extraction against a fixed enum, not open-ended reasoning. `strict: true` guarantees the JSON validates before it reaches a SQL query.

`color` deliberately omits `other` from the *extraction* enum even though the *classification* enum includes it: only 6 wines in the corpus are `other`, so a spurious `color='other'` filter would narrow 130k wines to 6 — a worse failure than not filtering at all. Ambiguous input should yield `null`.

### 2. Hybrid retrieval — both paths filter first

**Keyword path:** filters as a SQL `WHERE` clause, then BM25 scored only within that filtered set — so scores are meaningful relative to the universe actually being searched.

**Vector path:** filters via DuckDB to get eligible ids, then exact cosine similarity (embeddings are pre-normalized, so it's a dot product) over just those rows.

The vector path originally did the opposite — searched the full corpus with ANN, then dropped violators. That returns **zero results** whenever filters are selective and uncorrelated with the query's semantic direction: `country='Italy' AND price≤20 AND points≥90` matches 163 of 129,971 wines, and none appeared in the top-100 unfiltered neighbours for "tannic cherry". Filtering first is structurally immune to that.

### 3. Fusion — Reciprocal Rank Fusion

`score(doc) = Σ 1/(k + rank_in_list)`, k=60.

BM25 scores (unbounded, corpus-statistics-dependent) and cosine similarities (−1 to 1) aren't comparable, so averaging them would let whichever has larger magnitude dominate arbitrarily. RRF uses rank *position* instead, sidestepping normalization entirely. Documents both paths rank highly rise to the top — the actual point of hybrid search.

### 4. Reranking — cross-encoder

`cross-encoder/ms-marco-MiniLM-L-6-v2` rescores the fused top 50.

This is the conceptual opposite of the embedding model. A **bi-encoder** embeds query and document independently, so comparison is cheap vector math but there's no cross-attention between them. A **cross-encoder** feeds the pair jointly through a transformer — much more accurate, far too slow for 130k documents. Hence two stages: cheap broad retrieval, then expensive narrow reranking.

Its output scores are raw uncalibrated logits (e.g. `-5.84`, `-10.25`) and are meaningful only for ordering within one call — which is why the UI shows rank position, not the score.

## Cost and latency shape

From `WINE_TRACE=1` (see [tracing.md](tracing.md)):

| Stage | Latency | Note |
|---|---|---|
| Query understanding (LLM) | ~1920 ms | **~80% of the request** |
| Vector candidates | ~319 ms | includes query embedding |
| Reranking | ~112 ms | |
| Keyword candidates | ~43 ms | |
| Fusion | ~0 ms | |

The only per-query monetary cost is the LLM call (~2,340 input + ~100 output tokens on Haiku ≈ $0.003/query). Embedding, retrieval, fusion and reranking all run locally and cost nothing.

An unexplored optimization: the system prompt (country + variety lists + tool schema) is stable across queries and is most of those 2,340 tokens, making it a prompt-caching candidate — which would cut both the dominant latency component and the per-query cost.
