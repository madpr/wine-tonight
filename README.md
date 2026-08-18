# What wine do you want to drink tonight?

Ask in plain language, get a wine. A learning project: building a hybrid search stack (vector search + keyword/filter search + reranker) over the Kaggle [Wine Reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews) dataset.

See [PLAN.md](./PLAN.md) for the full architecture and phased implementation plan.

## Setup

1. Download `winemag-data-130k-v2.csv` from Kaggle and place it at `data/winemag-data-130k-v2.csv` (not committed to this repo — see `.gitignore`).
2. Create a virtualenv and install dependencies:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY` (used for LLM-based query understanding and the one-time wine-color classification).

## Build the indexes (offline, run in order)

```
python3 ingest/load_to_duckdb.py        # CSV -> DuckDB `wines` table (129,971 rows)
python3 ingest/build_vector_index.py    # embed descriptions -> embeddings.npy + FAISS HNSW index
python3 ingest/build_fts_index.py       # DuckDB FTS (BM25) index on `description`
python3 ingest/classify_wine_color.py   # derive a `color` field from `variety` (707 varieties, one-time LLM pass)
```

Each script prints its own verification output (row counts, HNSW-vs-exact recall@10, sample query results, color distribution). The first embedding run downloads `BAAI/bge-small-en-v1.5` (~137MB) to the shared Hugging Face cache and takes ~2 minutes on Apple Silicon.

Note on re-running: `load_to_duckdb.py` drops and recreates the `wines` table, so re-running it also requires re-running `build_fts_index.py` and `classify_wine_color.py`. Embeddings only need rebuilding if row count or `id` ordering changes.

## Serve locally

```
uvicorn api:app --app-dir serve --port 8000
```

Then open http://127.0.0.1:8000. The `--app-dir serve` flag is required — `api.py` imports its sibling modules flatly.

The UI shows the LLM-extracted filters and how many candidates each retrieval path contributed, alongside the ranked results.

To trace a single query through every stage on the command line instead:

```
python3 serve/hand_trace.py "cheap Italian red under \$20, 90+ points"
```

This prints the extracted filters, both candidate counts, and the RRF-only top 10 next to the reranked top 10 — useful for seeing what reranking actually changes.

## Tracing

Set `WINE_TRACE=1` to log every pipeline stage — inputs on entry, return value and duration on exit, all correlated by a per-request trace id:

```
WINE_TRACE=1 python3 serve/hand_trace.py "cheap Italian red under \$20, 90+ points"
WINE_TRACE=1 uvicorn api:app --app-dir serve --port 8000
```

```
[83e45354] → search('cheap Italian red under $20, 90+ points')
[83e45354]   → understand_query('cheap Italian red under $20, 90+ points')
[83e45354]   ← understand_query = {country='Italy', color='red', price_max=20, points_min=90, ...} [1920ms]
[83e45354]     → keyword_candidates({...}, 'cheap Italian red', 100)
[83e45354]     ← keyword_candidates = list[78403, 60268, ..., +54 more] (n=60) [43ms]
[83e45354]     ← vector_candidates = list[120895, 22797, ..., +94 more] (n=100) [319ms]
[83e45354]   ← rerank = list[tuple[60268, -5.84], ..., +44 more] (n=50) [112ms]
[83e45354] ← search = list[...] (n=3) [2395ms]
```

Values are summarized rather than dumped — a `(129971, 384)` embeddings matrix logs as its shape, and long lists as a head plus a count — so the output stays readable. Tracing is off unless `WINE_TRACE` is set, and the decorator short-circuits when disabled.

Useful immediately: the timings show the LLM query-understanding call is ~80% of total latency, while all the retrieval, fusion, and reranking together run in under 500ms.

## Project structure

- `data/` — raw input (gitignored)
- `index/` — regenerable derived artifacts: DuckDB file, FAISS index, embeddings (gitignored, rebuilt by `ingest/` scripts)
- `ingest/` — offline pipeline: load CSV into DuckDB, build embeddings + FAISS index, build DuckDB FTS index, classify wine color
- `serve/` — online pipeline: query understanding, hybrid retrieval, RRF fusion, cross-encoder reranking, FastAPI app
- `static/` — minimal HTML/JS search UI

## How it works

**Offline:** the CSV becomes a DuckDB table (structured filters + BM25 full-text index) and a matrix of 384-dim embeddings of each `description` (searchable via FAISS HNSW). A one-time LLM pass derives a `color` field from `variety`, since the dataset has no color column.

**Per query:** `claude-haiku-4-5` maps natural language onto hard filters (`country`, `variety`, `color`, price range, points range) plus a residual semantic query. Both retrieval paths filter first, then search — BM25 within the filtered set, and exact cosine similarity over the filtered subset of embeddings. The two ranked lists are fused with Reciprocal Rank Fusion (rank-based, since BM25 scores and cosine similarities aren't comparable scales), and the top 50 are reordered by a cross-encoder that reads the query and each description together.

### Known limitations

- The source CSV contains exact-duplicate reviews, which surface as near-identical results. Deduplication belongs at ingestion (content hash on `title`+`description`) but is not implemented.
- `price` is null for ~7% of wines; a price filter silently excludes those rows.
- Long-tail varieties beyond the top 200 aren't offered to the query-understanding step, so they fall through to free-text search rather than becoming a hard filter.
