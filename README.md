# What wine do you want to drink tonight?

Ask in plain language, get a wine. A hybrid search stack — keyword + vector retrieval, fused and reranked — over 129,971 [Wine Enthusiast reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews).

**Live demo:** [huggingface.co/spaces/pratheeksha11/wine-tonight](https://huggingface.co/spaces/pratheeksha11/wine-tonight)

```
"cheap Italian red under $20, 90+ points"
   ↓  claude-haiku-4-5 extracts: country=Italy, color=red, price≤20, points≥90
   ↓  BM25 keyword search + exact vector similarity, both over the filtered set
   ↓  Reciprocal Rank Fusion merges the two ranked lists
   ↓  cross-encoder reranks the top 50
→  Villa Pillo 2005 Sant'Adele Merlot · Tuscany · $16 · 90/100
```

## Setup

1. Download `winemag-data-130k-v2.csv` from Kaggle to `data/` (gitignored).
2. Install dependencies:
   ```
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

## Build the indexes

Run in order (~3 minutes total; the first run downloads a 137MB embedding model):

```
python3 ingest/load_to_duckdb.py        # CSV -> DuckDB `wines` table
python3 ingest/build_vector_index.py    # embeddings.npy + FAISS index
python3 ingest/build_fts_index.py       # DuckDB FTS (BM25) index
python3 ingest/classify_wine_color.py   # derive `color` from `variety` (one-time LLM pass)
```

Each script prints its own verification output. Re-running `load_to_duckdb.py` drops and recreates the table, so re-run `build_fts_index.py` and `classify_wine_color.py` after it.

## Run

```
uvicorn api:app --app-dir serve --port 8000     # then open http://127.0.0.1:8000
```

`--app-dir serve` is required — `api.py` imports its siblings flatly.

Or trace one query through every stage on the command line:

```
python3 serve/hand_trace.py "cheap Italian red under \$20, 90+ points"
```

## Layout

| Path | Contents |
|---|---|
| `ingest/` | Offline: load, embed, index, enrich |
| `serve/` | Online: query understanding, retrieval, fusion, reranking, FastAPI app |
| `eval/` | Measurement harnesses (exact vs approximate retrieval) |
| `app.py`, `static/` | Gradio frontend (deployed) and HTML frontend (local) |
| `data/`, `index/` | Input and derived artifacts — both gitignored |

## Documentation

| Doc | Covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | How each stage works and why it was chosen |
| [docs/query-understanding.md](docs/query-understanding.md) | The cache → LLM → rules fallback chain, and why a local 1.5B model lost to regex |
| [docs/retrieval-evaluation.md](docs/retrieval-evaluation.md) | Exact vs approximate vector search, measured — and why HNSW isn't in the query path |
| [docs/tracing.md](docs/tracing.md) | `WINE_TRACE=1` end-to-end stage logging |
| [docs/deployment.md](docs/deployment.md) | Hugging Face Spaces deployment, and why Vercel can't host this |
| [docs/data.md](docs/data.md) | Dataset schema, the ratings scale, known data limitations |
| [PLAN.md](PLAN.md) | Original phased implementation plan (historical) |
