# search-project-wine

A learning project: building a hybrid search stack (vector search + keyword/filter search + reranker) over the Kaggle [Wine Reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews) dataset.

See [PLAN.md](./PLAN.md) for the full architecture and phased implementation plan.

## Setup

1. Download `winemag-data-130k-v2.csv` from Kaggle and place it at `data/winemag-data-130k-v2.csv` (not committed to this repo — see `.gitignore`).
2. Create a virtualenv and install dependencies:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY` (used for LLM-based query understanding).

## Project structure

- `data/` — raw input (gitignored)
- `index/` — regenerable derived artifacts: DuckDB file, FAISS index, embeddings (gitignored, rebuilt by `ingest/` scripts)
- `ingest/` — offline pipeline: load CSV into DuckDB, build embeddings + FAISS index, build DuckDB FTS index
- `serve/` — online pipeline: query understanding, hybrid retrieval, RRF fusion, cross-encoder reranking, FastAPI app
- `static/` — minimal HTML/JS search UI
