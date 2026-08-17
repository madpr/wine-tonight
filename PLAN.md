# Hybrid Wine Search Stack (Vector + Keyword + Rerank)

## Context

This is a learning project to build a hybrid search stack — vector search, keyword/filter search, and a reranker — following the architecture the user sketched (ingestion offline/async → shared stores → serving online/per-query → ranked results, with an optional LLM-explanation bonus). The goal is to understand each component's role (ANN retrieval, BM25 keyword search, hybrid fusion, cross-encoder reranking, LLM query understanding) by building it, not just wiring together a framework.

Data: a Kaggle "Wine Reviews" CSV (`data/winemag-data-130k-v2.csv`, 129,971 rows) is already downloaded. Verified schema: `description` (free text, 0 nulls — the field to embed), `title`/`designation`/`winery` (identifiers), `country`/`province`/`region_1`/`region_2` (geo facets, some nulls), `variety` (707 distinct grape varieties), `points` (80–100 rating), `price` (numeric, ~7% null).

Environment: macOS arm64 (Apple Silicon), 24GB RAM, Python 3.12.13 via pyenv. No venv exists yet and the pip environment is essentially empty — nothing is installed. DuckDB CLI v1.5.5 is installed standalone but not the Python package. This is a brand-new project directory: no requirements.txt, README, or other scaffolding exists yet.

The user wants to: (1) do a setup pass first since almost nothing is installed, (2) build and review the offline ingestion pipeline, then (3) build local online serving. Vercel/hosting deployment is explicitly deferred — not planned in detail here.

**Decisions confirmed with the user:** fully local embeddings (no embedding API), a plain HTML/JS page (not Streamlit) for local testing, and **LLM-based query understanding from the start** (via the Anthropic API) rather than starting with rule-based parsing.

---

## Project structure

```
search-project-wine/
├── data/
│   └── winemag-data-130k-v2.csv        # raw, never modified
├── index/                               # all derived/regenerable artifacts
│   ├── wine.duckdb                      # structured table + FTS index (persisted in-file)
│   ├── embeddings.npy                   # (129971, 384) float32, row-aligned to id
│   └── faiss.index                      # persisted FAISS HNSW index
├── ingest/
│   ├── load_to_duckdb.py                # Phase 1: CSV -> DuckDB table with id PK
│   ├── build_vector_index.py            # Phase 2: embed + FAISS build + persist
│   └── build_fts_index.py               # Phase 3: PRAGMA create_fts_index
├── serve/
│   ├── query_understanding.py           # LLM (Anthropic) query -> {filters, query text}
│   ├── retrieval.py                     # DuckDB filter+FTS query + FAISS ANN query
│   ├── fusion.py                        # reciprocal rank fusion (RRF)
│   ├── rerank.py                        # cross-encoder rerank of top-K
│   └── api.py                           # FastAPI app, /search endpoint
├── static/
│   └── index.html                       # minimal single-page search UI
├── requirements.txt
├── .env                                  # ANTHROPIC_API_KEY (gitignored)
└── .gitignore
```

`data/` is read-only input. `index/` is fully regenerable output of the `ingest/` scripts. `ingest/` vs `serve/` maps 1:1 to the two halves of the architecture diagram (offline/async vs online/per-query).

---

## Phase 0 — Environment setup

1. Create a venv from the pyenv-managed 3.12.13 interpreter: `python3 -m venv .venv`, activate it.
2. Write `requirements.txt` (versions verified installable on this machine/platform — arm64 wheels confirmed to exist on PyPI for faiss-cpu and torch, no special index URLs needed):
   ```
   duckdb==1.5.5
   pandas>=2.2,<3
   numpy>=2.0,<3
   pyarrow>=18.0
   sentence-transformers>=3.0,<6.0
   torch>=2.5
   faiss-cpu>=1.9,<2.0
   fastapi>=0.115
   uvicorn[standard]>=0.30
   python-dotenv>=1.0
   anthropic>=0.40
   ```
3. `pip install -r requirements.txt`, then sanity-check each import in a throwaway `python -c` session (duckdb, faiss, sentence_transformers, torch, fastapi, anthropic).
4. Create `.env` for `ANTHROPIC_API_KEY` (needed for Phase 4 query understanding) and a `.gitignore` covering `.venv/`, `.env`, `index/`, `__pycache__/`.
5. Note: sentence-transformers will auto-pick `mps` as the torch device on Apple Silicon. Usually a speed win; if embeddings look off, fall back to `device="cpu"` — cheap to try both on this corpus size.

**Checkpoint:** all imports succeed cleanly; `ANTHROPIC_API_KEY` loads via `python-dotenv`.

---

## Phase 1 — ETL into DuckDB (structured store)

Teaches: turning a raw source into a queryable structured store with a stable key (the diagram's "Parse+structure / Extract fields → Structured store").

- Load via `read_csv_auto`; the CSV's blank-header first column is auto-named `column00` (confirmed: 0-indexed unique row id) — rename to `id` and use as primary key.
- `CREATE TABLE wines AS SELECT column00 AS id, * EXCLUDE (column00) FROM read_csv_auto('data/winemag-data-130k-v2.csv')`, persisted to `index/wine.duckdb` (not `:memory:`) so the FTS index (Phase 3) and every later query reuse the same on-disk file.
- Profiling check: confirm null counts match what was already observed (region_1 ~16%, price ~7%, variety 1 null) after load.
- One-line freshness note per the diagram's callout: this is a static one-time batch load, so reindex-lag-vs-cost doesn't apply here; it would become relevant (incremental ingestion + periodic reindex) only with a live data source.

**Checkpoint:** `SELECT count(*) FROM wines` = 129,971; a couple of known rows (e.g. id=0, the Nicosia Vulkà Bianco) spot-check correctly against the raw CSV.

---

## Phase 2 — Vector index (embeddings + FAISS)

Teaches: bi-encoder embedding + real ANN indexing (the diagram's "Embed chunks → Vector index (ANN/semantic)" path).

- **Model:** `BAAI/bge-small-en-v1.5` (384-dim, 33M params) — strong quality/speed tradeoff for short tasting notes; no chunking needed (descriptions are single-paragraph, unlike the diagram's generic resume-chunking case). Embedding matrix: ~130K × 384 × 4 bytes ≈ 200MB, trivial on 24GB RAM.
- **Important BGE detail:** the query side needs the instruction prefix `"Represent this sentence for searching relevant passages: "`; the passage/document side (i.e. `description`) does **not**. Embed `description` with no prefix; only add it when embedding the user's query at serve time. Getting this backwards silently hurts recall — call it out explicitly in code comments.
- **Index type: `IndexHNSWFlat`**, not flat, not IVF — no training/clustering step required (unlike IVF), and it's the closest match to what production vector DBs actually use (pgvector HNSW, Pinecone, etc.), so it's the most transferable ANN concept. Use inner product on L2-normalized vectors (cosine similarity). Optionally also build a brute-force `IndexFlatIP` once as a baseline to compute recall@10 of HNSW against exact search — a good, cheap way to *see* the ANN approximation tradeoff concretely.
- Sort by `id` before embedding so FAISS's internal row order equals `id` directly — no separate id-mapping file needed.
- Persist `embeddings.npy` and `faiss.index` under `index/`.

**Checkpoint:** manually embed a test query (e.g. "cherry and leather notes") and confirm FAISS nearest neighbors are qualitatively sane; run the HNSW-vs-Flat recall@10 comparison once as a learning artifact.

---

## Phase 3 — Keyword/filter search (DuckDB SQL + FTS)

Teaches: the lexical/exact-match counterpart to vector search — the "Structured store" box plus real BM25 (verified working end-to-end against this DuckDB version).

- Structured filtering: plain SQL `WHERE` over `country`, `variety`, `price`, `points`, `province` — already fully supported by the Phase 1 table.
- Full-text search on `description`:
  1. `INSTALL fts; LOAD fts;`
  2. `PRAGMA create_fts_index('wines', 'id', 'description');`
  3. Query via `fts_main_wines.match_bm25(id, 'query text')` joined back to `wines` by `id`, ordered by score — real BM25 scoring, persisted automatically inside `wine.duckdb` since it's a real on-disk file (no separate artifact to manage).
- This is the piece that catches exact terms (a specific varietal spelling, "tannic", etc.) that embedding similarity alone can blur.

**Checkpoint:** run a filter query (e.g. `country='Italy' AND points>=90 AND price<20`) and a BM25 query (e.g. `match_bm25` for "tannic cherry") independently and confirm both look sane.

**This is the natural pause point to review the full offline/ingestion design before moving to serving**, matching the stated preference to discuss offline first.

---

## Phase 4 — Online serving: query understanding + hybrid retrieval + fusion

Teaches: the diagram's "Query understanding," "Hybrid retrieval" (filter-then-search decision), and "coarse ranking" (fusion) steps.

- **Query understanding (LLM-based, per user's choice):** call the Anthropic API with a structured-output/tool-use prompt that takes the raw natural-language query (e.g. "cheap Italian red under $20, 90+ points") and returns JSON: `{country, variety, price_min, price_max, points_min, points_max, query}` — `query` being the residual semantic text to embed/search. Load the known distinct `country`/`variety` values from DuckDB once and pass them (or a relevant subset) into the prompt so the LLM maps loosely-worded terms ("Italian" → `country='Italy'`) onto real column values reliably.
- **Hybrid retrieval, filter-then-search vs search-then-filter:** at 130K rows both directions are cheap (DuckDB filter scan and FAISS HNSW search are both sub-100ms), so the choice is about correctness, not performance:
  - Structured-filter + BM25 candidate set (A): apply the extracted filters as a DuckDB `WHERE` clause *first*, then compute BM25 only within that filtered set — this keeps BM25 scores meaningful (scored within the universe actually being searched).
  - Vector candidate set (B): run FAISS ANN search *unfiltered* over the full corpus (top-N, e.g. N=100), then drop any hits that violate the structured filters as a post-filter. Building a properly filtered FAISS index (IDSelector etc.) is real added complexity that only pays off at much larger scale — worth noting explicitly as the reason this decision point exists in the diagram, even though the simple version is the right call here.
- **Fusion — Reciprocal Rank Fusion (RRF):** `score(doc) = sum over lists of 1/(k + rank_in_list)`, k≈60. Combines A and B using rank position rather than raw score, sidestepping the fact that BM25 scores and cosine similarities aren't on comparable scales — the standard, simplest correct answer here.

**Checkpoint:** hand-trace a query like "cheap Italian red under $20, 90+ points" end-to-end through LLM query understanding → both retrieval paths → RRF, and confirm the fused top-N looks sane.

---

## Phase 5 — Reranking (cross-encoder)

Teaches: why a second-stage reranker is distinct from bi-encoder retrieval — the diagram's "coarse → rerank" decision point. A bi-encoder embeds query and document independently (fast, scalable, no cross-attention); a cross-encoder scores the (query, document) pair jointly (much more accurate, too slow to run over the whole corpus — hence applied only to the top-K survivors of RRF).

- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (6-layer, ~22M params) — fast enough to rerank ~50 candidates in well under a second on CPU, keeping local iteration snappy. Note `BAAI/bge-reranker-base` as a documented, drop-in-compatible "more accurate but 10x+ slower" swap for later, since the interface (score a query/description pair) is identical.
- Apply only to the top-K (e.g. 50) of the RRF-fused list, not the full corpus.

**Checkpoint:** compare RRF-only top-10 vs. reranked top-10 for a couple of test queries — reranking should visibly reorder results in qualitatively sensible ways.

---

## Phase 6 — FastAPI + minimal UI

Teaches: wiring the full pipeline behind one request/response boundary — the diagram's "Ranked results list" output.

- `POST /search` in `serve/api.py`: query understanding → hybrid retrieval → RRF → rerank → JSON list of results (title, description snippet, country, variety, price, points, score).
- UI (per user's choice): a single static `index.html` with vanilla JS `fetch()`, served by FastAPI (`StaticFiles` mount or a `/` route). No build step, no extra framework — simplest way to type a query and see ranked results in a browser while tuning relevance.
- **Bonus, explicitly optional, not blocking:** LLM explanations — one more Anthropic call per result (or batched) generating a one-line "why this matches," i.e. true RAG on top of the ranked list, per the diagram's dashed "LLM explanations" box.
- Deployment (Vercel/hosting): explicitly out of scope for this plan — a separate future step once the local pipeline works end-to-end.

**Checkpoint:** type several realistic wine queries into the browser UI and confirm the full pipeline returns sensible ranked results within a second or two.

---

## Milestone summary

| Phase | Deliverable | Review point |
|---|---|---|
| 0 | venv + verified installs | imports succeed |
| 1 | `index/wine.duckdb` with `wines` table | row counts + spot checks match CSV |
| 2 | `index/embeddings.npy` + `index/faiss.index` | manual query sanity check, HNSW vs Flat recall comparison |
| 3 | FTS index inside `wine.duckdb` | filter query + BM25 query independently sane — **pause for offline review** |
| 4 | `serve/query_understanding.py`, `serve/retrieval.py`, `serve/fusion.py` | hand-traced hybrid query produces a sane fused list |
| 5 | `serve/rerank.py` | reranked top-10 visibly improves over RRF-only |
| 6 | `serve/api.py` + `static/index.html` | working browser search loop |

### Critical files

- `ingest/load_to_duckdb.py`, `ingest/build_vector_index.py`, `ingest/build_fts_index.py`
- `serve/query_understanding.py`, `serve/retrieval.py`, `serve/fusion.py`, `serve/rerank.py`, `serve/api.py`
- `requirements.txt`, `.env`, `static/index.html`

### Verification approach

Each phase has its own checkpoint above (row counts, manual query sanity checks, recall comparisons, before/after reranking comparisons). After Phase 3, pause to review the offline pipeline together before starting Phase 4+. After Phase 6, verify end-to-end by running `uvicorn serve.api:app --reload` and issuing real natural-language wine queries through the browser UI, checking that filters are extracted correctly, results are relevant, and reranking visibly improves ordering over RRF alone.
