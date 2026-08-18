"""Phase 4: hybrid retrieval -- structured-filter+BM25 candidates (A) and
filtered vector-similarity candidates (B).

Both paths filter first, then search:

- Keyword path: filters applied as a SQL WHERE clause, then BM25 scored
  only within that filtered set (keeps scores meaningful).
- Vector path: filters applied via DuckDB to get the eligible id set, then
  exact cosine similarity (embeddings are pre-normalized, so this is a dot
  product) is computed only over that subset, using the in-memory
  embeddings matrix from Phase 2.

  This started as search-then-filter using the FAISS ANN index (search
  unfiltered, drop violators afterward) -- but that silently returns zero
  results whenever a filter is selective and doesn't correlate with the
  query's semantic direction (empirically: country='Italy', price<=20,
  points>=90 matches only 375/129971 wines, and none of the top-100
  unfiltered neighbors for "tannic cherry" landed in that set). Filtering
  first avoids that failure mode. Exact brute-force search over even the
  full unfiltered corpus is well under 100ms at this size (a 130971x384
  dot product), so ANN's approximation isn't needed for serve-time
  correctness here -- FAISS/HNSW (Phase 2) remains the right tool once a
  corpus is too large to brute-force on every query (millions+ rows).
"""

import pathlib

import duckdb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "index" / "wine.duckdb"
EMBEDDINGS_PATH = ROOT / "index" / "embeddings.npy"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_device = "mps" if torch.backends.mps.is_available() else "cpu"
_model = SentenceTransformer(MODEL_NAME, device=_device)
_embeddings = np.load(EMBEDDINGS_PATH)  # (129971, 384) float32, row i == wine id i


def _build_where_clause(filters: dict) -> tuple[str, list]:
    clauses = []
    params = []
    if filters.get("country"):
        clauses.append("country = ?")
        params.append(filters["country"])
    if filters.get("variety"):
        clauses.append("variety = ?")
        params.append(filters["variety"])
    if filters.get("color"):
        clauses.append("color = ?")
        params.append(filters["color"])
    if filters.get("price_min") is not None:
        clauses.append("price >= ?")
        params.append(filters["price_min"])
    if filters.get("price_max") is not None:
        clauses.append("price <= ?")
        params.append(filters["price_max"])
    if filters.get("points_min") is not None:
        clauses.append("points >= ?")
        params.append(filters["points_min"])
    if filters.get("points_max") is not None:
        clauses.append("points <= ?")
        params.append(filters["points_max"])
    where = " AND ".join(clauses) if clauses else "TRUE"
    return where, params


def keyword_candidates(filters: dict, query_text: str, top_n: int = 100) -> list[int]:
    where, params = _build_where_clause(filters)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    # INSTALL as well as LOAD: the FTS *index data* travels inside the .duckdb
    # file, but the extension *binary* does not -- it's a platform-specific
    # shared library DuckDB fetches into ~/.duckdb/extensions/. A machine that
    # only ever reads the prebuilt database (e.g. a fresh deploy container,
    # different OS/arch than where the index was built) has never installed it,
    # so LOAD alone fails. INSTALL is a cheap no-op once cached.
    con.execute("INSTALL fts")
    con.execute("LOAD fts")
    sql = f"""
        SELECT w.id
        FROM wines w
        JOIN (
            SELECT id, fts_main_wines.match_bm25(id, ?) AS score FROM wines
        ) fts ON fts.id = w.id
        WHERE {where} AND fts.score IS NOT NULL
        ORDER BY fts.score DESC
        LIMIT {top_n}
    """
    rows = con.execute(sql, [query_text] + params).fetchall()
    con.close()
    return [r[0] for r in rows]


def vector_candidates(filters: dict, query_text: str, top_n: int = 100) -> list[int]:
    where, params = _build_where_clause(filters)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    eligible_ids = np.array(
        [r[0] for r in con.execute(f"SELECT id FROM wines WHERE {where}", params).fetchall()]
    )
    con.close()
    if len(eligible_ids) == 0:
        return []

    query_emb = _model.encode(
        [QUERY_PREFIX + query_text], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)[0]

    similarities = _embeddings[eligible_ids] @ query_emb
    top_k = min(top_n, len(eligible_ids))
    top_positions = np.argpartition(-similarities, top_k - 1)[:top_k]
    top_positions = top_positions[np.argsort(-similarities[top_positions])]

    return eligible_ids[top_positions].tolist()


def hybrid_candidates(filters: dict, query_text: str, top_n: int = 100) -> tuple[list[int], list[int]]:
    return keyword_candidates(filters, query_text, top_n), vector_candidates(filters, query_text, top_n)
