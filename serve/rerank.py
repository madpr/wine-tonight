"""Phase 5: cross-encoder reranking.

A bi-encoder (Phase 2's embedding model) embeds query and document
independently, then compares vectors -- fast, scalable, but the model
never sees the query and document together. A cross-encoder feeds the
(query, document) pair *jointly* through a transformer, so it can attend
across both texts at once -- much more accurate, but too slow to run over
the whole corpus. That's why it only reranks the top-K survivors of RRF
fusion, never the full 130K rows.
"""

import pathlib

import duckdb
import torch
from sentence_transformers import CrossEncoder

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "index" / "wine.duckdb"

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_device = "mps" if torch.backends.mps.is_available() else "cpu"
_cross_encoder = CrossEncoder(MODEL_NAME, device=_device)


def rerank(query_text: str, candidate_ids: list[int], top_n: int = 50) -> list[tuple[int, float]]:
    """Re-score the top `top_n` candidate_ids (already ordered by RRF) with
    the cross-encoder, and return them re-sorted by that score."""
    ids_to_score = candidate_ids[:top_n]
    if not ids_to_score:
        return []

    con = duckdb.connect(str(DB_PATH), read_only=True)
    placeholders = ",".join(str(i) for i in ids_to_score)
    rows = con.execute(f"SELECT id, description FROM wines WHERE id IN ({placeholders})").fetchall()
    con.close()
    id_to_description = dict(rows)

    # keep only ids that actually resolved to a row, preserving RRF order for the pairs list
    ordered_ids = [i for i in ids_to_score if i in id_to_description]
    pairs = [(query_text, id_to_description[i]) for i in ordered_ids]

    scores = _cross_encoder.predict(pairs)

    return sorted(zip(ordered_ids, scores), key=lambda pair: pair[1], reverse=True)
