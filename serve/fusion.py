"""Phase 4: Reciprocal Rank Fusion (RRF).

BM25 scores and cosine similarities aren't on comparable scales, so fusion
uses rank position instead of raw score -- the standard, simplest correct
answer for combining independently-ranked candidate lists.
"""

from tracing import traced


@traced
def reciprocal_rank_fusion(*ranked_lists: list[int], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
