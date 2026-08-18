"""Phase 4+5 checkpoint: hand-trace a query end-to-end through query
understanding -> hybrid retrieval -> RRF fusion -> cross-encoder rerank,
printing the RRF-only top-10 next to the reranked top-10 so the reordering
effect of Phase 5 is directly visible.
"""

import pathlib
import sys

import duckdb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fusion import reciprocal_rank_fusion
from query_understanding import understand_query
from rerank import rerank
from retrieval import hybrid_candidates

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "index" / "wine.duckdb"


def _print_ranked(con, ranked: list[tuple[int, float]], n: int = 10) -> None:
    for doc_id, score in ranked[:n]:
        title, country, variety, price, points = con.execute(
            "SELECT title, country, variety, price, points FROM wines WHERE id = ?", [doc_id]
        ).fetchone()
        print(f"  [{score:.4f}] {title} | {country} | {variety} | ${price} | {points}pts")


def main() -> None:
    raw_query = sys.argv[1] if len(sys.argv) > 1 else "cheap Italian red under $20, 90+ points"
    print(f"Query: {raw_query!r}\n")

    filters = understand_query(raw_query)
    print("Extracted filters:")
    for k, v in filters.items():
        print(f"  {k}: {v}")

    keyword_ids, vector_ids = hybrid_candidates(filters, filters["query"], top_n=100)
    print(f"\nKeyword candidates: {len(keyword_ids)}  Vector candidates: {len(vector_ids)}")

    fused = reciprocal_rank_fusion(keyword_ids, vector_ids)
    fused_ids = [doc_id for doc_id, _ in fused]

    reranked = rerank(raw_query, fused_ids, top_n=50)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    print("\n=== RRF-only top 10 ===")
    _print_ranked(con, fused)
    print("\n=== Reranked top 10 (cross-encoder over RRF's top 50) ===")
    _print_ranked(con, reranked)
    con.close()


if __name__ == "__main__":
    main()
