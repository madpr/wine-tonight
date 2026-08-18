"""Phase 6: FastAPI app wiring the whole pipeline behind one endpoint.

POST /search runs: LLM query understanding -> hybrid retrieval (filtered
BM25 + filtered vector similarity) -> RRF fusion -> cross-encoder rerank.

Every module-level import here loads a model or index into memory
(bge-small for embeddings, ms-marco-MiniLM for reranking, the embeddings
matrix), which is why startup takes a few seconds. That cost is paid once
at boot, not per request -- the whole point of loading them at import time
rather than inside the request handler.
"""

import pathlib

import duckdb
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from fusion import reciprocal_rank_fusion
from query_understanding import understand_query
from rerank import rerank
from retrieval import hybrid_candidates

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "index" / "wine.duckdb"
STATIC_DIR = ROOT / "static"

CANDIDATES_PER_PATH = 100
RERANK_TOP_N = 50

app = FastAPI(title="Wine Hybrid Search")


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


class SearchResult(BaseModel):
    id: int
    title: str
    description: str
    country: str | None
    province: str | None
    variety: str | None
    color: str | None
    price: float | None
    points: int
    score: float


class SearchResponse(BaseModel):
    query: str
    filters: dict
    keyword_candidate_count: int
    vector_candidate_count: int
    results: list[SearchResult]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    filters = understand_query(request.query)

    keyword_ids, vector_ids = hybrid_candidates(
        filters, filters["query"], top_n=CANDIDATES_PER_PATH
    )

    fused = reciprocal_rank_fusion(keyword_ids, vector_ids)
    fused_ids = [doc_id for doc_id, _ in fused]

    # Rerank only the top RERANK_TOP_N of the fused list -- running the
    # cross-encoder over every candidate would defeat the two-stage design.
    ranked = rerank(request.query, fused_ids, top_n=RERANK_TOP_N)

    results = []
    if ranked:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        for doc_id, score in ranked[: request.limit]:
            row = con.execute(
                """
                SELECT id, title, description, country, province, variety, color, price, points
                FROM wines WHERE id = ?
                """,
                [doc_id],
            ).fetchone()
            if row is None:
                continue
            results.append(
                SearchResult(
                    id=row[0],
                    title=row[1],
                    description=row[2],
                    country=row[3],
                    province=row[4],
                    variety=row[5],
                    color=row[6],
                    price=row[7],
                    points=row[8],
                    score=float(score),
                )
            )
        con.close()

    return SearchResponse(
        query=request.query,
        filters=filters,
        keyword_candidate_count=len(keyword_ids),
        vector_candidate_count=len(vector_ids),
        results=results,
    )
