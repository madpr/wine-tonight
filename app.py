"""Gradio entry point for the Hugging Face Space.

`import spaces` must come first: it monkey-patches torch.cuda.*, and once CUDA
is initialized in the main process it's too late. Everything under serve/
imports torch transitively.

This Space sits on ZeroGPU hardware because free accounts can't create
cpu-basic Spaces, but none of the work here is GPU work -- a 33M-param
embedding model, a 22M-param cross-encoder, and a numpy dot product all run
fine on CPU. ZeroGPU refuses to start without at least one decorated
function, so `_noop` below exists solely to satisfy that, and no real work
goes inside it. Nothing ever requests a GPU, so no quota is burned.
"""

import spaces  # noqa: F401  -- must precede any torch import

import pathlib
import sys

import duckdb
import gradio as gr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "serve"))

from fusion import reciprocal_rank_fusion
from query_understanding import understand_query
from rerank import rerank
from retrieval import hybrid_candidates

ROOT = pathlib.Path(__file__).resolve().parent
DB_PATH = ROOT / "index" / "wine.duckdb"

CANDIDATES_PER_PATH = 100
RERANK_TOP_N = 50
RESULT_LIMIT = 10

# Wine Enthusiast only publishes reviews scoring 80+, so 80-100 is the whole
# practical range and the dataset mean sits around 88.
RATING_BANDS = [
    (98, "Classic"),
    (94, "Superb"),
    (90, "Excellent"),
    (87, "Very Good"),
    (83, "Good"),
    (80, "Acceptable"),
]

FILTER_LABELS = {
    "country": "Country",
    "variety": "Grape",
    "color": "Style",
    "price_min": "Min price",
    "price_max": "Max price",
    "points_min": "Min rating",
    "points_max": "Max rating",
}


@spaces.GPU(duration=1)
def _noop():
    """ZeroGPU requires at least one decorated function. Never called."""
    pass


def _rating_label(points: int) -> str:
    for threshold, label in RATING_BANDS:
        if points >= threshold:
            return label
    return ""


def _format_filters(filters: dict, keyword_count: int, vector_count: int) -> str:
    parts = []
    for key, label in FILTER_LABELS.items():
        value = filters.get(key)
        if value is None:
            continue
        shown = f"${value:g}" if key.startswith("price") else value
        parts.append(f"**{label}:** {shown}")

    lines = ["### How your query was interpreted", ""]
    lines.append(" · ".join(parts) if parts else "_No hard filters — ranked purely on meaning._")

    residual = filters.get("query")
    if residual:
        lines.append("")
        lines.append(f"**Meaning search:** “{residual}”")

    lines.append("")
    lines.append(
        f"Matched **{keyword_count}** wines on exact keywords and **{vector_count}** "
        "on meaning, then merged and re-ranked both lists."
    )
    return "\n".join(lines)


def _format_results(rows: list[tuple]) -> str:
    if not rows:
        return (
            "### No matches\n\n"
            "No wines satisfied every filter. Try relaxing one — a high minimum "
            "rating combined with a low maximum price is the usual culprit."
        )

    lines = ["### Best matches", ""]
    for i, (title, description, country, province, variety, color, price, points) in enumerate(rows, 1):
        label = _rating_label(points)
        meta = " · ".join(str(v) for v in (variety, color, province, country) if v)
        price_text = f"${price:g}" if price is not None else "price not listed"
        lines.append(f"**{i}. {title}**  ")
        lines.append(f"`{points}/100 {label}` · {meta} · {price_text}  ")
        lines.append(f"{description}")
        lines.append("")
    return "\n".join(lines)


def search_wines(query: str) -> tuple[str, str]:
    """Search 129,971 wine reviews and return the best matches.

    Runs a hybrid retrieval pipeline: an LLM converts the natural-language
    request into structured filters plus a semantic query, then BM25 keyword
    search and vector similarity each retrieve candidates from the filtered
    set, the two ranked lists are merged with Reciprocal Rank Fusion, and a
    cross-encoder reranks the top candidates.

    Args:
        query: A natural-language wine request, e.g. "cheap Italian red
            under $20 with 90+ points" or "elegant French white with citrus".

    Returns:
        A tuple of (interpretation summary, formatted results), both Markdown.
    """
    query = (query or "").strip()
    if not query:
        return "", "_Enter a description of the wine you want._"

    filters = understand_query(query)
    keyword_ids, vector_ids = hybrid_candidates(
        filters, filters["query"], top_n=CANDIDATES_PER_PATH
    )

    fused = reciprocal_rank_fusion(keyword_ids, vector_ids)
    ranked = rerank(query, [doc_id for doc_id, _ in fused], top_n=RERANK_TOP_N)

    rows = []
    if ranked:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        for doc_id, _score in ranked[:RESULT_LIMIT]:
            row = con.execute(
                """
                SELECT title, description, country, province, variety, color, price, points
                FROM wines WHERE id = ?
                """,
                [doc_id],
            ).fetchone()
            if row:
                rows.append(row)
        con.close()

    return (
        _format_filters(filters, len(keyword_ids), len(vector_ids)),
        _format_results(rows),
    )


RATING_KEY = """
Ratings are **Wine Enthusiast** scores. Only wines rating 80+ are ever
published, so **80–100 is the full practical range** — and the average wine
here sits around **88**.

| Score | Meaning |
|---|---|
| 98–100 | Classic — the pinnacle of quality |
| 94–97 | Superb — a great achievement |
| 90–93 | Excellent — highly recommended |
| 87–89 | Very Good — often good value |
| 83–86 | Good — solid everyday drinking |
| 80–82 | Acceptable — fine in casual settings |

Rating and price correlate only loosely (0.42), so genuine bargains at 90+ exist.
"""

with gr.Blocks(title="What wine do you want to drink tonight?") as demo:
    gr.Markdown(
        "# 🍷 What wine do you want to drink tonight?\n"
        "Describe it however you like. Your request is split into hard filters "
        "(country, grape, style, price, rating) and a meaning-based search over "
        "129,971 tasting notes — then re-ranked for relevance."
    )

    with gr.Row():
        query_box = gr.Textbox(
            label="",
            placeholder="e.g. cheap Italian red under $20, 90+ points",
            scale=5,
            submit_btn=True,
        )
    search_btn = gr.Button("Search", variant="primary")

    gr.Examples(
        examples=[
            ["cheap Italian red under $20, 90+ points"],
            ["elegant French white with citrus and minerality"],
            ["bold Argentinian Malbec with dark fruit"],
            ["sparkling wine for a celebration under $40"],
            ["earthy Burgundy Pinot Noir with cherry and leather"],
        ],
        inputs=[query_box],
    )

    interpretation = gr.Markdown()
    results = gr.Markdown()

    with gr.Accordion("What does the rating mean?", open=False):
        gr.Markdown(RATING_KEY)

    for trigger in (query_box.submit, search_btn.click):
        trigger(fn=search_wines, inputs=[query_box], outputs=[interpretation, results])

if __name__ == "__main__":
    demo.launch(mcp_server=True)
