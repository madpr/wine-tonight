"""Phase 4: turn a natural-language wine query into structured filters +
a residual semantic query, using claude-haiku-4-5.

Haiku is used deliberately -- this is narrow structured extraction against
a fixed schema, not open-ended reasoning, so the cheapest current model is
the right fit (see PLAN.md Phase 4).

`color` is a derived field (see ingest/classify_wine_color.py) added after
noticing that a query like "red wine" had no hard filter to enforce it --
"red" only ever reached BM25/vector search as free text, so white and rose
wines could still rank highly. This makes wine color an actual filter.
"""

import pathlib

import anthropic
import duckdb
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "index" / "wine.duckdb"

load_dotenv(ROOT / ".env")

MODEL = "claude-haiku-4-5"

FILTER_TOOL = {
    "name": "extract_wine_filters",
    "description": (
        "Extract structured filters and a residual semantic search query "
        "from a natural-language wine request."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "country": {
                "type": ["string", "null"],
                "description": "Exact country name from KNOWN COUNTRIES, or null if not specified.",
            },
            "variety": {
                "type": ["string", "null"],
                "description": "Exact grape variety from KNOWN VARIETIES, or null if not specified.",
            },
            "color": {
                "anyOf": [
                    {
                        "type": "string",
                        "enum": ["red", "white", "rose", "sparkling", "dessert", "fortified", "orange"],
                    },
                    {"type": "null"},
                ],
                "description": (
                    "Wine color/style, if the request implies one (e.g. 'red' -> 'red', "
                    "'bubbly'/'champagne' -> 'sparkling', 'blush' -> 'rose'). Null if not implied."
                ),
            },
            "price_min": {"type": ["number", "null"]},
            "price_max": {"type": ["number", "null"]},
            "points_min": {"type": ["number", "null"]},
            "points_max": {"type": ["number", "null"]},
            "query": {
                "type": "string",
                "description": (
                    "Residual free text describing taste/style, with filter "
                    "mentions removed. If nothing remains, repeat the original query."
                ),
            },
        },
        "required": [
            "country",
            "variety",
            "color",
            "price_min",
            "price_max",
            "points_min",
            "points_max",
            "query",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}


def _load_known_values() -> tuple[list[str], list[str]]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    countries = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT country FROM wines WHERE country IS NOT NULL ORDER BY country"
        ).fetchall()
    ]
    # Top 200 varieties by frequency keeps the prompt small; long-tail varieties
    # fall through to the free-text query instead of a hard filter.
    varieties = [
        r[0]
        for r in con.execute(
            "SELECT variety FROM wines WHERE variety IS NOT NULL "
            "GROUP BY variety ORDER BY count(*) DESC LIMIT 200"
        ).fetchall()
    ]
    con.close()
    return countries, varieties


_COUNTRIES, _VARIETIES = _load_known_values()
_client = anthropic.Anthropic()


def understand_query(raw_query: str) -> dict:
    system = (
        "You translate natural-language wine search requests into structured filters. "
        "Only set country/variety if the request clearly implies one of the KNOWN VALUES "
        "below -- map loose phrasing onto the closest known value (e.g. 'Italian' -> 'Italy'). "
        "Set color independently from variety when the request implies a color/style -- "
        "e.g. 'red wine' has no variety but color='red'; a named variety like 'Pinot Noir' "
        "implies color='red' too, even if not stated explicitly. "
        "Leave a field null if not mentioned. Never invent a value outside the known lists.\n\n"
        f"KNOWN COUNTRIES: {', '.join(_COUNTRIES)}\n\n"
        f"KNOWN VARIETIES (top 200 by frequency): {', '.join(_VARIETIES)}"
    )

    response = _client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=system,
        tools=[FILTER_TOOL],
        tool_choice={"type": "tool", "name": "extract_wine_filters"},
        messages=[{"role": "user", "content": raw_query}],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    raise RuntimeError("Model did not return a tool_use block")


if __name__ == "__main__":
    import json
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "cheap Italian red under $20, 90+ points"
    print(json.dumps(understand_query(q), indent=2))
