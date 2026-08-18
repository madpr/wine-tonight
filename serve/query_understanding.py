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

import logging
import pathlib
import re

import anthropic
import duckdb
from dotenv import load_dotenv

from tracing import traced

logger = logging.getLogger("wine.query_understanding")

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
_COUNTRY_LOOKUP = {c.lower(): c for c in _COUNTRIES}
_VARIETY_LOOKUP = {v.lower(): v for v in _VARIETIES}

# Adjectival forms the rule-based fallback needs to resolve, since "Italian"
# never appears in the country column. Not exhaustive -- covers the countries
# with meaningful corpus volume.
_COUNTRY_ADJECTIVES = {
    "italian": "Italy", "french": "France", "spanish": "Spain",
    "portuguese": "Portugal", "german": "Germany", "austrian": "Austria",
    "australian": "Australia", "argentinian": "Argentina", "argentine": "Argentina",
    "chilean": "Chile", "american": "US", "californian": "US", "californian's": "US",
    "new zealand": "New Zealand", "south african": "South Africa",
    "greek": "Greece", "israeli": "Israel", "hungarian": "Hungary",
    "romanian": "Romania", "bulgarian": "Bulgaria", "canadian": "Canada",
    "uruguayan": "Uruguay", "mexican": "Mexico", "moldovan": "Moldova",
    "slovenian": "Slovenia", "croatian": "Croatia", "georgian": "Georgia",
    "turkish": "Turkey", "lebanese": "Lebanon", "brazilian": "Brazil",
    # US place names, since the corpus stores country as "US" and queries name
    # the state or region far more often than the country.
    "california": "US", "napa": "US", "oregon": "US", "sonoma": "US",
    "washington": "US", "willamette": "US", "paso robles": "US",
    "finger lakes": "US", "columbia valley": "US", "russian river": "US",
}

_COLOR_KEYWORDS = {
    "red": "red", "white": "white", "rose": "rose", "rosé": "rose", "blush": "rose",
    "sparkling": "sparkling", "champagne": "sparkling", "bubbly": "sparkling",
    "prosecco": "sparkling", "cava": "sparkling",
    "dessert": "dessert", "port": "fortified", "sherry": "fortified",
    "fortified": "fortified", "madeira": "fortified", "orange wine": "orange",
}

_EMPTY_FILTERS = {
    "country": None, "variety": None, "color": None,
    "price_min": None, "price_max": None, "points_min": None, "points_max": None,
}

# Bounded so a slow or unreachable API degrades to the rule-based path in
# seconds. The SDK default is a 10-minute timeout with 2 retries -- up to ~30
# minutes of hanging on a web request.
LLM_TIMEOUT_S = 6.0
LLM_MAX_RETRIES = 1
CACHE_SIZE = 512

# Safe to construct without credentials present -- the SDK resolves auth at
# request time, not here, so a missing key surfaces per-query rather than
# breaking startup.
_client = anthropic.Anthropic(timeout=LLM_TIMEOUT_S, max_retries=LLM_MAX_RETRIES)


@traced
def extract_filters_rules(raw_query: str) -> dict:
    """Rule-based filter extraction: no network call, no cost, no failure mode.

    Serves as the fallback when the LLM is slow, erroring, or unconfigured.
    Deliberately conservative -- a missed filter degrades to a broader result
    set, while a wrong one returns the wrong wines or nothing at all.
    """
    text = raw_query.lower()
    filters = dict(_EMPTY_FILTERS)
    filters["query"] = raw_query

    # price: "under $20", "below 30", "less than $25", "over $50", "$15 or less"
    for pattern, field in [
        (r"(?:under|below|less than|cheaper than|max(?:imum)?|up to)\s*\$?\s*(\d+)", "price_max"),
        (r"\$?\s*(\d+)\s*(?:or less|or under|and under)", "price_max"),
        (r"(?:over|above|more than|at least|min(?:imum)?)\s*\$\s*(\d+)", "price_min"),
    ]:
        match = re.search(pattern, text)
        if match:
            filters[field] = float(match.group(1))

    # points: "90+ points", "90 points or better", "at least 92", "rated 95"
    for pattern in [
        r"(\d{2,3})\s*\+\s*(?:points?|pts?)",
        r"(\d{2,3})\s*(?:points?|pts?)\s*(?:or (?:better|higher|above|more))",
        r"(?:at least|minimum|min|over|above)\s*(\d{2,3})\s*(?:points?|pts?)",
        r"rated\s*(?:at least\s*)?(\d{2,3})",
        r"(\d{2,3})\s*(?:points?|pts?)\s*(?:or\s*)?(?:up|plus)",
    ]:
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))
            if 80 <= value <= 100:  # the published scale; anything else is a price
                filters["points_min"] = float(value)
                break

    # country: exact name, then adjectival form
    for name, canonical in _COUNTRY_LOOKUP.items():
        if re.search(rf"\b{re.escape(name)}\b", text):
            filters["country"] = canonical
            break
    if filters["country"] is None:
        for adjective, canonical in _COUNTRY_ADJECTIVES.items():
            if re.search(rf"\b{re.escape(adjective)}\b", text):
                filters["country"] = canonical
                break

    # variety: longest match wins, so "Cabernet Sauvignon" beats "Cabernet"
    for name in sorted(_VARIETY_LOOKUP, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", text):
            filters["variety"] = _VARIETY_LOOKUP[name]
            break

    # color: explicit keyword, else inferred from the matched variety
    for keyword, color in _COLOR_KEYWORDS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            filters["color"] = color
            break
    if filters["color"] is None and filters["variety"] is not None:
        filters["color"] = _variety_color(filters["variety"])

    return filters


def _variety_color(variety: str) -> str | None:
    """Look up the derived colour for a variety (see ingest/classify_wine_color.py)."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    row = con.execute("SELECT color FROM variety_color WHERE variety = ?", [variety]).fetchone()
    con.close()
    return row[0] if row and row[0] != "other" else None


@traced
def understand_query(raw_query: str, allow_llm: bool = True) -> dict:
    """Extract filters, degrading rather than failing.

    Tiers: cache -> LLM (bounded timeout) -> rule-based -> raw query with no
    filters. The LLM is an enhancement, not a dependency: a search must still
    return wines when it is slow, erroring, or unconfigured, because BM25 and
    vector retrieval work perfectly well on the raw string.

    Generalises what the design already did for long-tail varieties -- anything
    that can't become a hard filter falls through to free-text search.
    """
    normalized = " ".join(raw_query.lower().split())

    if allow_llm:
        cached = _cached_llm_filters(normalized)
        if cached is not None:
            return {**cached, "_source": "cache"}
        try:
            filters = _llm_filters(raw_query)
            _store_in_cache(normalized, filters)
            return {**filters, "_source": "llm"}
        except Exception as exc:
            logger.warning(
                "query understanding fell back to rules (%s: %s)", type(exc).__name__, exc
            )

    # `_source` rides along so callers can surface that the search ran degraded.
    # _build_where_clause reads only known filter keys, so the extra one is inert.
    return {**extract_filters_rules(raw_query), "_source": "rules"}


_cache: dict[str, dict] = {}


def _cached_llm_filters(normalized: str) -> dict | None:
    return _cache.get(normalized)


def _store_in_cache(normalized: str, filters: dict) -> None:
    # Plain dict with a size cap rather than functools.lru_cache: the cached
    # value must be inspectable and clearable from tests, and lru_cache would
    # also cache the exceptions-as-control-flow path we deliberately avoid.
    if len(_cache) >= CACHE_SIZE:
        _cache.pop(next(iter(_cache)))
    _cache[normalized] = filters


def cache_stats() -> dict:
    return {"entries": len(_cache), "capacity": CACHE_SIZE}


@traced
def _llm_filters(raw_query: str) -> dict:
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
