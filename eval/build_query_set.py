"""Freeze a query set with its extracted filters, once.

Why this exists as a separate step: understand_query is non-deterministic --
the same input has produced "cheap Italian red", "red wine", and the full
original string as the residual query on different calls. Calling it inside the
comparison would measure LLM variance rather than retrieval behaviour, so
filters are extracted once here and reused for every path under test.

Queries deliberately span filter selectivity, which is the variable expected to
drive the exact-vs-approximate difference: ANN failed before precisely when
filters were narrow.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "serve"))

import duckdb

from query_understanding import understand_query
from retrieval import _build_where_clause

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "index" / "wine.duckdb"
OUT_PATH = pathlib.Path(__file__).resolve().parent / "query_set.json"

QUERIES = [
    # --- no hard filters: pure semantic ---
    "earthy notes of leather and dried cherry",
    "something crisp and refreshing for a hot afternoon",
    "big tannic wine to pair with steak",
    "elegant and mineral driven with high acidity",
    # --- broad filters (country or colour only) ---
    "French red wine",
    "Italian white",
    "a bottle from Argentina",
    "sparkling wine",
    # --- mid selectivity ---
    "California Chardonnay with oak and butter",
    "Spanish red under $30",
    "German Riesling with racy acidity",
    "Oregon Pinot Noir, 90 points or better",
    # --- narrow filters: where ANN previously returned nothing ---
    "cheap Italian red under $20, 90+ points",
    "Bordeaux-style red blend under $25 rated at least 92",
    "outstanding Napa Cabernet Sauvignon over 95 points",
    "bargain Portuguese red under $15 with 90+ points",
    # --- keyword-heavy: terms BM25 should catch verbatim ---
    "gooseberry and passionfruit Sauvignon Blanc",
    "brett and barnyard funk",
    "petrol note aged Riesling",
    "chocolate espresso Malbec",
]


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    records = []

    for query in QUERIES:
        filters = understand_query(query)
        where, params = _build_where_clause(filters)
        eligible = con.execute(f"SELECT count(*) FROM wines WHERE {where}", params).fetchone()[0]
        records.append({"query": query, "filters": filters, "eligible_count": eligible})
        set_filters = {k: v for k, v in filters.items() if v is not None and k != "query"}
        print(f"{eligible:>7,} eligible | {query}")
        print(f"          filters: {set_filters}")

    con.close()
    OUT_PATH.write_text(json.dumps(records, indent=2))
    print(f"\nWrote {len(records)} queries to {OUT_PATH}")


if __name__ == "__main__":
    main()
