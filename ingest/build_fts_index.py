"""Phase 3: build a DuckDB full-text-search (BM25) index on `description`.

Structured filtering (country/variety/price/points/province) already works
via plain SQL on the `wines` table from Phase 1 -- no extra index needed for
that. This script adds the lexical/keyword half: real BM25 scoring on the
free-text description field, persisted inside the same wine.duckdb file.
"""

import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "index" / "wine.duckdb"


def main() -> None:
    con = duckdb.connect(str(DB_PATH))

    con.execute("INSTALL fts")
    con.execute("LOAD fts")
    con.execute("PRAGMA create_fts_index('wines', 'id', 'description', overwrite=1)")
    print("Built FTS index on wines.description")

    # --- Checkpoint 1: structured filter, no keyword search involved ---
    print("\nFilter query: country='Italy' AND points>=90 AND price<20")
    filtered = con.execute(
        """
        SELECT title, points, price
        FROM wines
        WHERE country = 'Italy' AND points >= 90 AND price < 20
        ORDER BY points DESC
        LIMIT 5
        """
    ).fetchall()
    for row in filtered:
        print(f"  {row}")

    # --- Checkpoint 2: BM25 keyword search, no structured filter involved ---
    print("\nBM25 query: 'tannic cherry'")
    bm25 = con.execute(
        """
        SELECT w.title, fts.score
        FROM (
            SELECT id, fts_main_wines.match_bm25(id, 'tannic cherry') AS score
            FROM wines
        ) fts
        JOIN wines w ON w.id = fts.id
        WHERE fts.score IS NOT NULL
        ORDER BY fts.score DESC
        LIMIT 5
        """
    ).fetchall()
    for row in bm25:
        print(f"  {row}")

    con.close()


if __name__ == "__main__":
    main()
