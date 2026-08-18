"""Phase 1: load the raw wine reviews CSV into a persistent DuckDB table.

Static, one-time batch load (Kaggle CSV) -- no freshness/reindex-lag concerns
here. A live data source would need incremental ingestion instead.
"""

import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "winemag-data-130k-v2.csv"
DB_PATH = ROOT / "index" / "wine.duckdb"


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    # Trim every VARCHAR column at load time. The raw CSV has a few values
    # with stray leading/trailing whitespace (e.g. 'Tintilia ' vs 'Tintilia')
    # that silently break exact-match filters (WHERE variety = ?) downstream.
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_csv_auto('{CSV_PATH.as_posix()}')"
    ).fetchall()
    select_parts = []
    for col_name, col_type, *_ in schema:
        if col_name == "column00":
            continue
        if col_type == "VARCHAR":
            select_parts.append(f'TRIM("{col_name}") AS "{col_name}"')
        else:
            select_parts.append(f'"{col_name}"')

    con.execute("DROP TABLE IF EXISTS wines")
    con.execute(
        f"""
        CREATE TABLE wines AS
        SELECT column00 AS id, {', '.join(select_parts)}
        FROM read_csv_auto('{CSV_PATH.as_posix()}')
        """
    )
    con.execute("ALTER TABLE wines ADD PRIMARY KEY (id)")

    row_count = con.execute("SELECT count(*) FROM wines").fetchone()[0]
    null_id_count = con.execute("SELECT count(*) FROM wines WHERE id IS NULL").fetchone()[0]
    print(f"Loaded {row_count} rows into {DB_PATH} (null ids: {null_id_count})")

    print("\nNull counts by column:")
    columns = [r[0] for r in con.execute("DESCRIBE wines").fetchall()]
    for col in columns:
        n_null = con.execute(f'SELECT count(*) FROM wines WHERE "{col}" IS NULL').fetchone()[0]
        if n_null:
            print(f"  {col}: {n_null}")

    print("\nSpot check id=0:")
    print(con.execute("SELECT id, title, variety, country FROM wines WHERE id = 0").fetchall())

    con.close()


if __name__ == "__main__":
    main()
