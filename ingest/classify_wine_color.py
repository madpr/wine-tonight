"""Enrichment step: derive a `color` field for every wine from its `variety`.

The dataset has no color column. There are 707 distinct varieties, including
a long tail of single-occurrence, obscure grapes (e.g. "Zilavka",
"Karalahna") that a hand-curated mapping would likely miss or get wrong.
Since this classifies 707 unique variety *names* once -- not 130K rows --
it's a cheap one-time batch job, not a per-query cost. Uses claude-haiku-4-5
for the same reason as query_understanding.py: narrow classification against
a fixed enum, not open-ended reasoning.

Writes a variety_color lookup table, then joins `color` onto `wines`.
"""

import pathlib

import anthropic
import duckdb
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "index" / "wine.duckdb"

load_dotenv(ROOT / ".env")

MODEL = "claude-haiku-4-5"
BATCH_SIZE = 100
COLORS = ["red", "white", "rose", "sparkling", "dessert", "fortified", "orange", "other"]

CLASSIFY_TOOL = {
    "name": "classify_varieties",
    "description": "Classify each given grape variety by the color/style of wine it typically produces.",
    "input_schema": {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "variety": {"type": "string"},
                        "color": {"type": "string", "enum": COLORS},
                    },
                    "required": ["variety", "color"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["classifications"],
        "additionalProperties": False,
    },
    "strict": True,
}

_client = anthropic.Anthropic()


def _load_varieties() -> list[str]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute(
        "SELECT DISTINCT variety FROM wines WHERE variety IS NOT NULL ORDER BY variety"
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


def _classify_batch(varieties: list[str]) -> dict[str, str]:
    system = (
        "Classify each grape variety by the color/style of wine it typically produces. "
        "Use 'red' or 'white' for standard still wines, 'rose' for rose/blush, "
        "'sparkling' for Champagne-style, 'dessert' for late-harvest/ice wine/botrytized, "
        "'fortified' for Port/Sherry-style, 'orange' for skin-contact white, "
        "'other' only if genuinely unclassifiable (e.g. a generic blend name with no clear color, "
        "like 'Red Blend' -> red, 'White Blend' -> white, 'Rose' -> rose are NOT 'other'). "
        "Classify every single variety listed -- do not skip any."
    )
    response = _client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=system,
        tools=[CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "classify_varieties"},
        messages=[{"role": "user", "content": "\n".join(varieties)}],
    )
    for block in response.content:
        if block.type == "tool_use":
            return {c["variety"]: c["color"] for c in block.input["classifications"]}
    raise RuntimeError("Model did not return a tool_use block")


def main() -> None:
    varieties = _load_varieties()
    print(f"Classifying {len(varieties)} distinct varieties in batches of {BATCH_SIZE}...")

    variety_to_color: dict[str, str] = {}
    for i in range(0, len(varieties), BATCH_SIZE):
        batch = varieties[i : i + BATCH_SIZE]
        result = _classify_batch(batch)
        variety_to_color.update(result)
        print(f"  batch {i // BATCH_SIZE + 1}: classified {len(result)}/{len(batch)}")

    missing = [v for v in varieties if v not in variety_to_color]
    if missing:
        print(f"WARNING: {len(missing)} varieties got no classification, defaulting to 'other': {missing}")
        for v in missing:
            variety_to_color[v] = "other"

    con = duckdb.connect(str(DB_PATH))
    con.execute("DROP TABLE IF EXISTS variety_color")
    con.execute("CREATE TABLE variety_color (variety VARCHAR PRIMARY KEY, color VARCHAR)")
    con.executemany(
        "INSERT INTO variety_color VALUES (?, ?)", list(variety_to_color.items())
    )

    con.execute("ALTER TABLE wines DROP COLUMN IF EXISTS color")
    con.execute("ALTER TABLE wines ADD COLUMN color VARCHAR")
    con.execute(
        """
        UPDATE wines SET color = variety_color.color
        FROM variety_color
        WHERE wines.variety = variety_color.variety
        """
    )

    print("\nColor distribution across all wines:")
    for color, n in con.execute(
        "SELECT color, count(*) AS n FROM wines GROUP BY color ORDER BY n DESC"
    ).fetchall():
        print(f"  {color}: {n}")

    print("\nSpot checks:")
    for v in ["Pinot Noir", "Chardonnay", "Pinot Bianco", "Rosato", "Nebbiolo", "Riesling"]:
        row = con.execute("SELECT color FROM variety_color WHERE variety = ?", [v]).fetchone()
        print(f"  {v}: {row[0] if row else 'NOT FOUND'}")

    con.close()


if __name__ == "__main__":
    main()
