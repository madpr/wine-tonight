"""Local Qwen2.5-1.5B vs hosted Haiku for query understanding.

Third backend in the comparison, after the hosted LLM (the reference, frozen in
query_set.json) and the rule-based fallback (eval/compare_query_understanding).
The appeal of running weights locally is no API key, no per-query cost and no
third-party dependency in the request path; the question is what that costs in
extraction quality and reliability.

Reliability is measured, not assumed: constrained decoding can fail outright
(`outlines` raising on an unreachable guide state), and a backend that errors on
some fraction of queries is a different proposition from one that is merely less
accurate. Failures are counted as their own category rather than silently
skipped.
"""

import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "serve"))

import duckdb

from local_llm import MODEL_NAME, extract

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "index" / "wine.duckdb"
QUERY_SET = pathlib.Path(__file__).resolve().parent / "query_set.json"
FIELDS = ["country", "variety", "color", "price_min", "price_max", "points_min", "points_max"]
N_VARIETIES = 60  # a 1.5B model degrades on a very long enum; the hosted path uses 200


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    countries = [r[0] for r in con.execute(
        "SELECT DISTINCT country FROM wines WHERE country IS NOT NULL ORDER BY country"
    ).fetchall()]
    varieties = [r[0] for r in con.execute(
        "SELECT variety FROM wines WHERE variety IS NOT NULL "
        f"GROUP BY variety ORDER BY count(*) DESC LIMIT {N_VARIETIES}"
    ).fetchall()]
    con.close()

    records = json.loads(QUERY_SET.read_text())
    stats = {f: {"agree": 0, "missed": 0, "spurious": 0, "differ": 0, "both_none": 0}
             for f in FIELDS}
    timings, failures, rows = [], [], []

    print(f"model: {MODEL_NAME} | {len(varieties)} varieties in prompt\n")

    for record in records:
        started = time.perf_counter()
        try:
            local = extract(record["query"], countries, varieties)
        except Exception as exc:
            failures.append((record["query"], f"{type(exc).__name__}: {str(exc)[:80]}"))
            continue
        timings.append((time.perf_counter() - started) * 1000)

        reference = record["filters"]
        issues = []
        for field in FIELDS:
            a, b = reference.get(field), local.get(field)
            if a is None and b is None:
                stats[field]["both_none"] += 1
            elif a == b:
                stats[field]["agree"] += 1
            elif a is not None and b is None:
                stats[field]["missed"] += 1
                issues.append(f"missed {field}={a}")
            elif a is None and b is not None:
                stats[field]["spurious"] += 1
                issues.append(f"SPURIOUS {field}={b}")
            else:
                stats[field]["differ"] += 1
                issues.append(f"WRONG {field}: haiku={a} local={b}")
        rows.append({"query": record["query"], "issues": issues})

    print(f"{'field':>12} {'agree':>6} {'both∅':>6} {'missed':>7} {'spurious':>9} {'wrong':>6}")
    print("-" * 54)
    for field in FIELDS:
        s = stats[field]
        print(f"{field:>12} {s['agree']:>6} {s['both_none']:>6} {s['missed']:>7} "
              f"{s['spurious']:>9} {s['differ']:>6}")

    set_by_ref = sum(stats[f]["agree"] + stats[f]["missed"] + stats[f]["differ"] for f in FIELDS)
    agreed = sum(stats[f]["agree"] for f in FIELDS)
    spurious = sum(stats[f]["spurious"] for f in FIELDS)
    wrong = sum(stats[f]["differ"] for f in FIELDS)

    print()
    print(f"completed        : {len(timings)}/{len(records)} queries")
    print(f"hard failures    : {len(failures)}")
    if set_by_ref:
        print(f"filters matched  : {agreed}/{set_by_ref} ({agreed / set_by_ref:.0%}) of what Haiku set")
    print(f"wrong values     : {wrong}")
    print(f"spurious filters : {spurious}   (Haiku left these null)")
    perfect = sum(1 for r in rows if not r["issues"])
    print(f"identical queries: {perfect}/{len(rows)}")
    if timings:
        print(f"latency          : mean {statistics.mean(timings):.0f}ms  "
              f"median {statistics.median(timings):.0f}ms  max {max(timings):.0f}ms")

    if failures:
        print("\n=== hard failures ===")
        for query, err in failures:
            print(f"  {query[:46]}\n      {err}")

    print("\n=== disagreements ===")
    for row in rows:
        if row["issues"]:
            print(f"  {row['query'][:48]}")
            for issue in row["issues"]:
                print(f"      {issue}")


if __name__ == "__main__":
    main()
