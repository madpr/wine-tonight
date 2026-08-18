"""How much does the rule-based fallback recover when the LLM is unavailable?

The LLM sits in the critical path of every query: ~1920ms (about 80% of request
latency), ~$0.003, and a hard third-party dependency. serve/query_understanding
therefore degrades in tiers -- cache, then LLM with a bounded timeout, then
rule-based extraction, then raw query with no filters -- so a search still
returns wines when the API is slow, erroring or unconfigured.

This measures the rule tier against the LLM's output on the frozen query set.
The LLM is the *reference*, not ground truth: it is itself non-deterministic and
occasionally wrong. The question is how much filtering capability survives a
fallback, per field, so the degradation is understood rather than assumed.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "serve"))

from query_understanding import extract_filters_rules

QUERY_SET = pathlib.Path(__file__).resolve().parent / "query_set.json"
FIELDS = ["country", "variety", "color", "price_min", "price_max", "points_min", "points_max"]


def main() -> None:
    records = json.loads(QUERY_SET.read_text())

    # per field: exact agreement, missed (LLM set it, rules didn't), spurious
    stats = {f: {"agree": 0, "missed": 0, "spurious": 0, "differ": 0, "both_none": 0} for f in FIELDS}
    per_query = []

    for record in records:
        llm = record["filters"]
        rules = extract_filters_rules(record["query"])
        row = {"query": record["query"], "issues": []}

        for field in FIELDS:
            a, b = llm.get(field), rules.get(field)
            if a is None and b is None:
                stats[field]["both_none"] += 1
            elif a == b:
                stats[field]["agree"] += 1
            elif a is not None and b is None:
                stats[field]["missed"] += 1
                row["issues"].append(f"missed {field}={a}")
            elif a is None and b is not None:
                stats[field]["spurious"] += 1
                row["issues"].append(f"spurious {field}={b}")
            else:
                stats[field]["differ"] += 1
                row["issues"].append(f"{field}: llm={a} rules={b}")

        per_query.append(row)

    print(f"{'field':>12} {'agree':>6} {'both∅':>6} {'missed':>7} {'spurious':>9} {'differ':>7}")
    print("-" * 56)
    for field in FIELDS:
        s = stats[field]
        print(f"{field:>12} {s['agree']:>6} {s['both_none']:>6} {s['missed']:>7} "
              f"{s['spurious']:>9} {s['differ']:>7}")

    total_set = sum(stats[f]["agree"] + stats[f]["missed"] + stats[f]["differ"] for f in FIELDS)
    total_agree = sum(stats[f]["agree"] for f in FIELDS)
    total_missed = sum(stats[f]["missed"] for f in FIELDS)
    total_differ = sum(stats[f]["differ"] for f in FIELDS)
    total_spurious = sum(stats[f]["spurious"] for f in FIELDS)

    print()
    print(f"of {total_set} filters the LLM set: rules matched {total_agree} "
          f"({total_agree / total_set:.0%}), missed {total_missed}, differed {total_differ}")
    print(f"rules also set {total_spurious} filters the LLM left unset")

    perfect = sum(1 for r in per_query if not r["issues"])
    print(f"queries with identical filters: {perfect}/{len(per_query)}")

    print()
    print("=== disagreements ===")
    for row in per_query:
        if row["issues"]:
            print(f"  {row['query'][:52]}")
            for issue in row["issues"]:
                print(f"      {issue}")


if __name__ == "__main__":
    main()
