# Query understanding

Turning `"cheap Italian red under $20, 90+ points"` into `{country: Italy, color: red, price_max: 20, points_min: 90}` plus a residual semantic query.

This is the only stage that calls out to a third party, the only per-query cost, and — at ~1920ms — about **80% of request latency**. So it gets a fallback chain rather than being a hard dependency.

## Tiers

```
cache  →  claude-haiku-4-5 (6s timeout)  →  rule-based  →  raw query, no filters
```

Each tier returns the same shape, plus a `_source` field so the UI can say when a search ran degraded. `_build_where_clause` reads only known filter keys, so `_source` is inert in SQL.

| Tier | Latency | Cost | When |
|---|---|---|---|
| cache | **0.006 ms** | $0 | repeat query (case/whitespace-normalized) |
| Haiku | ~1900 ms | ~$0.003 | default |
| rules | ~0 ms | $0 | LLM slow, erroring, or unconfigured |
| raw query | 0 ms | $0 | genuine bug in the above |

**A search must never return zero wines because query understanding failed** — BM25 and vector retrieval work fine on the raw string. This generalizes what the design already did for long-tail varieties: whatever can't become a hard filter falls through to free-text search.

### What was wrong before

- `api.py` had no error handling — an LLM failure was an unhandled 500.
- `app.py` "handled" it by returning **zero results**.
- No timeout. The SDK default is 10 minutes × 2 retries — up to ~30 minutes of hanging on a web request.
- No caching — identical queries re-paid both latency and cost.

## The rule-based tier

Regex for price and points; matching against the real column values for country, variety and color, including adjectival and US place-name forms (`"Italian"` → `Italy`, `"Napa"` → `US`). Color is inferred from a matched variety when not stated.

Measured against Haiku on the frozen query set (`eval/compare_query_understanding.py`):

| | |
|---|---|
| Filters matched | **95%** (42/44 of what Haiku set) |
| Queries identical | **18/20** |
| Wrong values | **0** |
| Spurious filters | **0** |

The zeros matter more than the 95%. **The rules miss filters rather than invent them**, so degradation broadens the result set instead of returning the wrong wines. A missed `color=red` shows some whites; a *spurious* `country=Argentina` would wrongly discard 97% of the corpus.

The 2 misses are genuine inference — `"something crisp and refreshing for a hot afternoon"` → white, `"big tannic wine to pair with steak"` → red. There's no keyword to match; that needs world knowledge, which is exactly what the LLM tier is for.

## Rejected: a local 1.5B model

Running weights ourselves would remove the API key, the per-query cost and the third-party dependency. `serve/local_llm.py` implements it — Qwen2.5-1.5B-Instruct with `outlines` constrained decoding, which masks the token distribution at each step so output is schema-valid *by construction* rather than by parsing and retrying.

Measured (`eval/compare_local_llm.py`):

| | local 1.5B | rules | Haiku |
|---|---|---|---|
| Filters matched | 86% | **95%** | — |
| Wrong values | 1 | **0** | — |
| Spurious filters | **17** | **0** | — |
| Hard crashes | **3/20 (15%)** | 0 | 0 |
| Identical to Haiku | 4/17 | **18/20** | — |
| Median latency | 1350 ms | **~0 ms** | 1900 ms |

**Worse than 40 lines of regex on every axis that matters.** Two disqualifiers:

1. **15% hard failure rate.** `outlines` raises `ValueError: No next state found for the current state` — the decoding guide reaches an unreachable state. A backend that crashes on one query in seven isn't a fallback.
2. **It fabricates filters.** `"earthy notes of leather and dried cherry"` → `variety=Cabernet Sauvignon`; no variety was mentioned. The schema's `required` array forces every field present, and a small model fills rather than nulls them.

The underlying reason: this task's entire value from an LLM is *inference*. A 1.5B model can't do that reliably, so it delivers neither the rules' precision nor Haiku's inference — the worst of both, at 1350ms.

### A methodology note

The first run scored much worse (24 spurious, 3 wrong) because the schema declared `country` and `variety` as free-form strings. Constrained decoding can only enforce what the schema states, so the model was free to emit anything — one run produced `"United States林业"` as a country.

`build_schema()` now makes them enums over the real column values, which removes that class of error entirely. **Listing the valid values in the prompt does nothing to prevent invalid output; only the schema does.** Worth remembering when using constrained decoding at all: putting a constraint in the prose and not in the grammar wastes the mechanism.

## Cost

~2,340 input + ~100 output tokens per Haiku call ≈ **$0.003/query**; repeats are **$0** from cache.

An unexplored optimization: the system prompt (country list, top-200 varieties, tool schema) is stable across queries and is most of those 2,340 tokens, making it a prompt-caching candidate. That would cut both the dominant latency component and the per-query cost.
