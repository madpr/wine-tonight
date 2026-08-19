# The dataset

Kaggle [Wine Reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews) (`winemag-data-130k-v2.csv`) — 129,971 Wine Enthusiast reviews.

## Schema

| Column | Type | Nulls | Distinct | Role |
|---|---|---|---|---|
| `id` | bigint | 0 | 129,971 | primary key (the CSV's unnamed index column) |
| `description` | varchar | **0** | — | **the text that gets embedded and BM25-indexed** |
| `points` | bigint | 0 | 21 | rating, 80–100 — filter |
| `price` | double | 8,996 (6.9%) | 390 | $4–$3,300, median $25 — filter |
| `country` | varchar | 63 | 43 | filter |
| `variety` | varchar | 1 | 707 | grape — filter |
| `color` | varchar | 1 | 8 | **derived**, not in source — filter |
| `province` | varchar | 63 | 425 | available, unused |
| `region_1` | varchar | 21,247 (16%) | 1,229 | available, unused |
| `region_2` | varchar | 79,460 (61%) | 17 | too sparse to filter on |
| `title` | varchar | 0 | 118,840 | displayed |
| `designation` | varchar | 37,465 (29%) | 37,979 | near-unique; free text, not a facet |
| `winery` | varchar | 0 | 16,757 | too high-cardinality for LLM mapping |
| `taster_name` | varchar | 26,244 (20%) | 19 | unused |
| `taster_twitter_handle` | varchar | 31,213 | — | unused |

## Filters actually used

`country`, `variety`, `color`, `price` range, `points` range. Chosen for low null rates, enumerable value sets, and being the things people actually say out loud.

Deliberately excluded: `region_2` (61% null — filtering would silently drop most of the corpus), `designation` (near-unique free text), `winery` (16,757 values is too many to ground an LLM against reliably).

`province` is clean and would catch region-based queries ("a wine from Tuscany") that `country` can't resolve — a reasonable next addition.

## The ratings scale

`points` is Wine Enthusiast's 100-point score, as assigned by the reviewer.

| Range | Label | Count |
|---|---|---|
| 98–100 | Classic | 129 |
| 94–97 | Superb | 6,045 |
| 90–93 | Excellent | 42,871 |
| 87–89 | Very Good | 46,366 |
| 83–86 | Good | 31,946 |
| 80–82 | Acceptable | 2,925 |

**80 is the floor, not a bad score** — Wine Enthusiast only publishes reviews scoring 80+, so the scale is truncated and there is no such thing as a 50-point wine here. The distribution peaks at 87–88, mean 88.45, median 88.

`points >= 90` cuts the corpus roughly in half (~38%); `points >= 95` leaves ~1.8%. Combining a high rating floor with a low price ceiling gets selective fast — `Italy + red + ≤$20 + ≥90` matches just 163 wines. That's the usual cause of empty result sets.

**Price and rating correlate at only 0.42** — a real but loose relationship, which is what makes "cheap but highly rated" a genuinely useful search rather than a contradiction.

## Derived `color`

The dataset has no color column, so `ingest/classify_wine_color.py` has `claude-haiku-4-5` classify all 707 distinct varieties once:

| color | wines |
|---|---|
| red | 78,663 |
| white | 41,237 |
| sparkling | 4,715 |
| rose | 3,777 |
| fortified | 849 |
| dessert | 700 |
| orange | 23 |
| other | 6 |

The 6 `other` rows are legitimate: `variety` values of `'Other'`, `'Apple'` (fruit wine) and `'Black Monukka'` (a table grape). This is why `other` is excluded from the query-understanding enum — a spurious `color='other'` filter would narrow 130k wines to 6.

## Known limitations

**Duplicate reviews.** The CSV contains exact duplicates — e.g. ids 5574 and 68601 are byte-identical in both `title` and `description`. They surface as near-identical adjacent results. Deduplication belongs at ingestion (content hash over `title`+`description`), and is **not implemented**: it would change row count and `id` contiguity, which `build_vector_index.py` relies on for row-aligned embeddings, so it requires re-running the whole offline pipeline.

**Null prices exclude rows silently.** ~7% of wines have no price, so any price filter drops them without saying so.

**Long-tail varieties aren't filterable.** Only the top 200 by frequency are injected into the query-understanding prompt (to keep it small), so the remaining 507 fall through to free-text search rather than becoming hard filters.

**Whitespace in source data.** Four rows had trailing spaces in `variety` (`'Tintilia '`), which silently breaks exact-match filters. Now trimmed at load time — but a reminder that exact-match filters are only as good as the normalization upstream of them.

**No relevance judgments.** There are no human labels for "is this a good result for this query", so ranking quality is unmeasured. Everything in [retrieval-evaluation.md](retrieval-evaluation.md) measures *agreement between methods*, not correctness.
