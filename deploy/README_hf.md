---
title: What Wine Do You Want To Drink Tonight
emoji: 🍷
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.24.0
app_file: app.py
python_version: "3.12"
short_description: Hybrid wine search - keyword + vector, then reranked
pinned: false
---

# 🍷 What wine do you want to drink tonight?

Ask in plain language and get a wine. A hybrid search stack over 129,971
Wine Enthusiast reviews.

Per query:

1. **Query understanding** — `claude-haiku-4-5` maps natural language onto hard
   filters (country, grape, style, price range, rating range) plus a residual
   semantic query, grounded against the real column values so "Italian"
   resolves to `country = 'Italy'`.
2. **Hybrid retrieval** — both paths filter first, then search: BM25 keyword
   scoring within the filtered set (DuckDB FTS), and exact cosine similarity
   over the filtered subset of `bge-small-en-v1.5` embeddings.
3. **Fusion** — Reciprocal Rank Fusion merges the two ranked lists by rank
   position, since BM25 scores and cosine similarities aren't comparable scales.
4. **Reranking** — a `ms-marco-MiniLM-L-6-v2` cross-encoder rescores the fused
   top 50, reading the query and each tasting note together rather than as
   independent vectors.

This Space runs on ZeroGPU hardware but does no GPU work — both models are
small enough to run on CPU. See the note at the top of `app.py`.

Source: [github.com/madpr/wine-tonight](https://github.com/madpr/wine-tonight)
