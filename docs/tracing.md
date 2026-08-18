# Tracing

Set `WINE_TRACE=1` to log every pipeline stage — inputs on entry, return value and duration on exit, correlated by a per-request trace id.

```
WINE_TRACE=1 python3 serve/hand_trace.py "cheap Italian red under \$20, 90+ points"
WINE_TRACE=1 uvicorn api:app --app-dir serve --port 8000
```

```
[83e45354] → search('cheap Italian red under $20, 90+ points')
[83e45354]   → understand_query('cheap Italian red under $20, 90+ points')
[83e45354]   ← understand_query = {country='Italy', color='red', price_max=20, points_min=90, ...} [1920ms]
[83e45354]   → hybrid_candidates({...}, 'cheap Italian red', top_n=100)
[83e45354]     → keyword_candidates({...}, 'cheap Italian red', 100)
[83e45354]     ← keyword_candidates = list[78403, 60268, ..., +54 more] (n=60) [43ms]
[83e45354]     → vector_candidates({...}, 'cheap Italian red', 100)
[83e45354]     ← vector_candidates = list[120895, 22797, ..., +94 more] (n=100) [319ms]
[83e45354]   ← reciprocal_rank_fusion = list[tuple[78403, 0.030], ..., +106 more] (n=112) [0ms]
[83e45354]   ← rerank = list[tuple[60268, -5.84], ..., +44 more] (n=50) [112ms]
[83e45354] ← search = list[...] (n=3) [2395ms]
```

Off unless `WINE_TRACE` is set — the decorator short-circuits on `isEnabledFor`, so the disabled path costs one boolean check.

## Usage

```python
from tracing import traced

@traced                    # a pipeline stage
def keyword_candidates(filters, query_text, top_n=100): ...

@traced(root=True)         # a request entry point
def search(request): ...
```

## Two design problems it solves

**Values here are enormous.** The pipeline passes a `(129971, 384)` embeddings matrix, 100-element candidate id lists, and paragraph-length descriptions. Logging them verbatim would bury the signal, so `summarize()` renders structure instead of contents:

| Value | Logged as |
|---|---|
| numpy array | `ndarray(shape=(129971, 384), dtype=float32)` |
| long list | `list[78403, 60268, ..., +54 more] (n=60)` |
| filter dict | `{country='Italy', color='red'} (+3 unset)` — only keys actually set |
| long string | first 120 chars + `... (N chars)` |

**Stages need correlating.** One search spans five functions, and concurrent requests interleave in the log. The trace id lives in a `contextvars.ContextVar`, so no function signature had to change to accept it, and it stays correct across asyncio tasks and threads — which is why the same decorator works unmodified under both FastAPI and Gradio.

## Why `root=True` exists

`@traced` logs the entry line *before* the function body runs. Calling `new_trace()` inside the body is therefore too late — the entry line would carry the *previous* request's id. Marking an entry point `root=True` allocates the id first.

This was a real bug caught while building it, which is why the distinction exists rather than a single decorator.

## Error path

Failures log at each level with `✗`, including duration, then propagate unchanged:

```
[019b1203] → outer('some query')
[019b1203]   → inner('some query')
[019b1203]   ✗ inner raised ValueError: simulated failure [0ms]
[019b1203] ✗ outer raised ValueError: simulated failure [0ms]
```

## What it revealed immediately

The LLM query-understanding call is **~80% of request latency** (1920ms of 2395ms). Everything one might assume is expensive — vector search, BM25, cross-encoder reranking — totals under 500ms. That reframes where optimisation effort belongs; see the prompt-caching note in [architecture.md](architecture.md).
