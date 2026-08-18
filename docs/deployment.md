# Deployment

Live at [huggingface.co/spaces/pratheeksha11/wine-tonight](https://huggingface.co/spaces/pratheeksha11/wine-tonight) — Gradio SDK on `zero-a10g` hardware.

```
./deploy/push_to_hf.sh                                          # create + upload
hf spaces secrets add <user>/wine-tonight --secrets-file .env    # required; see below
```

## Why not Vercel

Three hard blockers, none of them configuration:

| Blocker | Detail |
|---|---|
| **Dependency size** | Vercel Python functions cap at 250 MB unzipped. `torch` alone is 529 MB. `sentence-transformers` requires it. Unfixable. |
| **No data** | `index/` is gitignored (correctly — it's regenerable), so a deploy contains zero searchable content. `retrieval.py` does `np.load()` at import and crashes immediately. |
| **Cold-start downloads** | Both models (~230 MB) would download per cold start onto a read-only filesystem, past the function timeout. |

Serverless is built for stateless lightweight handlers; this app holds ~700 MB of models and indexes in memory. Different shapes.

## Why Gradio + ZeroGPU rather than Docker

Hugging Face gates hardware by plan:

| SDK / hardware | Free account |
|---|---|
| Static | ✅ free for everyone (but no Python runtime) |
| ZeroGPU (Gradio only) | ⚠️ up to 2 Spaces, **requires account ≥30 days old** |
| `cpu-basic` | ❌ requires PRO |
| Docker | ❌ requires PRO |

So on a free account, a Gradio Space on ZeroGPU is the only way to run Python — and a brand-new account can't do even that (HTTP 402: *"wait 30 days or request a community grant"*).

**This Space runs on ZeroGPU but does no GPU work.** A 33M-param embedding model, a 22M-param cross-encoder and a numpy dot product all run fine on CPU. ZeroGPU refuses to start without at least one decorated function, so `app.py` carries a `_noop()` under `@spaces.GPU(duration=1)` that is never called. No GPU is ever requested, so no quota is consumed. This is the pattern HF documents for CPU-bound Spaces.

A `Dockerfile` remains in the repo for container hosts that aren't HF (Fly, Railway). It is **untested** — the local Docker daemon never started.

## What gets deployed

`deploy/push_to_hf.sh` assembles a staging directory in `/tmp` rather than using a git branch, because the Space needs a different `requirements.txt` and `README.md` than GitHub, plus two artifacts that are gitignored here on purpose. Staging keeps the LFS history out of the GitHub repo and leaves the working tree untouched.

| Included | Excluded |
|---|---|
| `app.py`, `serve/*.py` | `serve/api.py`, `static/` (Gradio replaces them) |
| `index/wine.duckdb` (59 MB) | `index/faiss.index` (unused at serving time) |
| `index/embeddings.npy` (200 MB) | `data/` (raw CSV not needed) |
| `deploy/requirements_hf.txt` → `requirements.txt` | `faiss-cpu`, `fastapi`, `uvicorn`, `pandas` |
| `deploy/README_hf.md` → `README.md` | `gradio`, `spaces`, `huggingface_hub` (platform-managed) |

`torch` is left unpinned so the ZeroGPU runtime supplies its own version. Pinning `gradio`/`spaces`/`huggingface_hub` breaks the runtime.

## Secrets

```
hf spaces secrets add <user>/wine-tonight --secrets-file .env
```

Use `--secrets-file`, not `-s KEY=VALUE` — the latter puts the key in shell history. The bare `-s NAME` form does **not** prompt for a value; that special case only works for `HF_TOKEN`.

Without `ANTHROPIC_API_KEY` set, the Space builds and starts fine, then fails **every query** with `TypeError: Could not resolve authentication method`. The Anthropic SDK resolves credentials at request time, not at client construction, so a missing key is a per-query failure rather than a startup crash.

## Bugs deployment surfaced

Worth recording because none appeared locally:

1. **DuckDB FTS extension missing.** The FTS *index data* travels inside `wine.duckdb`, but the extension *binary* is a platform-specific shared library in `~/.duckdb/extensions/`. The local copy was macOS-arm64; the Linux container had none. Fixed by calling `INSTALL fts` before `LOAD fts` — portable to any host reading the prebuilt index.
2. **Opaque errors.** A failing query returned only *"the upstream Gradio app has raised an exception"*, with the cause in server logs. Fixed with `show_error=True` and a `try`/`except` around query understanding that names the missing-secret case explicitly.

## Verifying a deploy

`RUNNING` does not mean working. In order:

```
hf spaces info <id> --expand runtime      # stage + hardware
hf spaces logs <id> --build --follow      # build errors
hf spaces logs <id> --follow              # runtime errors
```

Then call it for real — the Space is also an MCP server, and its Gradio endpoint is directly callable:

```python
from gradio_client import Client
c = Client("<user>/wine-tonight", httpx_kwargs={"timeout": 240})
interp, results = c.predict(query="cheap Italian red under $20, 90+ points",
                            api_name="/search_wines")
```
