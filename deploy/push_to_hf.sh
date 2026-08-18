#!/usr/bin/env bash
# Deploy this project to a Hugging Face Space (Gradio SDK on ZeroGPU).
#
# Why Gradio + ZeroGPU rather than Docker + cpu-basic: free personal accounts
# can't create Docker or cpu-basic Spaces -- both are gated behind a paid plan.
# The one free option is up to 2 ZeroGPU Spaces, which are Gradio-only. None of
# this app's work is GPU work; app.py carries a no-op @spaces.GPU function
# purely because ZeroGPU refuses to start without one, so no quota is burned.
#
# Why a staging directory: the Space needs a different requirements.txt and
# README.md than the GitHub repo, plus the two prebuilt index artifacts that are
# gitignored here on purpose. Assembling in /tmp keeps the repo untouched.
#
# Usage:
#   ./deploy/push_to_hf.sh [space-name]
#
# Prerequisites:
#   - hf auth login   (prints a URL + one-time code)
#   - indexes built locally: see "Build the indexes" in README.md

set -euo pipefail

SPACE_NAME="${1:-wine-tonight}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

command -v hf >/dev/null 2>&1 || {
  echo "ERROR: hf CLI not found. Run: pip install -U huggingface_hub" >&2
  exit 1
}

HF_USER="$(hf auth whoami 2>/dev/null | head -1 | tr -d '[:space:]')" || true
if [[ -z "${HF_USER}" || "${HF_USER}" == "Notloggedin" ]]; then
  echo "ERROR: not logged in to Hugging Face. Run: hf auth login" >&2
  exit 1
fi
SPACE_ID="${HF_USER}/${SPACE_NAME}"

# Without these the app dies at import: retrieval.py np.load's the embeddings
# eagerly, and every query reads the DuckDB file.
for artifact in index/wine.duckdb index/embeddings.npy; do
  [[ -f "$artifact" ]] || {
    echo "ERROR: missing $artifact -- build the indexes first (see README.md)" >&2
    exit 1
  }
done

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
echo "Staging Space contents in $STAGING"

mkdir -p "$STAGING/serve" "$STAGING/index"

# Gradio entry point + the unchanged pipeline modules it imports.
cp app.py "$STAGING/"
cp serve/fusion.py serve/query_understanding.py serve/rerank.py serve/retrieval.py "$STAGING/serve/"

# Space-specific variants (see the header comment for why these differ).
cp deploy/requirements_hf.txt "$STAGING/requirements.txt"
cp deploy/README_hf.md "$STAGING/README.md"

# faiss.index is intentionally excluded -- unused at serving time.
cp index/wine.duckdb index/embeddings.npy "$STAGING/index/"

echo "Creating Space ${SPACE_ID} (idempotent)"
hf repos create "$SPACE_ID" \
  --type space \
  --space-sdk gradio \
  --flavor zero-a10g \
  --public \
  --exist-ok

echo "Uploading (~247MB of index artifacts, so this takes a few minutes)"
# --repo-type space is required; hf upload otherwise creates a *model* repo.
hf upload "$SPACE_ID" "$STAGING" . \
  --repo-type space \
  --exclude "**/__pycache__/**" \
  --commit-message "Deploy hybrid wine search"

cat <<EOF

Pushed to https://huggingface.co/spaces/${SPACE_ID}

NEXT STEP -- the app will fail on every query until you do this:
  hf spaces secrets set ${SPACE_ID} ANTHROPIC_API_KEY=<your-key>

Then watch it come up:
  hf spaces logs ${SPACE_ID} --build --follow    # build errors
  hf spaces logs ${SPACE_ID} --follow            # runtime errors
  hf spaces info ${SPACE_ID} --expand runtime    # stage + hardware
EOF
