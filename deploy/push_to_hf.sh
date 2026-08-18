#!/usr/bin/env bash
# Push this project to a Hugging Face Space (Docker SDK).
#
# Why a staging directory instead of a git branch: the two prebuilt index
# artifacts (~247MB) are gitignored on GitHub deliberately, but the Space needs
# them committed via Git LFS. Building a throwaway repo in /tmp keeps that LFS
# history entirely out of the GitHub repo and never touches your working tree.
#
# Usage:
#   ./deploy/push_to_hf.sh <hf-username> [space-name]
#
# Prerequisites:
#   - git lfs installed          (brew install git-lfs && git lfs install)
#   - a Hugging Face account and an access token with write permission
#   - the Space already created as SDK=Docker at
#     https://huggingface.co/new-space
#   - indexes built locally (ingest/ scripts) so index/ exists
#   - ANTHROPIC_API_KEY set as a Space secret in the Space's Settings

set -euo pipefail

HF_USER="${1:?usage: ./deploy/push_to_hf.sh <hf-username> [space-name]}"
SPACE_NAME="${2:-wine-tonight}"
SPACE_URL="https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The Space serves search requests directly from these; without them the app
# crashes at import time (retrieval.py np.load's the embeddings eagerly).
for artifact in index/wine.duckdb index/embeddings.npy; do
  if [[ ! -f "$artifact" ]]; then
    echo "ERROR: missing $artifact" >&2
    echo "Build the indexes first: see the 'Build the indexes' section in README.md" >&2
    exit 1
  fi
done

command -v git-lfs >/dev/null 2>&1 || {
  echo "ERROR: git lfs not installed. Run: brew install git-lfs && git lfs install" >&2
  exit 1
}

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
echo "Staging deployment in $STAGING"

# Application code and the page it serves.
mkdir -p "$STAGING/serve" "$STAGING/static" "$STAGING/index"
cp serve/*.py "$STAGING/serve/"
cp static/index.html "$STAGING/static/"
cp requirements.txt Dockerfile "$STAGING/"

# The Space's README.md carries the YAML frontmatter that configures it
# (sdk: docker, app_port: 7860). That frontmatter would render as stray text
# on GitHub, which is why it lives in deploy/ and is swapped in only here.
cp deploy/README_hf.md "$STAGING/README.md"

# faiss.index is deliberately excluded -- nothing in serve/ imports faiss.
# It exists only for the offline HNSW-vs-exact recall comparison.
cp index/wine.duckdb index/embeddings.npy "$STAGING/index/"

cd "$STAGING"
git init -q
git lfs install --local
git lfs track "index/*.duckdb" "index/*.npy" >/dev/null
git add .gitattributes
git add -A
git -c user.email=deploy@localhost -c user.name=deploy \
  commit -q -m "Deploy hybrid wine search to Hugging Face Spaces"

echo
echo "Pushing to $SPACE_URL"
echo "When prompted, use your HF username and an access token (not your password)."
echo "Create one at: https://huggingface.co/settings/tokens (needs write access)"
echo
git remote add origin "$SPACE_URL"
git push --force origin HEAD:main

echo
echo "Done. The Space will now build the Docker image (expect ~10-15 min the"
echo "first time -- it installs torch and bakes both models into the image)."
echo "Watch progress at: ${SPACE_URL}"
echo
echo "REMINDER: set ANTHROPIC_API_KEY as a secret under the Space's"
echo "Settings -> Variables and secrets, or query understanding will fail."
