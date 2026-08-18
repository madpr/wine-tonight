# Hugging Face Spaces (Docker SDK) deployment.
#
# Two non-obvious things this handles:
#
# 1. torch on Linux. The default PyPI torch wheel for Linux bundles CUDA and is
#    ~2.5GB. On macOS arm64 the default wheel is already CPU-only, so this only
#    bites in Docker. The explicit CPU index URL keeps the image manageable.
#
# 2. Model weights are baked into the image at build time rather than downloaded
#    on first request. bge-small (~137MB) + the cross-encoder (~90MB) would
#    otherwise download on every cold start, into a filesystem that is
#    read-only outside the user's home.

FROM python:3.12-slim

# Spaces runs containers as UID 1000. Creating that user explicitly (rather
# than running as root) keeps HF_HOME and the pip user-install dir writable.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME="/home/user/.cache/huggingface" \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Pre-fetch both models so cold starts don't pay a ~230MB download.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user . ./

# Spaces expects the app on port 7860. --app-dir is required because
# serve/api.py imports its sibling modules flatly.
EXPOSE 7860
CMD ["uvicorn", "api:app", "--app-dir", "serve", "--host", "0.0.0.0", "--port", "7860"]
