"""Phase 2: embed wine descriptions and build a FAISS HNSW vector index.

Passage side (description) is embedded with no instruction prefix; only
query-side text needs the BGE "Represent this sentence for searching
relevant passages: " prefix (see serve/retrieval.py at query time).

Also builds a brute-force flat index in-memory (not persisted) purely to
measure HNSW's recall@10 against exact search, as a learning check on the
ANN approximation tradeoff.
"""

import pathlib
import random
import time

import duckdb
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "index" / "wine.duckdb"
EMBEDDINGS_PATH = ROOT / "index" / "embeddings.npy"
FAISS_INDEX_PATH = ROOT / "index" / "faiss.index"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_descriptions() -> tuple[np.ndarray, list[str]]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute("SELECT id, description FROM wines ORDER BY id").fetchall()
    con.close()
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    # id must equal row position 0..n-1 so FAISS's internal index doubles as the wine id.
    assert np.array_equal(ids, np.arange(len(ids))), "ids are not contiguous 0..n-1"
    descriptions = [r[1] for r in rows]
    return ids, descriptions


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on device={device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    dim = model.get_sentence_embedding_dimension()

    ids, descriptions = load_descriptions()
    print(f"Embedding {len(descriptions)} descriptions (dim={dim})...")

    t0 = time.time()
    embeddings = model.encode(
        descriptions,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"Embedded in {time.time() - t0:.1f}s")

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_PATH} ({embeddings.shape})")

    hnsw_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw_index.add(embeddings)
    faiss.write_index(hnsw_index, str(FAISS_INDEX_PATH))
    print(f"Saved HNSW index to {FAISS_INDEX_PATH}")

    # --- Learning check: HNSW vs exact flat search, recall@10 ---
    flat_index = faiss.IndexFlatIP(dim)
    flat_index.add(embeddings)

    random.seed(0)
    sample_ids = random.sample(range(len(embeddings)), 50)
    query_vecs = embeddings[sample_ids]

    _, flat_neighbors = flat_index.search(query_vecs, 10)
    _, hnsw_neighbors = hnsw_index.search(query_vecs, 10)

    recalls = [
        len(set(flat_neighbors[i]) & set(hnsw_neighbors[i])) / 10
        for i in range(len(sample_ids))
    ]
    print(f"\nHNSW recall@10 vs exact flat search (50 sampled queries): {np.mean(recalls):.3f}")

    # --- Sanity check: a real natural-language query ---
    test_query = QUERY_PREFIX + "cherry and leather notes"
    query_emb = model.encode([test_query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    _, neighbors = hnsw_index.search(query_emb, 5)
    print("\nTop 5 neighbors for 'cherry and leather notes':")
    for idx in neighbors[0]:
        print(f"  [{idx}] {descriptions[idx][:100]}")


if __name__ == "__main__":
    main()
