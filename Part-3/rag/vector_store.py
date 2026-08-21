"""
Part 3 - FAISS Vector Store

Builds a FAISS inner-product index from normalized
sentence-transformer embeddings.

Because embeddings are normalized, inner product
is equivalent to cosine similarity.
"""

from pathlib import Path
import json

import faiss
import numpy as np


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT_DIR
    / "Part-3"
    / "data"
    / "processed"
)

EMBEDDINGS_FILE = (
    DATA_DIR
    / "policy_embeddings.npz"
)

METADATA_FILE = (
    DATA_DIR
    / "embedding_metadata.jsonl"
)

INDEX_FILE = (
    DATA_DIR
    / "policy_faiss.index"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_embeddings():

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings not found:\n{EMBEDDINGS_FILE}"
        )

    embeddings = np.load(
        EMBEDDINGS_FILE
    )["embeddings"]

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    return embeddings


def load_metadata():

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata not found:\n{METADATA_FILE}"
        )

    metadata = []

    with METADATA_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if line:
                metadata.append(
                    json.loads(line)
                )

    return metadata


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(
    embeddings: np.ndarray
):

    dimension = embeddings.shape[1]

    # Inner Product index.
    #
    # Embeddings were normalized during the
    # embedding stage, therefore:
    #
    # inner product == cosine similarity
    #
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PART 3 - FAISS VECTOR STORE")
    print("=" * 70)

    print("\nEmbeddings:")
    print(EMBEDDINGS_FILE)

    embeddings = load_embeddings()

    print("\nEmbedding shape:")
    print(embeddings.shape)

    metadata = load_metadata()

    print("\nMetadata rows:")
    print(len(metadata))

    if len(metadata) != len(embeddings):
        raise ValueError(
            "Embedding count does not match metadata count."
        )

    # --------------------------------------------------------
    # Validate normalized embeddings
    # --------------------------------------------------------

    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-4
    ):
        raise ValueError(
            "Embeddings are not normalized."
        )

    print("\nEmbedding normalization:")
    print("PASSED")

    # --------------------------------------------------------
    # Build FAISS
    # --------------------------------------------------------

    print("\nBuilding FAISS index...")

    index = build_index(
        embeddings
    )

    print("FAISS index built.")

    print("\nIndex type:")
    print(type(index).__name__)

    print("\nVector dimension:")
    print(index.d)

    print("\nNumber of vectors:")
    print(index.ntotal)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    print("\nSaved index:")
    print(INDEX_FILE)

    # --------------------------------------------------------
    # Self-retrieval sanity check
    # --------------------------------------------------------

    print("\nRunning self-retrieval sanity check...")

    query_vector = embeddings[0:1]

    scores, indices = index.search(
        query_vector,
        3
    )

    print("\nTop 3 results for first embedding:")

    for rank, (
        score,
        idx
    ) in enumerate(
        zip(
            scores[0],
            indices[0]
        ),
        start=1
    ):

        item = metadata[int(idx)]

        print(
            f"{rank}. "
            f"{item['chunk_id']} | "
            f"{item['document_id']} | "
            f"score={score:.4f}"
        )

    # The first vector should retrieve itself.
    if int(indices[0][0]) != 0:
        raise AssertionError(
            "FAISS self-retrieval sanity check failed."
        )

    print("\nSelf-retrieval:")
    print("PASSED")

    print("\n" + "=" * 70)
    print("FAISS VECTOR STORE: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()