"""
Part 3 - Local Sentence Transformer Embeddings

Reads:
    Part-3/data/processed/policy_chunks.jsonl

Generates local sentence-transformer embeddings and saves:
    Part-3/data/processed/policy_embeddings.npz
    Part-3/data/processed/embedding_metadata.jsonl
"""

from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT_DIR
    / "Part-3"
    / "data"
    / "processed"
    / "policy_chunks.jsonl"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "Part-3"
    / "data"
    / "processed"
)

EMBEDDINGS_FILE = (
    OUTPUT_DIR
    / "policy_embeddings.npz"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "embedding_metadata.jsonl"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Chunk file not found:\n{INPUT_FILE}"
        )

    chunks = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if line:
                chunks.append(
                    json.loads(line)
                )

    if not chunks:
        raise ValueError(
            "No chunks found."
        )

    return chunks


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PART 3 - LOCAL SENTENCE TRANSFORMER EMBEDDINGS")
    print("=" * 70)

    print("\nInput:")
    print(INPUT_FILE)

    chunks = load_chunks()

    print("\nChunks loaded:")
    print(len(chunks))

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("\nEmbedding model:")
    print(MODEL_NAME)

    print("\nLoading local SentenceTransformer model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    print("Model loaded successfully.")

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype(
        np.float32
    )

    print("\nEmbedding shape:")
    print(embeddings.shape)

    print("\nEmbedding dimension:")
    print(embeddings.shape[1])

    # --------------------------------------------------------
    # Save vectors
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        EMBEDDINGS_FILE,
        embeddings=embeddings
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    with METADATA_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        for chunk in chunks:

            metadata = {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "source_file": chunk["source_file"],
                "text": chunk["text"],
            }

            file.write(
                json.dumps(
                    metadata,
                    ensure_ascii=False
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert embeddings.shape[0] == len(chunks)

    assert embeddings.dtype == np.float32

    assert np.all(
        np.isfinite(embeddings)
    )

    print("\nSaved embeddings:")
    print(EMBEDDINGS_FILE)

    print("\nSaved metadata:")
    print(METADATA_FILE)

    print("\nFirst embedding preview:")
    print(
        embeddings[0][:10]
    )

    print("\n" + "=" * 70)
    print("EMBEDDING GENERATION: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()