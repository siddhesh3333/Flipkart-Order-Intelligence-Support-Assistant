"""
Part 3 - Policy Retriever

Loads:
    - local SentenceTransformer model
    - FAISS policy index
    - policy metadata

Provides semantic top-k retrieval for policy questions.
"""

from pathlib import Path
import json

import faiss
from sentence_transformers import SentenceTransformer


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

INDEX_FILE = (
    DATA_DIR
    / "policy_faiss.index"
)

METADATA_FILE = (
    DATA_DIR
    / "embedding_metadata.jsonl"
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# RETRIEVER
# ============================================================

class PolicyRetriever:

    def __init__(
        self,
        model_name=MODEL_NAME
    ):

        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found:\n{INDEX_FILE}"
            )

        if not METADATA_FILE.exists():
            raise FileNotFoundError(
                f"Metadata not found:\n{METADATA_FILE}"
            )

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        print(
            f"FAISS vectors: {self.index.ntotal}"
        )

        print(
            f"Vector dimension: {self.index.d}"
        )

        print("\nLoading embedding model...")

        self.model = SentenceTransformer(
            model_name,
            device="cpu"
        )

        print("Embedding model loaded.")

        self.metadata = []

        with METADATA_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if line:
                    self.metadata.append(
                        json.loads(line)
                    )

        if len(self.metadata) != self.index.ntotal:

            raise ValueError(
                "Metadata count does not match FAISS vector count."
            )


    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        query,
        top_k=3
    ):

        if not isinstance(
            query,
            str
        ) or not query.strip():

            raise ValueError(
                "Query must be a non-empty string."
            )

        if top_k < 1:

            raise ValueError(
                "top_k must be >= 1."
            )

        top_k = min(
            top_k,
            self.index.ntotal
        )

        # ----------------------------------------------------
        # Embed query
        # ----------------------------------------------------

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # ----------------------------------------------------
        # FAISS search
        # ----------------------------------------------------

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for rank, (
            score,
            index_id
        ) in enumerate(
            zip(
                scores[0],
                indices[0]
            ),
            start=1
        ):

            index_id = int(index_id)

            if index_id < 0:
                continue

            metadata = self.metadata[
                index_id
            ]

            result = {
                "rank": rank,
                "score": float(score),
                "chunk_id": metadata["chunk_id"],
                "document_id": metadata["document_id"],
                "source_file": metadata["source_file"],
                "text": metadata["text"],
            }

            results.append(
                result
            )

        return results


# ============================================================
# CLI TEST
# ============================================================

def main():

    print("=" * 70)
    print("PART 3 - POLICY RETRIEVER TEST")
    print("=" * 70)

    retriever = PolicyRetriever()

    test_queries = [

        "Can I return a product after the return window expires?",

        "What conditions must a product satisfy for a return?",

        "Which products may not be eligible for return?",

    ]

    for query in test_queries:

        print("\n" + "-" * 70)
        print("QUERY:")
        print(query)

        results = retriever.retrieve(
            query,
            top_k=3
        )

        print("\nTOP 3 RESULTS:")

        for result in results:

            print(
                f"\n"
                f"Rank       : {result['rank']}\n"
                f"Score      : {result['score']:.4f}\n"
                f"Chunk ID    : {result['chunk_id']}\n"
                f"Document ID : {result['document_id']}\n"
                f"Source      : {result['source_file']}\n"
                f"Text       : {result['text']}"
            )

    print("\n" + "=" * 70)
    print("POLICY RETRIEVER: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()