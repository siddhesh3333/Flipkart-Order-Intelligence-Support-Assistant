"""
Part 3 - RAG Retrieval Evaluation

Evaluates policy retrieval using:

    Precision@3
    Recall@3

Evaluation is performed at the parent-document level.

Example:

Retrieved:
    POL-001-S003

is evaluated as:
    POL-001
"""

from pathlib import Path
import json
import sys


# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

ANSWER_KEY_FILE = (
    ROOT_DIR
    / "Part-3"
    / "evaluation"
    / "retrieval_answer_key.json"
)

# Allow importing Part-3/rag modules when running this file
sys.path.insert(
    0,
    str(ROOT_DIR / "Part-3")
)

from rag.retriever import PolicyRetriever


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 3


# ============================================================
# LOAD ANSWER KEY
# ============================================================

def load_answer_key():

    if not ANSWER_KEY_FILE.exists():

        raise FileNotFoundError(
            f"Answer key not found:\n{ANSWER_KEY_FILE}"
        )

    with ANSWER_KEY_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if not data:

        raise ValueError(
            "Answer key is empty."
        )

    return data


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def precision_at_k(
    retrieved_documents,
    relevant_documents,
    k=3
):

    retrieved = retrieved_documents[:k]

    if not retrieved:
        return 0.0

    hits = sum(
        doc_id in relevant_documents
        for doc_id in retrieved
    )

    return hits / len(retrieved)


def recall_at_k(
    retrieved_documents,
    relevant_documents,
    k=3
):

    retrieved = retrieved_documents[:k]

    if not relevant_documents:
        return 0.0

    hits = sum(
        doc_id in relevant_documents
        for doc_id in set(retrieved)
    )

    return hits / len(
        set(relevant_documents)
    )


# ============================================================
# EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("PART 3 - RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    answer_key = load_answer_key()

    print("\nAnswer-key queries:")
    print(len(answer_key))

    print("\nLoading retriever...")

    retriever = PolicyRetriever()

    print("\nRetriever loaded successfully.")

    results = []

    total_precision = 0.0
    total_recall = 0.0

    # --------------------------------------------------------
    # Evaluate each query
    # --------------------------------------------------------

    for i, item in enumerate(
        answer_key,
        start=1
    ):

        query = item["query"]

        relevant_documents = set(
            item["relevant_document_ids"]
        )

        retrieved = retriever.retrieve(
            query,
            top_k=TOP_K
        )

        retrieved_documents = [
            result["document_id"]
            for result in retrieved
        ]

        precision = precision_at_k(
            retrieved_documents,
            relevant_documents,
            TOP_K
        )

        recall = recall_at_k(
            retrieved_documents,
            relevant_documents,
            TOP_K
        )

        total_precision += precision
        total_recall += recall

        result = {
            "query": query,
            "relevant_documents": sorted(
                relevant_documents
            ),
            "retrieved_documents": retrieved_documents,
            "precision_at_3": precision,
            "recall_at_3": recall
        }

        results.append(result)

        print("\n" + "-" * 70)

        print(
            f"Query {i}/{len(answer_key)}"
        )

        print(
            f"Query: {query}"
        )

        print(
            f"Relevant documents: "
            f"{sorted(relevant_documents)}"
        )

        print(
            f"Retrieved documents: "
            f"{retrieved_documents}"
        )

        print(
            f"Precision@3: {precision:.4f}"
        )

        print(
            f"Recall@3   : {recall:.4f}"
        )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    mean_precision = (
        total_precision
        / len(answer_key)
    )

    mean_recall = (
        total_recall
        / len(answer_key)
    )

    print("\n" + "=" * 70)
    print("OVERALL RETRIEVAL METRICS")
    print("=" * 70)

    print(
        f"\nPrecision@3 : {mean_precision:.4f}"
    )

    print(
        f"Recall@3    : {mean_recall:.4f}"
    )

    print(
        f"\nQueries evaluated: {len(answer_key)}"
    )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    output_file = (
        ROOT_DIR
        / "Part-3"
        / "evaluation"
        / "retrieval_evaluation_results.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "top_k": TOP_K,
                "num_queries": len(answer_key),
                "mean_precision_at_3": mean_precision,
                "mean_recall_at_3": mean_recall,
                "queries": results
            },
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\nDetailed results saved:")
    print(output_file)

    print("\n" + "=" * 70)
    print("RAG RETRIEVAL EVALUATION: COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()