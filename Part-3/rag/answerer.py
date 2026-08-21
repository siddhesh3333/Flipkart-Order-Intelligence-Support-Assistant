"""
Part 3 - Policy Grounded Answerer

Retrieves policy evidence and generates a deterministic answer
only from sufficiently relevant policy chunks.

The answerer distinguishes between:
    - grounded evidence
    - sufficiently supported answers
    - unsupported questions

No external LLM/API is required.
"""

from pathlib import Path
import sys
import re


# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

RAG_DIR = ROOT_DIR / "Part-3" / "rag"

if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from retriever import PolicyRetriever


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 5

# Minimum semantic similarity required before evidence
# can normally participate in the answer.
MIN_SCORE = 0.50

# Stronger threshold for deciding whether the question
# has meaningful policy support.
SUPPORT_SCORE = 0.55


# ============================================================
# ANSWERER
# ============================================================

class PolicyAnswerer:

    def __init__(
        self,
        top_k=DEFAULT_TOP_K,
        min_score=MIN_SCORE,
        support_score=SUPPORT_SCORE
    ):

        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        if not 0 <= min_score <= 1:
            raise ValueError(
                "min_score must be between 0 and 1."
            )

        if not 0 <= support_score <= 1:
            raise ValueError(
                "support_score must be between 0 and 1."
            )

        self.top_k = top_k
        self.min_score = min_score
        self.support_score = support_score

        self.retriever = PolicyRetriever()

    # ========================================================
    # PUBLIC ANSWER METHOD
    # ========================================================

    def answer(self, query):

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Query must be a non-empty string."
            )

        # ----------------------------------------------------
        # Retrieve evidence
        # ----------------------------------------------------

        retrieved = self.retriever.retrieve(
            query,
            top_k=self.top_k
        )

        # ----------------------------------------------------
        # Filter weak evidence
        # ----------------------------------------------------

        relevant = [
            result
            for result in retrieved
            if result["score"] >= self.min_score
        ]

        # ----------------------------------------------------
        # Determine whether evidence is strong enough
        # ----------------------------------------------------

        supported = self._is_supported(
            query,
            retrieved,
            relevant
        )

        # ----------------------------------------------------
        # Unsupported question
        # ----------------------------------------------------

        if not supported:

            return {
                "query": query,

                "answer": (
                    "I could not find a sufficiently relevant policy "
                    "for this question in the available policy "
                    "documents. Please use the support option "
                    "associated with the relevant order."
                ),

                "sources": self._format_sources(
                    relevant
                ),

                "grounded": False,

                "supported": False,

                "retrieved_count": len(retrieved),

                "relevant_count": len(relevant)
            }

        # ----------------------------------------------------
        # Build grounded answer
        # ----------------------------------------------------

        answer = self._build_answer(
            relevant
        )

        return {
            "query": query,

            "answer": answer,

            "sources": self._format_sources(
                relevant
            ),

            "grounded": True,

            "supported": True,

            "retrieved_count": len(retrieved),

            "relevant_count": len(relevant)
        }

    # ========================================================
    # SUPPORT DETECTION
    # ========================================================

    def _is_supported(
        self,
        query,
        retrieved,
        relevant
    ):

        if not retrieved:
            return False

        if not relevant:
            return False

        # Strongest retrieved result.
        best_score = retrieved[0]["score"]

        if best_score < self.support_score:
            return False

        # ----------------------------------------------------
        # Lightweight semantic keyword guard
        #
        # This is deliberately conservative. It prevents
        # obviously unrelated policy text from being treated
        # as an answer simply because the embedding similarity
        # is moderately high.
        # ----------------------------------------------------

        query_terms = self._important_terms(
            query
        )

        if not query_terms:
            return True

        combined_text = " ".join(
            result["text"].lower()
            for result in relevant
        )

        matched_terms = [
            term
            for term in query_terms
            if term in combined_text
        ]

        # At least one meaningful query concept should
        # appear in the retrieved policy evidence.
        return len(matched_terms) >= 1

    # ========================================================
    # IMPORTANT QUERY TERMS
    # ========================================================

    def _important_terms(self, query):

        stop_words = {
            "what",
            "when",
            "where",
            "which",
            "who",
            "how",
            "can",
            "could",
            "should",
            "would",
            "may",
            "might",
            "do",
            "does",
            "did",
            "is",
            "are",
            "was",
            "were",
            "the",
            "a",
            "an",
            "to",
            "for",
            "of",
            "from",
            "on",
            "in",
            "if",
            "my",
            "me",
            "i",
            "it",
            "product",
            "order"
        }

        words = re.findall(
            r"[a-zA-Z]+",
            query.lower()
        )

        terms = [
            word
            for word in words
            if len(word) >= 4
            and word not in stop_words
        ]

        return terms

    # ========================================================
    # ANSWER CONSTRUCTION
    # ========================================================

    def _build_answer(
        self,
        relevant
    ):

        answer_lines = []

        for result in relevant:

            text = result["text"]

            if text not in answer_lines:

                answer_lines.append(text)

        return " ".join(
            answer_lines
        )

    # ========================================================
    # SOURCE FORMAT
    # ========================================================

    def _format_sources(
        self,
        results
    ):

        sources = []

        for result in results:

            sources.append(
                {
                    "chunk_id": result["chunk_id"],

                    "document_id": result["document_id"],

                    "source_file": result["source_file"],

                    "score": round(
                        result["score"],
                        4
                    )
                }
            )

        return sources


# ============================================================
# CLI TEST
# ============================================================

def main():

    print("=" * 70)
    print("PART 3 - POLICY GROUNDED ANSWERER TEST")
    print("=" * 70)

    answerer = PolicyAnswerer(
        top_k=5
    )

    test_queries = [

        "Can I return a product after the return window expires?",

        "What conditions must a product satisfy for a return?",

        "Which products may not be eligible for return?",

        "What should I do if I receive a defective product?",

        "When should I use the warranty process?",

        "What should I do if I receive the wrong product?",

    ]

    for query in test_queries:

        print("\n" + "-" * 70)

        print("QUESTION:")
        print(query)

        result = answerer.answer(
            query
        )

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES:")

        for source in result["sources"]:

            print(
                f" - {source['document_id']} "
                f"| {source['chunk_id']} "
                f"| score={source['score']:.4f}"
            )

        print(
            "\nGrounded:",
            result["grounded"]
        )

        print(
            "Supported:",
            result["supported"]
        )

        print(
            "Retrieved:",
            result["retrieved_count"]
        )

        print(
            "Relevant:",
            result["relevant_count"]
        )

    print("\n" + "=" * 70)

    print(
        "POLICY ANSWERER TEST: COMPLETED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()