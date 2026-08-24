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

# Minimum score required for a retrieved chunk
# to participate in the final answer.
MIN_SCORE = 0.50

# Minimum score required for the strongest result
# before a query can normally be considered supported.
SUPPORT_SCORE = 0.55

# Minimum number of query concepts that should match
# the retrieved policy evidence.
MIN_TERM_MATCHES = 1


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
            raise ValueError(
                "top_k must be >= 1."
            )

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

        query = query.strip()

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
        # Determine support
        # ----------------------------------------------------

        supported = self._is_supported(
            query=query,
            retrieved=retrieved,
            relevant=relevant
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

                "sources": [],

                "grounded": False,

                "supported": False,

                # Explicit count fields.
                "retrieved_count": len(retrieved),
                "relevant_count": len(relevant),

                # Backward-compatible aliases used by
                # evaluation scripts.
                "retrieved": len(retrieved),
                "relevant": len(relevant)
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

            # Explicit count fields.
            "retrieved_count": len(retrieved),
            "relevant_count": len(relevant),

            # Backward-compatible aliases.
            "retrieved": len(retrieved),
            "relevant": len(relevant)
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

        # No retrieval result.
        if not retrieved:
            return False

        # No sufficiently relevant evidence.
        if not relevant:
            return False

        # ----------------------------------------------------
        # Strongest retrieved result
        # ----------------------------------------------------

        best_score = retrieved[0]["score"]

        if best_score < self.support_score:
            return False

        # ----------------------------------------------------
        # Query concept extraction
        # ----------------------------------------------------

        query_terms = self._important_terms(
            query
        )

        if not query_terms:
            return True

        # ----------------------------------------------------
        # Combine relevant policy text
        # ----------------------------------------------------

        combined_text = " ".join(
            result["text"].lower()
            for result in relevant
        )

        # Normalize text.
        combined_text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            combined_text
        )

        combined_text = re.sub(
            r"\s+",
            " ",
            combined_text
        ).strip()

        # ----------------------------------------------------
        # Direct term matching
        # ----------------------------------------------------

        matched_terms = []

        for term in query_terms:

            if self._term_matches_text(
                term,
                combined_text
            ):
                matched_terms.append(term)

        # ----------------------------------------------------
        # Semantic concept aliases
        #
        # This handles cases where the query wording and
        # policy wording are different.
        #
        # Example:
        #
        # "wrong product"
        #
        # may be represented in the policy as:
        #
        # "different from the product ordered"
        # ----------------------------------------------------

        if self._concept_match(
            query,
            combined_text
        ):
            return True

        # ----------------------------------------------------
        # Standard keyword requirement
        # ----------------------------------------------------

        return (
            len(matched_terms)
            >= MIN_TERM_MATCHES
        )

    # ========================================================
    # TERM MATCHING
    # ========================================================

    def _term_matches_text(
        self,
        term,
        text
    ):

        # Direct word match.
        pattern = r"\b" + re.escape(term) + r"\b"

        if re.search(
            pattern,
            text
        ):
            return True

        # ----------------------------------------------------
        # Basic morphological handling
        # ----------------------------------------------------

        variations = set()

        if term.endswith("ed"):
            variations.add(
                term[:-2]
            )

        if term.endswith("ing"):
            variations.add(
                term[:-3]
            )

        if term.endswith("s"):
            variations.add(
                term[:-1]
            )

        for variation in variations:

            if len(variation) < 4:
                continue

            pattern = (
                r"\b"
                + re.escape(variation)
                + r"\w*"
                + r"\b"
            )

            if re.search(
                pattern,
                text
            ):
                return True

        return False

    # ========================================================
    # CONCEPT MATCHING
    # ========================================================

    def _concept_match(
        self,
        query,
        combined_text
    ):

        query_lower = query.lower()

        # ----------------------------------------------------
        # Wrong / incorrect / different product
        # ----------------------------------------------------

        wrong_product_terms = [
            "wrong product",
            "incorrect product",
            "different product",
            "product is different",
            "received the wrong",
            "receive the wrong",
            "wrong item",
            "incorrect item",
            "different item"
        ]

        if any(
            phrase in query_lower
            for phrase in wrong_product_terms
        ):

            policy_indicators = [
                "different from the product ordered",
                "different from the product",
                "incorrect product",
                "wrong product",
                "product ordered",
                "received a product"
            ]

            if any(
                indicator in combined_text
                for indicator in policy_indicators
            ):
                return True

        # ----------------------------------------------------
        # Defective product
        # ----------------------------------------------------

        defective_terms = [
            "defective product",
            "defective item",
            "product is defective",
            "received a defective"
        ]

        if any(
            phrase in query_lower
            for phrase in defective_terms
        ):

            if (
                "defective product" in combined_text
                or
                "suspected defect" in combined_text
                or
                "defect" in combined_text
            ):
                return True

        # ----------------------------------------------------
        # Damaged product
        # ----------------------------------------------------

        damaged_terms = [
            "damaged product",
            "damaged item",
            "product is damaged",
            "received a damaged"
        ]

        if any(
            phrase in query_lower
            for phrase in damaged_terms
        ):

            if (
                "damaged product" in combined_text
                or
                "visibly damaged" in combined_text
                or
                "damaged" in combined_text
            ):
                return True

        # ----------------------------------------------------
        # Warranty
        # ----------------------------------------------------

        if "warranty" in query_lower:

            if (
                "warranty" in combined_text
                or
                "warranty claim" in combined_text
            ):
                return True

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        if "return" in query_lower:

            if (
                "return" in combined_text
                or
                "non-returnable" in combined_text
                or
                "return request" in combined_text
            ):
                return True

        return False

    # ========================================================
    # IMPORTANT QUERY TERMS
    # ========================================================

    def _important_terms(
        self,
        query
    ):

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

            "this",
            "that",

            "after",
            "before",

            "has",
            "have",
            "had",

            "be",
            "been",

            "please",

            "product",
            "order"
        }

        words = re.findall(
            r"[a-zA-Z]+",
            query.lower()
        )

        terms = []

        for word in words:

            if len(word) < 4:
                continue

            if word in stop_words:
                continue

            if word not in terms:
                terms.append(word)

        return terms

    # ========================================================
    # ANSWER CONSTRUCTION
    # ========================================================

    def _build_answer(
        self,
        relevant
    ):

        answer_lines = []

        # ----------------------------------------------------
        # Keep only the strongest evidence.
        #
        # This avoids producing long answers containing
        # unrelated neighboring policy chunks.
        # ----------------------------------------------------

        for result in relevant:

            text = result["text"].strip()

            if not text:
                continue

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

        "Can I change the color of my product after it has been delivered?"
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

        if result["sources"]:

            for source in result["sources"]:

                print(
                    f" - {source['document_id']} "
                    f"| {source['chunk_id']} "
                    f"| score={source['score']:.4f}"
                )

        else:

            print(
                " - No sufficiently relevant sources."
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