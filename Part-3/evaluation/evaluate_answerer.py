"""
Part 3 - Policy Grounded Answerer Evaluation

Evaluates:
    - supported policy questions
    - unsupported / insufficiently supported questions
    - grounding
    - source presence
    - relevant policy usage
    - safe fallback behavior

Output:
    Part-3/evaluation/answerer_evaluation_results.json
"""

from pathlib import Path
import json
import sys


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

EVALUATION_DIR = (
    ROOT_DIR
    / "Part-3"
    / "evaluation"
)

OUTPUT_FILE = (
    EVALUATION_DIR
    / "answerer_evaluation_results.json"
)


# ============================================================
# IMPORT ANSWERER
# ============================================================

RAG_DIR = (
    ROOT_DIR
    / "Part-3"
    / "rag"
)

if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from answerer import PolicyAnswerer


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [

    {
        "id": "A01",
        "query": (
            "Can I return a product after the return window expires?"
        ),
        "expected_supported": True,
        "relevant_document_ids": [
            "POL-001"
        ],
    },

    {
        "id": "A02",
        "query": (
            "What conditions must a product satisfy for a return?"
        ),
        "expected_supported": True,
        "relevant_document_ids": [
            "POL-001"
        ],
    },

    {
        "id": "A03",
        "query": (
            "Which products may not be eligible for return?"
        ),
        "expected_supported": True,
        "relevant_document_ids": [
            "POL-001"
        ],
    },

    {
        "id": "A04",
        "query": (
            "What should I do if I receive a defective product?"
        ),
        "expected_supported": True,
        "relevant_document_ids": [
            "POL-007"
        ],
    },

    {
        "id": "A05",
        "query": (
            "When should I use the warranty process?"
        ),
        "expected_supported": True,
        "relevant_document_ids": [
            "POL-007",
            "POL-010"
        ],
    },

    {
        "id": "A06",
        "query": (
            "What should I do if I receive the wrong product?"
        ),
        "expected_supported": True,
        "relevant_document_ids": [
            "POL-006"
        ],
    },

    {
        "id": "A07",
        "query": (
            "Can I change the color of my product "
            "after it has been delivered?"
        ),
        "expected_supported": False,
        "relevant_document_ids": [],
    },
]


# ============================================================
# HELPERS
# ============================================================

def get_document_ids(result):

    """
    Extract unique document IDs from answerer sources.
    """

    sources = result.get(
        "sources",
        []
    )

    document_ids = []

    for source in sources:

        document_id = source.get(
            "document_id"
        )

        if document_id:

            document_ids.append(
                document_id
            )

    return sorted(
        set(document_ids)
    )


# ============================================================
# CASE EVALUATION
# ============================================================

def evaluate_case(
    answerer,
    case
):

    query = case["query"]

    expected_supported = (
        case["expected_supported"]
    )

    relevant_documents = set(
        case["relevant_document_ids"]
    )

    # --------------------------------------------------------
    # Run answerer
    # --------------------------------------------------------

    result = answerer.answer(
        query
    )

    actual_supported = bool(
        result.get(
            "supported",
            False
        )
    )

    grounded = bool(
        result.get(
            "grounded",
            False
        )
    )

    sources = result.get(
        "sources",
        []
    )

    # Use canonical count fields.
    retrieved = result.get(
        "retrieved_count",
        0
    )

    relevant = result.get(
        "relevant_count",
        0
    )

    source_document_ids = set(
        get_document_ids(
            result
        )
    )

    # --------------------------------------------------------
    # Supported question checks
    # --------------------------------------------------------

    if expected_supported:

        support_match = (
            actual_supported
            == expected_supported
        )

        grounded_check = (
            grounded is True
        )

        sources_check = (
            len(sources) > 0
        )

        relevant_source_check = (
            len(
                source_document_ids
                & relevant_documents
            ) > 0
        )

        passed = all([
            support_match,
            grounded_check,
            sources_check,
            relevant_source_check,
        ])

    # --------------------------------------------------------
    # Unsupported question checks
    # --------------------------------------------------------

    else:

        support_match = (
            actual_supported
            == expected_supported
        )

        grounded_check = (
            grounded is False
        )

        sources_check = (
            len(sources) == 0
        )

        relevant_source_check = (
            len(
                source_document_ids
                & relevant_documents
            ) == 0
        )

        passed = all([
            support_match,
            grounded_check,
            sources_check,
            relevant_source_check,
        ])

    # --------------------------------------------------------
    # Return detailed evaluation result
    # --------------------------------------------------------

    return {

        "id": case["id"],

        "query": query,

        "expected_supported":
            expected_supported,

        "actual_supported":
            actual_supported,

        "grounded":
            grounded,

        "retrieved":
            retrieved,

        "relevant":
            relevant,

        "source_document_ids":
            sorted(
                source_document_ids
            ),

        "expected_document_ids":
            sorted(
                relevant_documents
            ),

        "source_count":
            len(sources),

        "support_match":
            support_match,

        "grounded_check":
            grounded_check,

        "sources_check":
            sources_check,

        "relevant_source_check":
            relevant_source_check,

        "passed":
            passed,

        "answer":
            result.get(
                "answer",
                ""
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PART 3 - POLICY ANSWERER EVALUATION")
    print("=" * 70)

    print("\nTest cases:")
    print(
        len(TEST_CASES)
    )

    print("\nLoading answerer...")

    answerer = PolicyAnswerer()

    print(
        "\nAnswerer loaded successfully."
    )

    results = []

    # ========================================================
    # RUN TEST CASES
    # ========================================================

    for index, case in enumerate(
        TEST_CASES,
        start=1
    ):

        print(
            "\n"
            + "-"
            * 70
        )

        print(
            f"Test {index}/{len(TEST_CASES)}"
        )

        print(
            "ID       :",
            case["id"]
        )

        print(
            "Question :",
            case["query"]
        )

        result = evaluate_case(
            answerer,
            case
        )

        results.append(
            result
        )

        print(
            "Expected supported:",
            result[
                "expected_supported"
            ]
        )

        print(
            "Actual supported  :",
            result[
                "actual_supported"
            ]
        )

        print(
            "Grounded          :",
            result[
                "grounded"
            ]
        )

        print(
            "Retrieved          :",
            result[
                "retrieved"
            ]
        )

        print(
            "Relevant           :",
            result[
                "relevant"
            ]
        )

        print(
            "Sources            :",
            result[
                "source_count"
            ]
        )

        print(
            "Source documents   :",
            result[
                "source_document_ids"
            ]
        )

        print(
            "Result             :",
            "PASSED"
            if result["passed"]
            else "FAILED"
        )

    # ========================================================
    # METRICS
    # ========================================================

    total = len(
        results
    )

    passed = sum(
        1
        for result in results
        if result["passed"]
    )

    failed = (
        total
        - passed
    )

    supported_cases = [
        result
        for result in results
        if result[
            "expected_supported"
        ]
    ]

    unsupported_cases = [
        result
        for result in results
        if not result[
            "expected_supported"
        ]
    ]

    supported_passed = sum(
        1
        for result in supported_cases
        if result["passed"]
    )

    unsupported_passed = sum(
        1
        for result in unsupported_cases
        if result["passed"]
    )

    overall_pass_rate = (
        passed / total
        if total
        else 0.0
    )

    supported_pass_rate = (
        supported_passed
        / len(supported_cases)
        if supported_cases
        else 0.0
    )

    unsupported_pass_rate = (
        unsupported_passed
        / len(unsupported_cases)
        if unsupported_cases
        else 0.0
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    evaluation_output = {

        "evaluation":
            "policy_answerer",

        "total_cases":
            total,

        "passed_cases":
            passed,

        "failed_cases":
            failed,

        "overall_pass_rate":
            overall_pass_rate,

        "supported_cases":
            len(supported_cases),

        "supported_cases_passed":
            supported_passed,

        "supported_pass_rate":
            supported_pass_rate,

        "unsupported_cases":
            len(unsupported_cases),

        "unsupported_cases_passed":
            unsupported_passed,

        "unsupported_pass_rate":
            unsupported_pass_rate,

        "acceptance_criteria": {

            "supported_queries_are_supported":
                True,

            "supported_queries_are_grounded":
                True,

            "supported_queries_have_sources":
                True,

            "supported_queries_use_relevant_documents":
                True,

            "unsupported_queries_are_rejected":
                True,

            "unsupported_queries_are_not_marked_grounded":
                True,

            "unsupported_queries_have_no_policy_sources":
                True,
        },

        "results":
            results,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation_output,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print(
        "\n"
        + "="
        * 70
    )

    print(
        "ANSWERER EVALUATION SUMMARY"
    )

    print(
        "="
        * 70
    )

    print(
        f"\nTotal cases       : "
        f"{total}"
    )

    print(
        f"Passed cases      : "
        f"{passed}"
    )

    print(
        f"Failed cases      : "
        f"{failed}"
    )

    print(
        f"Overall pass rate : "
        f"{overall_pass_rate:.4f}"
    )

    print(
        f"\nSupported cases   : "
        f"{len(supported_cases)}"
    )

    print(
        f"Supported passed  : "
        f"{supported_passed}"
    )

    print(
        f"Supported rate    : "
        f"{supported_pass_rate:.4f}"
    )

    print(
        f"\nUnsupported cases : "
        f"{len(unsupported_cases)}"
    )

    print(
        f"Unsupported passed: "
        f"{unsupported_passed}"
    )

    print(
        f"Unsupported rate  : "
        f"{unsupported_pass_rate:.4f}"
    )

    print(
        "\nResults saved:"
    )

    print(
        OUTPUT_FILE
    )

    # ========================================================
    # PASS / FAIL
    # ========================================================

    if failed == 0:

        print(
            "\n"
            + "="
            * 70
        )

        print(
            "POLICY ANSWERER EVALUATION: PASSED"
        )

        print(
            "="
            * 70
        )

    else:

        print(
            "\n"
            + "="
            * 70
        )

        print(
            "POLICY ANSWERER EVALUATION: REVIEW REQUIRED"
        )

        print(
            "="
            * 70
        )


if __name__ == "__main__":
    main()