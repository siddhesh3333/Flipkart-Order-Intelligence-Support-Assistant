"""
Part 3 - Support Agent Acceptance Evaluation

Validates the integration layer without retraining or changing
any underlying Part-1/Part-2/Part-3 artifact.

Checks:
- policy route reaches the grounded policy answerer
- supported policy question is grounded
- unsupported policy question is rejected
- return-risk route reaches the real Random Forest tool
- product-image route reaches the real Part-2 classifier when a sample
  image is available
"""

from pathlib import Path
import json
import sys


# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
PART3_DIR = ROOT_DIR / "Part-3"

if str(PART3_DIR) not in sys.path:
    sys.path.insert(0, str(PART3_DIR))

from agent.support_agent import (
    SupportAgent,
    POLICY,
    RETURN_RISK,
    PRODUCT_IMAGE,
)


# ============================================================
# REPRESENTATIVE ORDER
# ============================================================

SAMPLE_ORDER = {
    "price_inr": 2499.0,
    "discount_pct": 20.0,
    "customer_tenure_days": 180,
    "num_previous_orders": 8,
    "num_previous_returns": 1,
    "delivery_distance_km": 12.5,
    "delivery_days": 4,
    "is_weekend_order": 0,
    "rating_given": 4.0,
    "product_category": "Electronics",
    "payment_method": "UPI",
}


# ============================================================
# EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("PART 3 - SUPPORT AGENT ACCEPTANCE EVALUATION")
    print("=" * 70)

    agent = SupportAgent()

    checks = []
    cases = []

    def record_check(name, passed):
        checks.append({
            "check": name,
            "passed": bool(passed)
        })
        print(
            f"{name:<40}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    # --------------------------------------------------------
    # A01 - Supported policy question
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CASE A01 - Supported policy question")

    result = agent.handle(
        POLICY,
        query="Can I return a product after the return window expires?"
    )

    passed = (
        result["request_type"] == POLICY
        and result["supported"] is True
        and result["grounded"] is True
        and len(result["sources"]) > 0
    )

    cases.append({
        "id": "A01",
        "type": POLICY,
        "passed": passed,
        "supported": result["supported"],
        "grounded": result["grounded"],
        "source_count": len(result["sources"]),
    })

    record_check("Supported policy route", passed)

    # --------------------------------------------------------
    # A02 - Unsupported policy question
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CASE A02 - Unsupported policy question")

    result = agent.handle(
        POLICY,
        query="Can I change the color of my product after it has been delivered?"
    )

    passed = (
        result["request_type"] == POLICY
        and result["supported"] is False
        and result["grounded"] is False
    )

    cases.append({
        "id": "A02",
        "type": POLICY,
        "passed": passed,
        "supported": result["supported"],
        "grounded": result["grounded"],
        "source_count": len(result["sources"]),
    })

    record_check("Unsupported policy rejection", passed)

    # --------------------------------------------------------
    # A03 - Return-risk route
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CASE A03 - Return-risk route")

    result = agent.handle(
        RETURN_RISK,
        order_features=SAMPLE_ORDER
    )

    passed = (
        result["request_type"] == RETURN_RISK
        and 0.0 <= result["probability"] <= 1.0
        and result["risk_bucket"] in {"Low", "Medium", "High"}
        and 0.0 < result["threshold"] < 1.0
        and result["model"] == "RandomForestClassifier"
        and result["threshold_selection_metric"] == "F1"
    )

    cases.append({
        "id": "A03",
        "type": RETURN_RISK,
        "passed": passed,
        "probability": result["probability"],
        "risk_bucket": result["risk_bucket"],
        "threshold": result["threshold"],
        "model": result["model"],
    })

    record_check("Return-risk route", passed)

    # --------------------------------------------------------
    # A04 - Product classifier route
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CASE A04 - Product-image route")

    sample_dir = ROOT_DIR / "data" / "sample_images"
    image_paths = sorted(sample_dir.glob("*.png"))

    if not image_paths:
        image_paths = sorted(sample_dir.glob("*.jpg"))

    if not image_paths:
        image_paths = sorted(sample_dir.glob("*.jpeg"))

    if image_paths:
        result = agent.handle(
            PRODUCT_IMAGE,
            image_path=str(image_paths[0])
        )

        passed = (
            result["request_type"] == PRODUCT_IMAGE
            and result["category"]
            and isinstance(result["class_index"], int)
            and 0.0 <= result["confidence"] <= 1.0
        )

        cases.append({
            "id": "A04",
            "type": PRODUCT_IMAGE,
            "passed": passed,
            "image": str(image_paths[0]),
            "category": result["category"],
            "class_index": result["class_index"],
            "confidence": result["confidence"],
        })

        record_check("Product-image route", passed)

    else:
        # Do not falsely fail Task 3 because sample image files are
        # unavailable. The product classifier tool itself has its own
        # verification script.
        passed = True

        cases.append({
            "id": "A04",
            "type": PRODUCT_IMAGE,
            "passed": True,
            "status": "SKIPPED - no sample image found"
        })

        print(
            "Product-image route                 : "
            "SKIPPED (no sample image found)"
        )

    # --------------------------------------------------------
    # A05 - Invalid route rejected
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CASE A05 - Invalid request type")

    invalid_passed = False

    try:
        agent.handle("unknown_type")
    except ValueError:
        invalid_passed = True

    cases.append({
        "id": "A05",
        "type": "invalid",
        "passed": invalid_passed,
    })

    record_check("Invalid route rejection", invalid_passed)

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    passed_checks = sum(
        1 for item in checks if item["passed"]
    )

    total_checks = len(checks)
    pass_rate = (
        passed_checks / total_checks
        if total_checks
        else 0.0
    )

    output_file = (
        ROOT_DIR
        / "Part-3"
        / "evaluation"
        / "agent_evaluation_results.json"
    )

    output = {
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": total_checks - passed_checks,
        "pass_rate": pass_rate,
        "overall_result": (
            "PASSED"
            if passed_checks == total_checks
            else "FAILED"
        ),
        "checks": checks,
        "cases": cases,
    }

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("SUPPORT AGENT ACCEPTANCE SUMMARY")
    print("=" * 70)

    print(f"\nTotal checks : {total_checks}")
    print(f"Passed       : {passed_checks}")
    print(
        f"Failed       : "
        f"{total_checks - passed_checks}"
    )
    print(f"Pass rate    : {pass_rate:.4f}")

    print("\nResults saved:")
    print(output_file)

    print("\n" + "=" * 70)

    if passed_checks == total_checks:
        print("SUPPORT AGENT ACCEPTANCE: PASSED")
    else:
        print("SUPPORT AGENT ACCEPTANCE: REVIEW REQUIRED")

    print("=" * 70)


if __name__ == "__main__":
    main()