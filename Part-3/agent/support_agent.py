"""
Part 3 - Support Agent Orchestrator

Integrates the three already-validated Part-3 capabilities:

1. PolicyAnswerer
2. Return-risk tool
3. Product-image classifier

This module does not train models, alter thresholds, or implement
new ML logic. It only routes a request to the appropriate existing
component and returns a consistent structured response.
"""

from pathlib import Path
import sys


# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
PART3_DIR = ROOT_DIR / "Part-3"

if str(PART3_DIR) not in sys.path:
    sys.path.insert(0, str(PART3_DIR))


from rag.answerer import PolicyAnswerer
from tools.return_risk_tool import check_return_risk
from tools.product_classifier_tool import classify_product_image


# ============================================================
# SUPPORTED REQUEST TYPES
# ============================================================

POLICY = "policy"
RETURN_RISK = "return_risk"
PRODUCT_IMAGE = "product_image"


# ============================================================
# SUPPORT AGENT
# ============================================================

class SupportAgent:
    """
    Thin orchestration layer for the Task-3 support workflow.

    The agent deliberately does not:
      - retrain any model
      - modify the Random Forest threshold
      - generate policy facts
      - replace the Part-2 classifier
      - call an external LLM/API
    """

    def __init__(self, policy_answerer=None):
        self.policy_answerer = (
            policy_answerer
            if policy_answerer is not None
            else PolicyAnswerer(top_k=5)
        )

    # ========================================================
    # POLICY
    # ========================================================

    def answer_policy(self, query: str) -> dict:
        """Answer a policy question using the grounded RAG answerer."""
        result = self.policy_answerer.answer(query)

        return {
            "request_type": POLICY,
            "supported": result["supported"],
            "grounded": result["grounded"],
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_count": result["retrieved_count"],
            "relevant_count": result["relevant_count"],
        }

    # ========================================================
    # RETURN RISK
    # ========================================================

    def assess_return_risk(self, order_features: dict) -> dict:
        """Assess return risk using the real Part-1 Random Forest tool."""
        result = check_return_risk(order_features)

        return {
            "request_type": RETURN_RISK,
            "probability": result["probability"],
            "risk_bucket": result["risk_bucket"],
            "threshold": result["threshold"],
            "threshold_selection_metric": result[
                "threshold_selection_metric"
            ],
            "model": result["model"],
        }

    # ========================================================
    # PRODUCT IMAGE
    # ========================================================

    def classify_product(self, image_path: str) -> dict:
        """Classify an image using the real Part-2 classifier."""
        result = classify_product_image(image_path)

        return {
            "request_type": PRODUCT_IMAGE,
            "image_path": result["image_path"],
            "category": result["category"],
            "class_index": result["class_index"],
            "confidence": result["confidence"],
        }

    # ========================================================
    # EXPLICIT ROUTING
    # ========================================================

    def handle(self, request_type: str, **kwargs) -> dict:
        """
        Route an explicitly typed support request.

        Explicit request types are intentional here: Task 3 requires
        integration of the validated capabilities, not a second
        unvalidated intent-classification model.
        """

        if not isinstance(request_type, str):
            raise TypeError("request_type must be a string.")

        request_type = request_type.strip().lower()

        if request_type == POLICY:
            if "query" not in kwargs:
                raise ValueError("Policy request requires 'query'.")
            return self.answer_policy(kwargs["query"])

        if request_type == RETURN_RISK:
            if "order_features" not in kwargs:
                raise ValueError(
                    "Return-risk request requires 'order_features'."
                )
            return self.assess_return_risk(kwargs["order_features"])

        if request_type == PRODUCT_IMAGE:
            if "image_path" not in kwargs:
                raise ValueError(
                    "Product-image request requires 'image_path'."
                )
            return self.classify_product(kwargs["image_path"])

        raise ValueError(
            f"Unsupported request_type '{request_type}'. "
            f"Supported types: {POLICY}, {RETURN_RISK}, {PRODUCT_IMAGE}"
        )


# ============================================================
# CLI SMOKE TEST
# ============================================================

def main():
    print("=" * 70)
    print("PART 3 - SUPPORT AGENT INTEGRATION TEST")
    print("=" * 70)

    agent = SupportAgent()

    # --------------------------------------------------------
    # Policy route
    # --------------------------------------------------------

    policy_result = agent.handle(
        POLICY,
        query="Can I return a product after the return window expires?"
    )

    assert policy_result["request_type"] == POLICY
    assert policy_result["supported"] is True
    assert policy_result["grounded"] is True
    assert policy_result["sources"]

    print("\nPOLICY ROUTE: PASS")
    print("Answer:")
    print(policy_result["answer"])

    # --------------------------------------------------------
    # Return-risk route
    # --------------------------------------------------------

    sample_order = {
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

    risk_result = agent.handle(
        RETURN_RISK,
        order_features=sample_order
    )

    assert risk_result["request_type"] == RETURN_RISK
    assert 0.0 <= risk_result["probability"] <= 1.0
    assert risk_result["risk_bucket"] in {"Low", "High", "Medium"}
    assert 0.0 < risk_result["threshold"] < 1.0
    assert risk_result["model"] == "RandomForestClassifier"

    print("\nRETURN-RISK ROUTE: PASS")
    print(
        f"Probability: {risk_result['probability']:.4f}"
    )
    print(
        f"Risk bucket: {risk_result['risk_bucket']}"
    )
    print(
        f"Threshold: {risk_result['threshold']:.2f}"
    )

    # --------------------------------------------------------
    # Product-image route is validated separately because
    # it requires an actual sample image path.
    # --------------------------------------------------------

    print("\nPRODUCT-IMAGE ROUTE: READY")
    print(
        "Use agent.handle('product_image', image_path=<PNG/JPG>) "
        "to invoke the real Part-2 classifier."
    )

    print("\n" + "=" * 70)
    print("SUPPORT AGENT INTEGRATION TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()