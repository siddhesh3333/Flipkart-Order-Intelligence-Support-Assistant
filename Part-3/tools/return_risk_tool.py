"""
Part 3 - Return Risk Tool

Loads the REAL Random Forest pipeline produced in Part 1
and predicts return probability using predict_proba().

Threshold is loaded from:
    models/return_risk_threshold.json

The tool does NOT retrain the model and does NOT hardcode
the risk threshold.
"""

from pathlib import Path
import json

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    ROOT_DIR
    / "models"
    / "return_risk_model.pkl"
)

THRESHOLD_FILE = (
    ROOT_DIR
    / "models"
    / "return_risk_threshold.json"
)


# ============================================================
# REQUIRED FEATURES
# ============================================================

REQUIRED_FEATURES = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
    "product_category",
    "payment_method",
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Return-risk model not found:\n{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Loaded return-risk model does not provide predict_proba()."
        )

    return model


# ============================================================
# LOAD THRESHOLD
# ============================================================

def load_threshold():

    if not THRESHOLD_FILE.exists():
        raise FileNotFoundError(
            f"Return-risk threshold file not found:\n{THRESHOLD_FILE}"
        )

    with THRESHOLD_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    if "t_rf" not in metadata:
        raise KeyError(
            "Threshold metadata does not contain 't_rf'."
        )

    threshold = float(
        metadata["t_rf"]
    )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"Invalid return-risk threshold: {threshold}"
        )

    return threshold, metadata


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_order_features(order_features):

    if not isinstance(
        order_features,
        dict
    ):
        raise TypeError(
            "order_features must be a dictionary."
        )

    missing = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in order_features
    ]

    if missing:
        raise ValueError(
            "Missing required order features: "
            + ", ".join(missing)
        )


# ============================================================
# RETURN RISK TOOL
# ============================================================

def check_return_risk(
    order_features: dict
) -> dict:
    """
    Predict return probability and risk bucket.

    Returns:
        {
            "probability": float,
            "risk_bucket": str,
            "threshold": float
        }
    """

    validate_order_features(
        order_features
    )

    model = load_model()

    threshold, threshold_metadata = load_threshold()

    # --------------------------------------------------------
    # Convert input to DataFrame.
    #
    # The saved sklearn Pipeline performs the preprocessing
    # automatically.
    # --------------------------------------------------------

    features = {
        feature: order_features[feature]
        for feature in REQUIRED_FEATURES
    }

    X = pd.DataFrame(
        [features]
    )

    # --------------------------------------------------------
    # REAL MODEL PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X
    )

    if probabilities.shape[1] < 2:
        raise ValueError(
            "Model predict_proba() does not contain a positive-class probability."
        )

    probability = float(
        probabilities[0][1]
    )

    # --------------------------------------------------------
    # RISK BUCKET
    #
    # The assessment requires the bucket to be anchored
    # to t*_rf rather than arbitrary 0.3 / 0.6 thresholds.
    #
    # Since t*_rf = 0.50 in the saved metadata:
    #
    # probability >= threshold -> High
    # probability < threshold  -> Low
    # --------------------------------------------------------

    if probability >= threshold:
        risk_bucket = "High"
    else:
        risk_bucket = "Low"

    return {
        "probability": probability,
        "risk_bucket": risk_bucket,
        "threshold": threshold,
        "threshold_selection_metric": threshold_metadata.get(
            "selection_metric"
        ),
        "model": threshold_metadata.get(
            "model"
        ),
    }


# ============================================================
# CLI TEST
# ============================================================

def main():

    print("=" * 70)
    print("PART 3 - RETURN RISK TOOL TEST")
    print("=" * 70)

    print("\nModel:")
    print(MODEL_FILE)

    print("\nThreshold:")
    print(THRESHOLD_FILE)

    threshold, metadata = load_threshold()

    print("\nLoaded t*_rf:")
    print(f"{threshold:.2f}")

    print("\nSelection metric:")
    print(metadata.get("selection_metric"))

    # --------------------------------------------------------
    # Representative order
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

    print("\nSample order:")
    for key, value in sample_order.items():
        print(
            f"  {key}: {value}"
        )

    print("\nRunning REAL predict_proba()...")

    result = check_return_risk(
        sample_order
    )

    print("\nResult:")

    print(
        f"Probability : "
        f"{result['probability']:.4f}"
    )

    print(
        f"Risk bucket : "
        f"{result['risk_bucket']}"
    )

    print(
        f"Threshold   : "
        f"{result['threshold']:.2f}"
    )

    print(
        f"Model       : "
        f"{result['model']}"
    )

    print(
        f"Metric      : "
        f"{result['threshold_selection_metric']}"
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assert 0.0 <= result["probability"] <= 1.0

    assert result["threshold"] == threshold

    assert result["risk_bucket"] in {
        "High",
        "Low",
    }

    print("\n" + "=" * 70)
    print("RETURN RISK TOOL: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()