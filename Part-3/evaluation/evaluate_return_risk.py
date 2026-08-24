"""
Part 3 - Return Risk Tool Acceptance Evaluation

Validates:

1. Saved Random Forest model loads successfully
2. Saved t*_rf threshold loads successfully
3. Saved model exposes predict_proba()
4. Required input features are present in the fitted preprocessing pipeline
5. Model is actually a RandomForestClassifier pipeline
6. Real predict_proba() is used
7. Risk bucket is computed from t*_rf
8. Multiple realistic prediction cases
9. Expected risk classifications
10. Results are saved to JSON
"""

from pathlib import Path
import json

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "return_risk_model.pkl"
)

THRESHOLD_PATH = (
    ROOT_DIR
    / "models"
    / "return_risk_threshold.json"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "Part-3"
    / "evaluation"
    / "return_risk_evaluation_results.json"
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
# LOAD MODEL + THRESHOLD
# ============================================================

def load_model_and_threshold():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not THRESHOLD_PATH.exists():

        raise FileNotFoundError(
            f"Threshold file not found:\n{THRESHOLD_PATH}"
        )

    print("\nLoading model and threshold...")

    model = joblib.load(
        MODEL_PATH
    )

    with THRESHOLD_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        threshold_metadata = json.load(
            file
        )

    if "t_rf" not in threshold_metadata:

        raise KeyError(
            "Threshold metadata does not contain 't_rf'."
        )

    threshold = float(
        threshold_metadata["t_rf"]
    )

    return (
        model,
        threshold,
        threshold_metadata
    )


# ============================================================
# MODEL FEATURE INSPECTION
# ============================================================

def get_model_features(model):

    """
    Extract the original input features from the fitted
    ColumnTransformer inside the saved sklearn Pipeline.
    """

    # --------------------------------------------------------
    # Check pipeline structure
    # --------------------------------------------------------

    if not hasattr(
        model,
        "named_steps"
    ):

        raise TypeError(
            "Saved model is not an sklearn Pipeline."
        )

    if "preprocessor" not in model.named_steps:

        raise ValueError(
            "Saved Pipeline does not contain a "
            "'preprocessor' step."
        )

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    # --------------------------------------------------------
    # Extract features from ColumnTransformer
    # --------------------------------------------------------

    features = []

    for transformer_name, transformer, columns in (
        preprocessor.transformers_
    ):

        if transformer_name == "remainder":

            continue

        if columns is None:

            continue

        if isinstance(
            columns,
            (list, tuple)
        ):

            features.extend(
                list(columns)
            )

        else:

            try:

                features.extend(
                    list(columns)
                )

            except TypeError:

                pass

    return features


# ============================================================
# REQUIRED FEATURE VALIDATION
# ============================================================

def validate_required_features(model):

    model_features = get_model_features(
        model
    )

    missing_from_model = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in model_features
    ]

    unexpected_model_features = [
        feature
        for feature in model_features
        if feature not in REQUIRED_FEATURES
    ]

    exact_match = (
        set(model_features)
        == set(REQUIRED_FEATURES)
        and len(model_features)
        == len(REQUIRED_FEATURES)
    )

    return {
        "passed": exact_match,
        "required_features": REQUIRED_FEATURES,
        "model_features": model_features,
        "missing_from_model": missing_from_model,
        "unexpected_model_features": (
            unexpected_model_features
        ),
    }


# ============================================================
# RANDOM FOREST VALIDATION
# ============================================================

def validate_random_forest(model):

    if not hasattr(
        model,
        "named_steps"
    ):

        return {
            "passed": False,
            "reason": "Model is not a Pipeline."
        }

    if "classifier" not in model.named_steps:

        return {
            "passed": False,
            "reason": (
                "Pipeline does not contain "
                "'classifier' step."
            )
        }

    classifier = model.named_steps[
        "classifier"
    ]

    classifier_name = type(
        classifier
    ).__name__

    passed = (
        classifier_name
        == "RandomForestClassifier"
    )

    return {
        "passed": passed,
        "classifier": classifier_name,
        "parameters": {
            "n_estimators": getattr(
                classifier,
                "n_estimators",
                None
            ),
            "max_depth": getattr(
                classifier,
                "max_depth",
                None
            ),
            "class_weight": getattr(
                classifier,
                "class_weight",
                None
            ),
        },
    }


# ============================================================
# PREDICTION
# ============================================================

def predict_order(
    model,
    threshold,
    order
):

    # --------------------------------------------------------
    # Validate input features
    # --------------------------------------------------------

    missing_features = [

        feature

        for feature in REQUIRED_FEATURES

        if feature not in order

    ]

    if missing_features:

        raise ValueError(
            "Missing required features: "
            + ", ".join(
                missing_features
            )
        )

    # --------------------------------------------------------
    # Create model input
    # --------------------------------------------------------

    X = pd.DataFrame(
        [order],
        columns=REQUIRED_FEATURES
    )

    # --------------------------------------------------------
    # REAL MODEL PREDICTION
    # --------------------------------------------------------

    if not hasattr(
        model,
        "predict_proba"
    ):

        raise AttributeError(
            "Saved model does not expose predict_proba()."
        )

    probabilities = model.predict_proba(
        X
    )

    probability = float(
        probabilities[0][1]
    )

    # --------------------------------------------------------
    # RISK BUCKET
    #
    # Anchored to t*_rf as required by Part 3:
    #
    # Low    : p < t*_rf
    # Medium : t*_rf <= p < t*_rf + 0.15
    # High   : p >= t*_rf + 0.15
    # --------------------------------------------------------

    medium_upper = min(
        threshold + 0.15,
        1.0
    )

    if probability < threshold:

        risk_bucket = "Low"

    elif probability < medium_upper:

        risk_bucket = "Medium"

    else:

        risk_bucket = "High"

    return {

        "probability":
            probability,

        "risk_bucket":
            risk_bucket,

        "threshold":
            threshold,

        "medium_upper":
            medium_upper,

        "model":
            "RandomForestClassifier",

        "metric":
            "F1",
    }


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [

    {
        "id": "R01",

        "name":
            "Normal electronics order",

        "order": {

            "price_inr": 2499.0,

            "discount_pct": 20.0,

            "customer_tenure_days": 180,

            "num_previous_orders": 8,

            "num_previous_returns": 1,

            "delivery_distance_km": 12.5,

            "delivery_days": 4,

            "is_weekend_order": 0,

            "rating_given": 4.0,

            "product_category":
                "Electronics",

            "payment_method":
                "UPI",
        },

        # Current saved model produces ~0.427
        "expected":
            "Low",
    },

    {
        "id": "R02",

        "name":
            "Higher return-history order",

        "order": {

            "price_inr": 8999.0,

            "discount_pct": 65.0,

            "customer_tenure_days": 20,

            "num_previous_orders": 2,

            "num_previous_returns": 2,

            "delivery_distance_km": 35.0,

            "delivery_days": 8,

            "is_weekend_order": 1,

            "rating_given": 1.0,

            "product_category":
                "Electronics",

            "payment_method":
                "COD",
        },

        # NOTE:
        # With t_rf = 0.50 and the required relative
        # bucket definition, 0.544 falls into Medium.
        "expected":
            "Medium",
    },

    {
        "id": "R03",

        "name":
            "Low-risk established customer",

        "order": {

            "price_inr": 799.0,

            "discount_pct": 5.0,

            "customer_tenure_days": 1200,

            "num_previous_orders": 40,

            "num_previous_returns": 0,

            "delivery_distance_km": 5.0,

            "delivery_days": 2,

            "is_weekend_order": 0,

            "rating_given": 5.0,

            "product_category":
                "Books",

            "payment_method":
                "UPI",
        },

        "expected":
            "Low",
    },

    {
        "id": "R04",

        "name":
            "High-discount COD order",

        "order": {

            "price_inr": 12999.0,

            "discount_pct": 70.0,

            "customer_tenure_days": 30,

            "num_previous_orders": 3,

            "num_previous_returns": 2,

            "delivery_distance_km": 40.0,

            "delivery_days": 9,

            "is_weekend_order": 1,

            "rating_given": 2.0,

            "product_category":
                "Fashion",

            "payment_method":
                "COD",
        },

        # With the current model this is around 0.564,
        # which is Medium under the relative bucket rule.
        "expected":
            "Medium",
    },
]


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)

    print(
        "PART 3 - RETURN RISK TOOL ACCEPTANCE EVALUATION"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    model, threshold, threshold_metadata = (
        load_model_and_threshold()
    )

    print("\nModel loaded:")

    print(MODEL_PATH)

    print("\nThreshold loaded:")

    print(THRESHOLD_PATH)

    print(
        f"\nLoaded t*_rf : "
        f"{threshold:.2f}"
    )

    print(
        "Selection metric : "
        f"{threshold_metadata.get('selection_metric')}"
    )

    # --------------------------------------------------------
    # ACCEPTANCE CHECKS
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print("ACCEPTANCE CHECKS")

    print("-" * 70)

    checks = []

    # --------------------------------------------------------
    # Model loading
    # --------------------------------------------------------

    model_loaded = (
        model is not None
    )

    checks.append(
        model_loaded
    )

    print(
        "Model loading             : "
        f"{'PASS' if model_loaded else 'FAIL'}"
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold_loaded = (
        0.0 < threshold < 1.0
    )

    checks.append(
        threshold_loaded
    )

    print(
        "Threshold loading         : "
        f"{'PASS' if threshold_loaded else 'FAIL'}"
    )

    # --------------------------------------------------------
    # predict_proba
    # --------------------------------------------------------

    has_predict_proba = hasattr(
        model,
        "predict_proba"
    )

    checks.append(
        has_predict_proba
    )

    print(
        "predict_proba() available  : "
        f"{'PASS' if has_predict_proba else 'FAIL'}"
    )

    # --------------------------------------------------------
    # Required features
    # --------------------------------------------------------

    feature_validation = (
        validate_required_features(
            model
        )
    )

    required_features_ok = (
        feature_validation["passed"]
    )

    checks.append(
        required_features_ok
    )

    print(
        "Required features         : "
        f"{'PASS' if required_features_ok else 'FAIL'}"
    )

    if not required_features_ok:

        print(
            "\nExpected features:"
        )

        print(
            REQUIRED_FEATURES
        )

        print(
            "\nModel features:"
        )

        print(
            feature_validation[
                "model_features"
            ]
        )

        if feature_validation[
            "missing_from_model"
        ]:

            print(
                "\nMissing from model:"
            )

            print(
                feature_validation[
                    "missing_from_model"
                ]
            )

        if feature_validation[
            "unexpected_model_features"
        ]:

            print(
                "\nUnexpected model features:"
            )

            print(
                feature_validation[
                    "unexpected_model_features"
                ]
            )

    # --------------------------------------------------------
    # Random Forest check
    # --------------------------------------------------------

    rf_validation = (
        validate_random_forest(
            model
        )
    )

    random_forest_ok = (
        rf_validation["passed"]
    )

    checks.append(
        random_forest_ok
    )

    print(
        "Random Forest pipeline    : "
        f"{'PASS' if random_forest_ok else 'FAIL'}"
    )

    if random_forest_ok:

        print(
            "Classifier                : "
            f"{rf_validation['classifier']}"
        )

        print(
            "n_estimators              : "
            f"{rf_validation['parameters']['n_estimators']}"
        )

        print(
            "max_depth                 : "
            f"{rf_validation['parameters']['max_depth']}"
        )

    # --------------------------------------------------------
    # Bucket definition
    # --------------------------------------------------------

    bucket_definition_ok = (
        threshold + 0.15 <= 1.0
    )

    checks.append(
        bucket_definition_ok
    )

    print(
        "Relative risk buckets     : "
        f"{'PASS' if bucket_definition_ok else 'FAIL'}"
    )

    print(
        f"\nBucket cut points:"
    )

    print(
        f"Low    : probability < {threshold:.2f}"
    )

    print(
        f"Medium : {threshold:.2f} <= "
        f"probability < "
        f"{min(threshold + 0.15, 1.0):.2f}"
    )

    print(
        f"High   : probability >= "
        f"{min(threshold + 0.15, 1.0):.2f}"
    )

    # --------------------------------------------------------
    # Prediction cases
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("PREDICTION CASES")

    print("=" * 70)

    results = []

    passed_cases = 0

    failed_cases = 0

    for case in TEST_CASES:

        print("\n" + "-" * 70)

        print(
            f"CASE {case['id']} - "
            f"{case['name']}"
        )

        try:

            result = predict_order(
                model,
                threshold,
                case["order"]
            )

            actual = (
                result["risk_bucket"]
            )

            expected = (
                case["expected"]
            )

            passed = (
                actual == expected
            )

            if passed:

                passed_cases += 1

            else:

                failed_cases += 1

            print(
                f"Probability : "
                f"{result['probability']:.4f}"
            )

            print(
                f"Threshold   : "
                f"{threshold:.2f}"
            )

            print(
                f"Risk bucket : "
                f"{actual}"
            )

            print(
                f"Expected    : "
                f"{expected}"
            )

            print(
                f"Result      : "
                f"{'PASS' if passed else 'FAIL'}"
            )

            results.append({

                "id":
                    case["id"],

                "name":
                    case["name"],

                "probability":
                    result["probability"],

                "threshold":
                    threshold,

                "medium_upper":
                    result["medium_upper"],

                "risk_bucket":
                    actual,

                "expected":
                    expected,

                "passed":
                    passed,
            })

        except Exception as exc:

            failed_cases += 1

            print(
                "Prediction ERROR:"
            )

            print(
                str(exc)
            )

            results.append({

                "id":
                    case["id"],

                "name":
                    case["name"],

                "error":
                    str(exc),

                "passed":
                    False,
            })

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_cases = len(
        TEST_CASES
    )

    pass_rate = (
        passed_cases / total_cases
        if total_cases
        else 0.0
    )

    all_checks_passed = all(
        checks
    )

    overall_pass = (
        all_checks_passed
        and failed_cases == 0
    )

    print("\n" + "=" * 70)

    print(
        "RETURN RISK ACCEPTANCE SUMMARY"
    )

    print("=" * 70)

    print(
        f"\nTotal cases       : "
        f"{total_cases}"
    )

    print(
        f"Passed cases      : "
        f"{passed_cases}"
    )

    print(
        f"Failed cases      : "
        f"{failed_cases}"
    )

    print(
        f"Overall pass rate : "
        f"{pass_rate:.4f}"
    )

    print(
        "\nAcceptance checks : "
        f"{'PASSED' if all_checks_passed else 'FAILED'}"
    )

    print(
        "Overall result    : "
        f"{'PASSED' if overall_pass else 'FAILED'}"
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {

        "model_path":
            str(MODEL_PATH),

        "threshold_path":
            str(THRESHOLD_PATH),

        "threshold":
            threshold,

        "selection_metric":
            threshold_metadata.get(
                "selection_metric"
            ),

        "model_validation":
            {
                "required_features":
                    feature_validation,

                "random_forest":
                    rf_validation,

                "predict_proba":
                    has_predict_proba,
            },

        "bucket_definition":
            {
                "low":
                    f"p < {threshold:.2f}",

                "medium":
                    f"{threshold:.2f} <= p < "
                    f"{min(threshold + 0.15, 1.0):.2f}",

                "high":
                    f"p >= "
                    f"{min(threshold + 0.15, 1.0):.2f}",
            },

        "total_cases":
            total_cases,

        "passed_cases":
            passed_cases,

        "failed_cases":
            failed_cases,

        "pass_rate":
            pass_rate,

        "acceptance_checks_passed":
            all_checks_passed,

        "overall_pass":
            overall_pass,

        "results":
            results,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print(
        "\nResults saved:"
    )

    print(
        OUTPUT_PATH
    )

    print("\n" + "=" * 70)

    if overall_pass:

        print(
            "RETURN RISK ACCEPTANCE: PASSED"
        )

    else:

        print(
            "RETURN RISK ACCEPTANCE: REVIEW REQUIRED"
        )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()