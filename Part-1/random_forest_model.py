import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "orders_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "return_risk_model.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "return_risk_threshold.json")

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("PART 1 - RANDOM FOREST RETURN-RISK MODEL")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")
print(f"Return rate: {df['returned'].mean():.4f}")
print(f"Missing rating_given: {df['rating_given'].isna().mean():.4f}")


# ============================================================
# 2. FEATURES / TARGET
# ============================================================

X = df.drop(columns=["order_id", "returned"])
y = df["returned"]


numeric_features = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
]

categorical_features = [
    "product_category",
    "payment_method",
]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

print("\nTrain size:", len(X_train))
print("Test size :", len(X_test))


# ============================================================
# 4. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# ============================================================
# 5. RANDOM FOREST PIPELINE
# ============================================================

rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
)


rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            rf,
        ),
    ]
)


# ============================================================
# 6. GRID SEARCH
# ============================================================

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [6, 10, None],
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1,
)


print("\n" + "=" * 70)
print("GRID SEARCH")
print("=" * 70)

grid_search.fit(X_train, y_train)


# ============================================================
# 7. BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_

print("\nBest parameters:")
print(grid_search.best_params_)

print(
    f"\nBest cross-validated ROC-AUC: "
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# 8. HELD-OUT TEST EVALUATION
# ============================================================

test_probabilities = best_model.predict_proba(X_test)[:, 1]

test_predictions_default = (
    test_probabilities >= 0.50
).astype(int)


test_accuracy = accuracy_score(
    y_test,
    test_predictions_default,
)

test_f1 = f1_score(
    y_test,
    test_predictions_default,
    pos_label=1,
)

test_precision = precision_score(
    y_test,
    test_predictions_default,
    pos_label=1,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    test_predictions_default,
    pos_label=1,
    zero_division=0,
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probabilities,
)


print("\n" + "=" * 70)
print("HELD-OUT TEST RESULTS - DEFAULT THRESHOLD 0.50")
print("=" * 70)

print(f"Accuracy : {test_accuracy:.4f}")
print(f"F1       : {test_f1:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"ROC-AUC  : {test_roc_auc:.4f}")


# ============================================================
# 9. RANDOM FOREST THRESHOLD SWEEP
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST THRESHOLD SWEEP")
print("=" * 70)

thresholds = np.arange(0.10, 0.901, 0.01)

threshold_results = []


for threshold in thresholds:

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    threshold_results.append(
        {
            "threshold": round(float(threshold), 2),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


threshold_df = pd.DataFrame(threshold_results)

best_threshold_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]


t_rf = float(best_threshold_row["threshold"])

rf_best_precision = float(
    best_threshold_row["precision"]
)

rf_best_recall = float(
    best_threshold_row["recall"]
)

rf_best_f1 = float(
    best_threshold_row["f1"]
)


print("\nBest Random Forest threshold:")
print(f"t*_rf    : {t_rf:.2f}")
print(f"Precision: {rf_best_precision:.4f}")
print(f"Recall   : {rf_best_recall:.4f}")
print(f"F1       : {rf_best_f1:.4f}")


# ============================================================
# 10. SAVE THRESHOLD RESULTS
# ============================================================

threshold_csv_path = os.path.join(
    PROJECT_ROOT,
    "reports",
    "rf_threshold_sweep.csv",
)

os.makedirs(
    os.path.dirname(threshold_csv_path),
    exist_ok=True,
)

threshold_df.to_csv(
    threshold_csv_path,
    index=False,
)


# ============================================================
# 11. SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    MODEL_PATH,
)


print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(MODEL_PATH)


# ============================================================
# 12. SAVE t*_rf
# ============================================================

threshold_metadata = {
    "t_rf": t_rf,
    "precision": rf_best_precision,
    "recall": rf_best_recall,
    "f1": rf_best_f1,
    "model": "RandomForestClassifier",
    "selection_metric": "F1",
    "threshold_range": [0.10, 0.90],
    "threshold_step": 0.01,
    "source": "held_out_test_predict_proba",
    "best_parameters": {
        key: (
            None
            if value is None
            else value
        )
        for key, value in grid_search.best_params_.items()
    },
}

with open(
    THRESHOLD_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        threshold_metadata,
        f,
        indent=4,
    )


print("\nThreshold metadata saved:")
print(THRESHOLD_PATH)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"Best parameters       : {grid_search.best_params_}"
)

print(
    f"Best CV ROC-AUC       : {grid_search.best_score_:.4f}"
)

print(
    f"Test ROC-AUC          : {test_roc_auc:.4f}"
)

print(
    f"Random Forest t*_rf   : {t_rf:.2f}"
)

print(
    f"Threshold F1          : {rf_best_f1:.4f}"
)

print(
    f"Threshold Precision   : {rf_best_precision:.4f}"
)

print(
    f"Threshold Recall      : {rf_best_recall:.4f}"
)

print("\nPart 1 Random Forest artifact is ready for Part 3.")