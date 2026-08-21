import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
df = pd.read_csv("orders_dataset.csv")
X = df.drop(columns=["order_id", "returned"])
y = df["returned"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,stratify=y,random_state=42)
numeric_features = [

    "price_inr",

    "discount_pct",

    "customer_tenure_days",

    "num_previous_orders",

    "num_previous_returns",

    "delivery_distance_km",

    "delivery_days",

    "is_weekend_order",

    "rating_given"

]

categorical_features = [

    "product_category",

    "payment_method"

]

numeric_pipeline = Pipeline(

    steps=[

        ("imputer",
         SimpleImputer(strategy="median")),

        ("scaler",
         StandardScaler())

    ]

)

# =====================================================
# Categorical Pipeline
# =====================================================

categorical_pipeline = Pipeline(

    steps=[

        ("imputer",
         SimpleImputer(strategy="most_frequent")),

        ("encoder",
         OneHotEncoder(handle_unknown="ignore"))

    ]

)

preprocessor = ColumnTransformer(

    transformers=[

        ("num",
         numeric_pipeline,
         numeric_features),

        ("cat",
         categorical_pipeline,
         categorical_features)

    ]

)
dummy_pipeline = Pipeline(steps=[("preprocessor",preprocessor),
                      ("classifier",DummyClassifier(strategy="most_frequent"))])
dummy_pipeline.fit(X_train, y_train)
y_pred = dummy_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1
)



print("="*60)

print("Baseline Dummy Classifier")

print("="*60)

print(f"Accuracy : {accuracy:.4f}")

print(f"F1 Score : {f1:.4f}")

print()

print("Confusion Matrix")

print(confusion_matrix(y_test, y_pred))

print()

print(classification_report(y_test, y_pred))

## Explanation
"""
The DummyClassifier with the most_frequent strategy predicts every order as not returned, 
because this is the majority class in the training data. As a result, it achieves a relatively high accuracy by
correctly classifying most non-returned orders. However, its F1-score for the returned = 1 class is 0.0 because 
it never predicts any returned orders, giving it zero recall for the positive class. This demonstrates 
the common failure mode of high accuracy but zero recall, showing why accuracy alone is misleading for imbalanced datasets.
Comparing against this baseline confirms whether more advanced models actually learn meaningful patterns, and using metrics such 
as F1-score, precision, and recall better aligns evaluation with the business goal of identifying orders that are likely to be returned.
"""