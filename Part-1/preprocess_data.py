import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("orders_dataset.csv")
X = df.drop(columns=["returned"])
y = df["returned"]
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    stratify=y,

    random_state=42
)
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

numeric_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),

        ("scaler",StandardScaler())])

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)
preprocessor = ColumnTransformer(transformers=[("num",numeric_pipeline,numeric_features),
                                               ("cat",categorical_pipeline,categorical_features)])


preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)

X_test_processed = preprocessor.transform(X_test)
print("Training Shape :", X_train_processed.shape)

print("Testing Shape :", X_test_processed.shape)