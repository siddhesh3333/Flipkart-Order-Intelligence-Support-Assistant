import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Load data
df = pd.read_csv('orders_dataset.csv')
X = df.drop(columns=['order_id','returned'])
y = df['returned']

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

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

numeric_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_pipeline, numeric_features), ('cat', cat_pipeline, categorical_features)])

lr = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42, max_iter=1000)
pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', lr)])

pipe.fit(X_train, y_train)

probs = pipe.predict_proba(X_test)[:,1]

default_preds = (probs >= 0.5).astype(int)

acc = accuracy_score(y_test, default_preds)
f1 = f1_score(y_test, default_preds, pos_label=1)
precision = precision_score(y_test, default_preds, pos_label=1, zero_division=0)
recall = recall_score(y_test, default_preds, pos_label=1, zero_division=0)
roc = roc_auc_score(y_test, probs)

print('='*60)
print('LOGISTIC REGRESSION - DEFAULT THRESHOLD (0.5)')
print('='*60)
print(f'Accuracy : {acc:.4f}')
print(f'F1 (class=1): {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall   : {recall:.4f}')
print(f'ROC-AUC : {roc:.4f}')

# Threshold sweep
thresholds = np.arange(0.1,0.91,0.02)
best = None
results = []
for t in thresholds:
    preds = (probs >= t).astype(int)
    p = precision_score(y_test, preds, pos_label=1, zero_division=0)
    r = recall_score(y_test, preds, pos_label=1, zero_division=0)
    f = f1_score(y_test, preds, pos_label=1, zero_division=0)
    results.append({'threshold': round(float(t),2), 'precision': p, 'recall': r, 'f1': f})
    if best is None or f > best['f1']:
        best = {'threshold': round(float(t),2), 'precision': p, 'recall': r, 'f1': f}

print('\nBEST THRESHOLD FROM SWEEP (LOGISTIC)')
print(f"Threshold: {best['threshold']:.2f}")
print(f"Precision: {best['precision']:.4f}")
print(f"Recall   : {best['recall']:.4f}")
print(f"F1       : {best['f1']:.4f}")

recall_increase = best['recall'] - recall
print('\nRecall increase vs default:', round(recall_increase*100,2), 'percentage points')

# Print numeric drop in precision
precision_drop = precision - best['precision']
print('Precision drop vs default:', round(precision_drop*100,2), 'percentage points')

# Exit status

