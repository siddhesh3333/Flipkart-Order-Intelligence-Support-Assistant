import joblib
import json
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'return_risk_model.pkl')
THRESHOLD_PATH = os.path.join(PROJECT_ROOT, 'models', 'return_risk_threshold.json')
DATA_PATH = os.path.join(BASE_DIR, 'orders_dataset.csv')

model = joblib.load(MODEL_PATH)
with open(THRESHOLD_PATH,'r',encoding='utf-8') as f:
    meta = json.load(f)
    t_rf = float(meta.get('t_rf',0.5))

print('Using threshold t_rf=', t_rf)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['order_id','returned'])
y = df['returned']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

# Predict probabilities and apply threshold
probs = model.predict_proba(X_test)[:,1]
preds = (probs >= t_rf).astype(int)
# Align preds to X_test index for subgroup slicing
preds_series = pd.Series(preds, index=X_test.index)

# Overall metrics
overall_precision = precision_score(y_test, preds_series, pos_label=1, zero_division=0)
overall_recall = recall_score(y_test, preds_series, pos_label=1, zero_division=0)
overall_f1 = f1_score(y_test, preds_series, pos_label=1, zero_division=0)
print('\nOverall (thresholded) precision, recall, f1:', round(overall_precision,4), round(overall_recall,4), round(overall_f1,4))

# Subgroup by product_category
print('\nSubgroup metrics by product_category:')
prod_groups = []
for grp, rows in X_test.groupby('product_category'):
    idx = rows.index
    p = precision_score(y_test.loc[idx], preds_series.loc[idx], pos_label=1, zero_division=0)
    r = recall_score(y_test.loc[idx], preds_series.loc[idx], pos_label=1, zero_division=0)
    f = f1_score(y_test.loc[idx], preds_series.loc[idx], pos_label=1, zero_division=0)
    prod_groups.append((grp, len(idx), p, r, f))

prod_df = pd.DataFrame(prod_groups, columns=['product_category','n','precision','recall','f1'])
print(prod_df.sort_values('recall'))

# Subgroup by payment_method
print('\nSubgroup metrics by payment_method:')
pay_groups = []
for grp, rows in X_test.groupby('payment_method'):
    idx = rows.index
    p = precision_score(y_test.loc[idx], preds_series.loc[idx], pos_label=1, zero_division=0)
    r = recall_score(y_test.loc[idx], preds_series.loc[idx], pos_label=1, zero_division=0)
    f = f1_score(y_test.loc[idx], preds_series.loc[idx], pos_label=1, zero_division=0)
    pay_groups.append((grp, len(idx), p, r, f))

pay_df = pd.DataFrame(pay_groups, columns=['payment_method','n','precision','recall','f1'])
print(pay_df.sort_values('recall'))

# Identify weakest subgroup by recall relative to overall
weak_prod = prod_df.iloc[prod_df['recall'].argmin()]
weak_pay = pay_df.iloc[pay_df['recall'].argmin()]

print('\nWeakest product_category by recall:', weak_prod.to_dict())
print('Weakest payment_method by recall:', weak_pay.to_dict())

# Propose fixes: if a subgroup recall is >0.1 below overall recall, suggest a specific fix
if overall_recall - weak_prod['recall'] > 0.10:
    print('\nProposed fix for product_category', weak_prod['product_category'], ':')
    print('- Consider category-specific threshold adjustment (increase sensitivity for this category)')
    print("- Or add category-specific features such as more granular product attributes or return-reason encoding if available")

if overall_recall - weak_pay['recall'] > 0.10:
    print('\nProposed fix for payment_method', weak_pay['payment_method'], ':')
    print('- Consider adding interaction features between payment_method and price/discount')
    print('- Or apply a payment-method-specific decision threshold to boost recall for this payment type')

print('\nDone')
