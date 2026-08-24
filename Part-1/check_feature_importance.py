import joblib
import pandas as pd
import numpy as np
import os
from sklearn.inspection import permutation_importance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'return_risk_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'orders_dataset.csv')

print('Loading model:', MODEL_PATH)
model = joblib.load(MODEL_PATH)

# Load data
print('Loading data:', DATA_PATH)
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['order_id','returned'])
y = df['returned']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

# Extract preprocessor and classifier
preprocessor = model.named_steps['preprocessor']
classifier = model.named_steps['classifier']

# Obtain transformed feature names
try:
    # If ColumnTransformer has get_feature_names_out
    feature_names = preprocessor.get_feature_names_out()
except Exception:
    # Fallback: reconstruct names from transformers_
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'remainder':
            continue
        if hasattr(transformer, 'named_steps') and 'encoder' in transformer.named_steps:
            encoder = transformer.named_steps['encoder']
            # try to get categories
            try:
                cats = encoder.get_feature_names_out(cols)
                feature_names.extend(list(cats))
            except Exception:
                # generic expansion
                for c in cols:
                    feature_names.append(c)
        else:
            if isinstance(cols, (list, tuple)):
                feature_names.extend(list(cols))
            else:
                try:
                    feature_names.extend(list(cols))
                except Exception:
                    feature_names.append(str(cols))

print('\nNumber of transformed features:', len(feature_names))

# classifier.feature_importances_
if hasattr(classifier, 'feature_importances_'):
    importances = classifier.feature_importances_
    idx = np.argsort(importances)[::-1]
    top5_idx = idx[:5]
    print('\nTop-5 impurity-based feature importances:')
    for i in top5_idx:
        name = feature_names[i] if i < len(feature_names) else f'col_{i}'
        print(f"{name}: {importances[i]:.6f}")
else:
    print('Classifier has no feature_importances_')
    raise SystemExit(1)

# Permutation importance on transformed features for same top-5
print('\nComputing permutation importance (n_repeats=10, scoring="f1")...')
X_test_trans = preprocessor.transform(X_test)
perm = permutation_importance(classifier, X_test_trans, y_test, n_repeats=10, random_state=42, scoring='f1', n_jobs=-1)
perm_mean = perm.importances_mean

# report permutation importance for the top-5 impurity features
print('\nPermutation importances for top-5 impurity features:')
for i in top5_idx:
    name = feature_names[i] if i < len(feature_names) else f'col_{i}'
    print(f"{name}: mean={perm_mean[i]:.6f}")

# Compare rankings
imp_ranking = [feature_names[i] for i in idx]
perm_ranking_idx = np.argsort(perm_mean)[::-1]
perm_ranking = [feature_names[i] for i in perm_ranking_idx]

print('\nTop 10 by impurity:')
print(imp_ranking[:10])
print('\nTop 10 by permutation:')
print(perm_ranking[:10])

# Identify which of top-5 lose most under permutation (difference in rank)
imp_positions = {name: pos for pos, name in enumerate(imp_ranking)}
perm_positions = {name: pos for pos, name in enumerate(perm_ranking)}

print('\nTop-5 comparison (impurity_rank, permutation_rank, drop):')
for name in imp_ranking[:5]:
    ir = imp_positions.get(name, None)
    pr = perm_positions.get(name, None)
    drop = (pr - ir) if (ir is not None and pr is not None) else None
    print(f"{name}: impurity_pos={ir}, perm_pos={pr}, drop={drop}")

print('\nDone')
