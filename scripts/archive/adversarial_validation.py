import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import os

print("=========================================================")
print("  ADVERSARIAL VALIDATION: CHECKING FOR COVARIATE SHIFT")
print("=========================================================")

# 1. Load Data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Create target: 0 for train, 1 for test
train['is_test'] = 0
test['is_test'] = 1

# 2. Combine and prep
combined = pd.concat([train.drop(columns=['flood_risk_score'], errors='ignore'), test], ignore_index=True)

# Drop identifiers and backend columns that aren't used in real modeling
drop_cols = ['record_id', 'place_name', 'is_synthetic', 'generation_date', 'is_test']
features = [c for c in combined.columns if c not in drop_cols]

# 3. Handle categorical columns with LabelEncoder for LGBM
for col in features:
    if combined[col].dtype == 'object':
        combined[col] = combined[col].astype(str).fillna("missing")
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col])
    else:
        combined[col] = combined[col].fillna(combined[col].median())

X = combined[features]
y = combined['is_test']

# 4. Train Adversarial Model (5-Fold CV)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y))
feature_importances = np.zeros(len(features))

print(f"Checking {len(features)} raw features for distribution shifts...")

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42 + fold,
        n_jobs=-1,
        verbosity=-1
    )
    
    # We don't use early stopping because we want to see if it can fit the shift at all
    model.fit(X_tr, y_tr)
    
    oof_preds[va_idx] = model.predict_proba(X_va)[:, 1]
    feature_importances += model.feature_importances_ / 5.0

# 5. Evaluate
auc = roc_auc_score(y, oof_preds)
print("\n" + "=" * 57)
print(f"  ADVERSARIAL AUC SCORE: {auc:.5f}")
print("=" * 57)

if auc > 0.60:
    print("\n[WARNING] Massive Covariate Shift Detected!")
    print("The train and test sets have different distributions.")
    print("Dropping the top features below may drastically improve generalization:")
elif auc > 0.52:
    print("\n[NOTE] Slight Covariate Shift Detected.")
    print("There are minor differences between the train and test sets.")
else:
    print("\n[SUCCESS] No Covariate Shift Detected! (AUC ~0.50)")
    print("The train and test sets are perfectly identical in distribution.")
    print("Your current models will generalize perfectly to the test set.")

print("\nTop 15 Features Distinguishing Train vs Test:")
imp_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
imp_df = imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
print(imp_df.head(15))
