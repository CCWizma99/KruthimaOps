import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score
from xgboost import XGBClassifier, XGBRegressor

print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Prep Target
y = train_df['flood_risk_score'].values
y_synth = train_df['is_synthetic'].fillna(False).astype(int).values

# Prep Features
features = [c for c in train_df.columns if c not in ['record_id', 'flood_risk_score', 'is_synthetic', 'generation_date']]
X = train_df[features].copy()
X_test = test_df[features].copy()

# Categoricals
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype('category')
    X_test[c] = X_test[c].astype('category')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- STAGE 1: META-FEATURE GENERATION (Synthetic Fingerprint) ---
print("\n[STAGE 1] Extracting Synthetic Fingerprint...")
oof_synth_prob = np.zeros(len(train_df))
test_synth_prob = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, y_tr = X.iloc[train_idx], y_synth[train_idx]
    X_va = X.iloc[val_idx]
    
    clf = XGBClassifier(n_estimators=300, enable_categorical=True, random_state=42, eval_metric='logloss')
    clf.fit(X_tr, y_tr)
    
    oof_synth_prob[val_idx] = clf.predict_proba(X_va)[:, 1]
    test_synth_prob += clf.predict_proba(X_test)[:, 1] / kf.n_splits

X['prob_synthetic'] = oof_synth_prob
X_test['prob_synthetic'] = test_synth_prob

# --- STAGE 2: MAIN REGRESSOR ---
print("\n[STAGE 2] Training Regressor with Fingerprint Meta-Feature...")
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    reg = XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.01, enable_categorical=True, random_state=42, tree_method='hist', early_stopping_rounds=50)
    reg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    oof_preds[val_idx] = reg.predict(X_va)
    test_preds += reg.predict(X_test) / kf.n_splits
    print(f"  Fold {fold+1} finished.")

ev = explained_variance_score(y, oof_preds)
print(f"\nFinal EV with Synthetic Fingerprint: {ev:.4f}")

# Export Blend
test_preds = np.clip(test_preds, 0.0, 1.0)
sub = pd.DataFrame({'record_id': test_df['record_id'], 'flood_risk_score': test_preds})
sub.to_csv('submission_v9_fingerprint.csv', index=False)
print("Saved submission_v9_fingerprint.csv")
