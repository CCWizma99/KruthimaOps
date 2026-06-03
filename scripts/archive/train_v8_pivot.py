import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
import xgboost as xgb

# 1. LOAD DATA
print("[INFO] Loading data...")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

target_col = 'flood_risk_score'

# 2. THE PIVOT: Retain ONLY the generative columns and key spatial anchors
core_downstream = [
    'flood_occurrence_current_event', 
    'inundation_area_sqm', 
    'is_good_to_live', 
    'reason_not_good_to_live'
]
# Retaining district as a broad geographic regularizer
features = core_downstream + ['district'] 

# 3. INTERACTION SIGNAL ENGINEERING
print("[INFO] Constructing the generative key signature...")
for df in [train_df, test_df]:
    # Combine categorical indicators into a unique logical signature
    df['downstream_sig'] = (
        df['flood_occurrence_current_event'].astype(str) + "_" +
        df['is_good_to_live'].astype(str) + "_" +
        df['reason_not_good_to_live'].astype(str)
    )
    
    # Contextualize inundation area relative to its specific failure reason
    df['sig_mean_inundation'] = df.groupby('downstream_sig')['inundation_area_sqm'].transform('mean')
    df['inundation_ratio'] = df['inundation_area_sqm'] / (df['sig_mean_inundation'] + 1e-5)

# Append our newly engineered features to the feature list
extended_features = features + ['downstream_sig', 'inundation_ratio']

# Cast categorical columns explicitly for XGBoost native handling
categorical_cols = ['flood_occurrence_current_event', 'is_good_to_live', 'reason_not_good_to_live', 'district', 'downstream_sig']
for col in extended_features:
    if col in categorical_cols:
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')

X = train_df[extended_features]
y = train_df[target_col].values
X_test = test_df[extended_features]

# 4. 5-FOLD CV WITH SAFE OUT-OF-FOLD TARGET ENCODING
print("[INFO] Training focused Rule-Extraction Model...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(train_df))
test_predictions = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx].copy(), y[train_idx]
    X_val, y_val = X.iloc[val_idx].copy(), y[val_idx]
    X_test_fold = X_test.copy()
    
    # Strict Out-of-Fold Target Encoding for the signature column
    # This maps the exact mean target per signature without leaking validation info
    train_fold_df = train_df.iloc[train_idx]
    sig_target_map = train_fold_df.groupby('downstream_sig', observed=False)[target_col].mean().to_dict()
    global_mean = y_train.mean()
    
    X_train['sig_encoded'] = X_train['downstream_sig'].astype(str).map(sig_target_map).astype(float).fillna(global_mean)
    X_val['sig_encoded'] = X_val['downstream_sig'].astype(str).map(sig_target_map).astype(float).fillna(global_mean)
    X_test_fold['sig_encoded'] = X_test_fold['downstream_sig'].astype(str).map(sig_target_map).astype(float).fillna(global_mean)
    
    # Initialize a shallower tree model focused purely on the structural signal
    model = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=4,             # Shallow depth prevents overfitting to synthetic noise
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        tree_method='hist',
        enable_categorical=True,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    val_preds = model.predict(X_val)
    oof_predictions[val_idx] = val_preds
    test_predictions += model.predict(X_test_fold) / kf.n_splits
    print(f"   Fold {fold + 1} finalized. (Best iteration: {model.best_iteration})")

# 5. EVALUATION METRIC REPORT
mae = mean_absolute_error(y, oof_predictions)
rmse = root_mean_squared_error(y, oof_predictions)
exp_var = explained_variance_score(y, oof_predictions)

print("\n--- LOCAL METRIC REPORT (PIVOTED) ---")
print(f"  Balanced Base MAE : {mae:.4f}")
print(f"  Balanced Base RMSE: {rmse:.4f}")
print(f"  Explained Variance Score: {exp_var:.4f}")

# 6. EXPORT
test_predictions = np.clip(test_predictions, 0.0, 1.0)
submission = pd.DataFrame({'record_id': test_df['record_id'], 'flood_risk_score': test_predictions})
submission.to_csv('pivoted_submission.csv', index=False)
print("\n[DONE] Pivoted rule-extraction submission file generated!")
