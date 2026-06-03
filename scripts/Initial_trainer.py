import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
import xgboost as xgb

# 1. LOAD DATA
print("🚀 Loading data...")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Hard drop exact duplicates to prevent CV contamination
train_df = train_df.drop_duplicates()

target_col = 'flood_risk_score'

# Drop columns that cause severe overfitting or are backend tracking metadata
drop_cols = ['record_id', 'place_name', 'is_synthetic', 'generation_date']
ignore_cols = drop_cols + [target_col]
features = [col for col in train_df.columns if col not in ignore_cols]

# Explicitly list integer columns that represent categorical classifications
categorical_features = [
    'district', 'landcover', 'soil_type', 'water_supply', 'electricity', 
    'road_quality', 'urban_rural', 'water_presence_flag', 
    'flood_occurrence_current_event', 'is_good_to_live', 'reason_not_good_to_live'
]

# 2. PREPROCESSING
print("🧹 Preprocessing features...")
for col in features:
    if col in categorical_features:
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')
    elif train_df[col].dtype in ['int64', 'float64']:
        # Using median for numeric imputation to stay robust against outlier spikes
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

# Crucial: Keep X and X_test as DataFrames (do not use .values) so XGBoost reads 'category' dtypes
X = train_df[features]
y = train_df[target_col].values
X_test = test_df[features]

# 3. 5-FOLD CROSS VALIDATION FOR REGRESSION
print("🔄 Training baseline XGBRegressor...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(train_df))
test_predictions = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    # Use .iloc since X is a pandas DataFrame holding categorical states
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
        enable_categorical=True # 👈 Critical: Turns on native categorical features handles
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    val_preds = model.predict(X_val)
    oof_predictions[val_idx] = val_preds
    
    # Track test predictions safely across folds
    test_predictions += model.predict(X_test) / kf.n_splits
    print(f"   Fold {fold + 1} finished.")

# 4. EVALUATION REPORT (Tracking both metric pillars)
mae = mean_absolute_error(y, oof_predictions)
rmse = root_mean_squared_error(y, oof_predictions)
exp_var = explained_variance_score(y, oof_predictions)

print("\n📊 --- LOCAL METRIC REPORT ---")
print(f"👉 Balanced Base MAE : {mae:.4f}")
print(f"👉 Balanced Base RMSE: {rmse:.4f}")
print(f"👉 Explained Variance Score: {exp_var:.4f} (Closer to 1.0 = Less Penalty!)")

# 5. BOUNDARY CHECK & SUBMISSION
# Clip predictions strictly between 0 and 1 just in case the regressor overshoots
test_predictions = np.clip(test_predictions, 0.0, 1.0)

submission = pd.DataFrame({
    'record_id': test_df['record_id'],
    'flood_risk_score': test_predictions
})
submission.to_csv('baseline_submission.csv', index=False)
print("\n✅ Submission file generated successfully!")