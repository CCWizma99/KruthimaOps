import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings

warnings.filterwarnings("ignore")

print("=" * 60)
print("[START] ML Opsidian - Fast Fun 3-Minute Ensemble")
print("=" * 60)

# 1. LOAD DATA
DATA_DIR = "data"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

print("[LOAD] Loading dataset...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print(f"   Train shape: {train_df.shape} | Test shape: {test_df.shape}")

# Deduplicate train set
train_df = train_df.drop_duplicates().reset_index(drop=True)
print(f"   After deduplication: {train_df.shape}")

# 2. FEATURE ENGINEERING
print("\n[FEAT] Engineering features...")
def engineer_features(df):
    df = df.copy()
    
    # Inundation area log
    df["inundation_area_log"] = np.log1p(df["inundation_area_sqm"])
    
    # Downstream indicators
    df["confirmed_severe_risk"] = (
        (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes") &
        (df["is_good_to_live"].astype(str).str.strip().str.lower() == "no") &
        (df["inundation_area_sqm"] > 0)
    ).astype(int)
    
    df["no_flood_confirmed"] = (
        (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "no") &
        (df["is_good_to_live"].astype(str).str.strip().str.lower() == "yes")
    ).astype(int)
    
    df["inundation_per_capita"] = df["inundation_area_sqm"] / (np.expm1(df["population_density_per_km2_log1p"]) + 1.0)
    
    has_reason = (~df["reason_not_good_to_live"].astype(str).str.strip().str.lower().isin(["nan", "none", "", "missing", "n/a"])).astype(int)
    df["downstream_risk_count"] = (
        (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes").astype(int) +
        (df["is_good_to_live"].astype(str).str.strip().str.lower() == "no").astype(int) +
        has_reason +
        (df["inundation_area_sqm"] > 0).astype(int)
    )
    
    # River / Rain / Terrain interactions
    df['fluvial_risk_score_feat'] = df['rainfall_7d_mm'] * (1.0 / (df['distance_to_river_m'] + 1.0))
    df['pseudo_twi'] = np.log1p((df['distance_to_river_m'] + 1.0) / (df['elevation_m'].clip(lower=0.0) + 1.0))
    
    # Hand-crafted Target-Inversion Score (TIS)
    df["target_inversion_score"] = (
        0.081 * df["distance_to_river_m_log1p"].fillna(0) 
        - 0.069 * df["inundation_area_log"].fillna(0) 
        - 0.063 * df["rainfall_7d_mm_log1p"].fillna(0) 
        - 0.042 * df["extreme_weather_index"].fillna(0) 
        + 0.037 * df["infrastructure_score"].fillna(0)
    )
    
    # Coordinate precision fingerprints
    df['lat_decimal_len'] = df['latitude'].apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) and str(x).lower() != 'nan' else 0)
    df['lon_decimal_len'] = df['longitude'].apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) and str(x).lower() != 'nan' else 0)
    
    # Deeper environmental / topography interactions
    df["evacuation_difficulty"]   = df["nearest_hospital_km_log1p"] + df["nearest_evac_km_log1p"]
    df["inundation_density_risk"] = df["inundation_area_log"] / (df["population_density_per_km2_log1p"] + 1e-6)
    df["terrain_veg_risk"]        = df["terrain_roughness_index"] * (1.0 - df["ndvi_qmap"].clip(-1, 1))
    df["flood_pressure"]          = df["extreme_weather_index"] * df["seasonal_index"].clip(lower=0)
    df["is_repeat_flood_zone"]    = (df["historical_flood_count"] > 2).astype(int)
    df["rain_spike_ratio"]        = df["rainfall_7d_mm"] / (df["monthly_rainfall_mm"] + 1e-6)
    
    ndwi_clip = df["ndwi_qmap"].clip(lower=0.0)
    ndvi_clip = df["ndvi_qmap"].clip(-1.0, 1.0).clip(lower=0.0)
    df["pooling_vulnerability"] = ndwi_clip * (1.0 - ndvi_clip)
    df["soil_infiltration"] = df['soil_type'].astype(str).map({'Sandy': 0.8, 'Loamy': 0.6, 'Silty': 0.4, 'Clay': 0.2, 'Peaty': 0.1}).fillna(0.4)
    df["soil_saturation_limit"] = df['rainfall_7d_mm'] / (df['soil_infiltration'] + 0.1)
    df["soil_drainage_saturation"] = df["soil_saturation_limit"] * (1.0 - df["drainage_index_yeojohnson"].clip(0.0, 1.0))
    
    return df

train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

# Define column categories
target_col = 'flood_risk_score'
drop_cols = ['record_id', 'place_name', 'is_synthetic', 'generation_date']
ignore_cols = drop_cols + [target_col]
features = [col for col in train_df.columns if col not in ignore_cols]

categorical_features = [
    'district', 'landcover', 'soil_type', 'water_supply', 'electricity', 
    'road_quality', 'urban_rural', 'water_presence_flag', 
    'flood_occurrence_current_event', 'is_good_to_live', 'reason_not_good_to_live'
]

# 3. PREPROCESSING & DTYPES
print("\n[PREPARE] Formatting columns and imputing missing values...")
for col in features:
    if col in categorical_features:
        # Fill missing values with 'missing' and cast as category
        train_df[col] = train_df[col].fillna("missing").astype(str).astype('category')
        test_df[col] = test_df[col].fillna("missing").astype(str).astype('category')
    elif train_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
        # Numeric imputation
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

X = train_df[features]
y = train_df[target_col].values
X_test = test_df[features]

# We also prepare string formatted features for CatBoost
X_cb = X.copy()
X_test_cb = X_test.copy()
for col in categorical_features:
    X_cb[col] = X_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

# 4. TRAINING & EVALUATION (5-Fold KFold with Early Stopping)
N_FOLDS = 5
print(f"\n[TRAIN] Training 5-fold Ensemble (LGBM + XGBoost + CatBoost)...")
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# OOF Arrays
oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_cb  = np.zeros(len(train_df))

# Test Arrays
test_lgb = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))
test_cb  = np.zeros(len(test_df))

fold_reports = []
start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    fold_start = time.time()
    print(f"\n--- Fold {fold + 1} / {N_FOLDS} ---")
    
    # Split
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    X_train_cb, X_val_cb = X_cb.iloc[train_idx], X_cb.iloc[val_idx]
    
    # 4.1 LightGBM
    print("   Training LightGBM...")
    model_lgb = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    oof_lgb[val_idx] = model_lgb.predict(X_val)
    test_lgb += model_lgb.predict(X_test) / N_FOLDS
    
    # 4.2 XGBoost
    print("   Training XGBoost...")
    model_xgb = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2024 + fold,
        tree_method='hist',
        enable_categorical=True,
        n_jobs=-1
    )
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    test_xgb += model_xgb.predict(X_test) / N_FOLDS
    
    # 4.3 CatBoost
    print("   Training CatBoost...")
    model_cb = cb.CatBoostRegressor(
        iterations=2000,
        learning_rate=0.04,
        depth=6,
        eval_metric='MAE',
        random_seed=12345 + fold,
        verbose=0,
        thread_count=-1
    )
    model_cb.fit(
        X_train_cb, y_train,
        cat_features=categorical_features,
        eval_set=[(X_val_cb, y_val)],
        early_stopping_rounds=100,
        verbose=False
    )
    oof_cb[val_idx] = model_cb.predict(X_val_cb)
    test_cb += model_cb.predict(X_test_cb) / N_FOLDS
    
    # Evaluate individual & blended fold predictions
    f_lgb_mae = mean_absolute_error(y_val, oof_lgb[val_idx])
    f_xgb_mae = mean_absolute_error(y_val, oof_xgb[val_idx])
    f_cb_mae  = mean_absolute_error(y_val, oof_cb[val_idx])
    
    fold_blend = (oof_lgb[val_idx] + oof_xgb[val_idx] + oof_cb[val_idx]) / 3.0
    f_blend_mae = mean_absolute_error(y_val, fold_blend)
    f_blend_rmse = root_mean_squared_error(y_val, fold_blend)
    f_blend_ev = explained_variance_score(y_val, fold_blend)
    f_blend_score = (0.539328 * f_blend_mae + 1.152263 * f_blend_rmse) * (1.0 + 0.048467 * (1.0 - f_blend_ev))
    
    fold_reports.append({
        'Fold': fold + 1,
        'LGB_MAE': f_lgb_mae,
        'XGB_MAE': f_xgb_mae,
        'CB_MAE': f_cb_mae,
        'Blend_MAE': f_blend_mae,
        'Blend_RMSE': f_blend_rmse,
        'Blend_EV': f_blend_ev,
        'Blend_Score': f_blend_score,
        'Time': time.time() - fold_start
    })
    print(f"   Fold {fold + 1} finished in {time.time() - fold_start:.2f}s | LGB MAE: {f_lgb_mae:.4f} | XGB MAE: {f_xgb_mae:.4f} | CB MAE: {f_cb_mae:.4f} | Blend Score: {f_blend_score:.5f}")

total_time = time.time() - start_time
print(f"\n[DONE] Training completed in {total_time/60:.2f} minutes!")

# 5. OVERALL METRICS REPORT
def report_metrics(name, oof_preds):
    m_mae = mean_absolute_error(y, oof_preds)
    m_rmse = root_mean_squared_error(y, oof_preds)
    m_ev = explained_variance_score(y, oof_preds)
    m_score = (0.539328 * m_mae + 1.152263 * m_rmse) * (1.0 + 0.048467 * (1.0 - m_ev))
    print(f"* {name:<10} | MAE: {m_mae:.5f} | RMSE: {m_rmse:.5f} | EV: {m_ev:.5f} | LB Estimate: {m_score:.5f}")
    return m_mae, m_rmse, m_ev, m_score

print("\n--- GLOBAL OOF REPORT ---")
report_metrics("LightGBM", oof_lgb)
report_metrics("XGBoost", oof_xgb)
report_metrics("CatBoost", oof_cb)

print("\n--- BLENDED ENSEMBLE REPORT ---")
# Simple average blend
oof_blend = (oof_lgb + oof_xgb + oof_cb) / 3.0
test_blend = (test_lgb + test_xgb + test_cb) / 3.0
mae_b, rmse_b, ev_b, score_b = report_metrics("Blend OOF", oof_blend)

# Save metrics report file for verification
with open("fold_report_fast_fun.csv", "w") as f:
    f.write("Fold,LGB_MAE,XGB_MAE,CB_MAE,Blend_MAE,Blend_RMSE,Blend_ExpVar,Blend_CustomScore\n")
    for r in fold_reports:
        f.write(f"{r['Fold']},{r['LGB_MAE']:.6f},{r['XGB_MAE']:.6f},{r['CB_MAE']:.6f},{r['Blend_MAE']:.6f},{r['Blend_RMSE']:.6f},{r['Blend_EV']:.6f},{r['Blend_Score']:.6f}\n")
    f.write(f"OOF,-,-,-,{mae_b:.6f},{rmse_b:.6f},{ev_b:.6f},{score_b:.6f}\n")

# Save OOF predictions for evaluator integration
os.makedirs("submissions", exist_ok=True)
np.save("submissions/oof_fast_fun.npy", oof_blend)
print("OOF predictions saved to submissions/oof_fast_fun.npy")

# 6. BOUNDARY PRESERVATION & SUBMISSION
# Clip predictions strictly between 0.0 and 1.0
test_blend = np.clip(test_blend, 0.0, 1.0)

submission = pd.DataFrame({
    'record_id': test_df['record_id'],
    'flood_risk_score': test_blend
})
submission.to_csv('submission_fast_fun.csv', index=False)
print("\nSubmission file 'submission_fast_fun.csv' generated and clipped successfully!")
