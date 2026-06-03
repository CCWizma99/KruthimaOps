"""
ML Opsidian: Genesis - Advanced Feature Engineering Pipeline v11
=================================================================
This script implements:
  - Geospatial Hot-Deck Imputation (Place+District matching & Spatial KNN)
  - Climatological & Hydrological engineered features:
      * Monsoon Impact Score (Wet/Dry zones x Yala/Maha monsoons)
      * Urban Pluvial vs Rural Fluvial Runoff Engine
      * Soil Sponge Infiltration Physics
      * Slope Proxies, TWI, and Flatness Indices
      * Outlier residuals & Vulnerability Crosses
  - KFold-safe out-of-fold target encoding for district, grid_id, downstream_sig, and infra_deficit_sig
  - Stratified 5-Fold Ensemble (XGBoost, LightGBM, CatBoost)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
import time

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# 1. LOAD & DEDUPLICATE
# -----------------------------------------------------------------
print("[LOAD] Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")
print(f"   Train shape      : {train_df.shape}")
print(f"   Test shape       : {test_df.shape}")
train_df = train_df.drop_duplicates()
print(f"   Train after dedup: {train_df.shape}")

# -----------------------------------------------------------------
# 2. GEOSPATIAL HOT-DECK IMPUTATION
# -----------------------------------------------------------------
print("[IMPUTE] Starting Geospatial Hot-Deck Imputation...")

# Combine train and test to build the largest possible coordinate donor pool
combined = pd.concat([
    train_df.drop(columns=['flood_risk_score'], errors='ignore'),
    test_df
], ignore_index=True)

# A. Create Place-District coordinate lookup map
print("   -> Creating coordinate lookup maps from place_name and district...")
coords_lookup = combined.groupby(['place_name', 'district'])[['latitude', 'longitude']].median().to_dict('index')

# Impute missing coordinates using place+district lookup
imputed_coords_count = 0
for df in [train_df, test_df]:
    mask = df['latitude'].isnull() & df['place_name'].notnull() & df['district'].notnull()
    for idx in df[mask].index:
        key = (df.loc[idx, 'place_name'], df.loc[idx, 'district'])
        if key in coords_lookup and not np.isnan(coords_lookup[key]['latitude']):
            df.loc[idx, 'latitude'] = coords_lookup[key]['latitude']
            df.loc[idx, 'longitude'] = coords_lookup[key]['longitude']
            imputed_coords_count += 1
print(f"   -> Imputed missing coordinates for {imputed_coords_count} rows using place+district matches.")

# B. Train Spatial KNN Regressors for elevation and river distance
print("   -> Training spatial KNN models to impute elevation and river distance...")
for col in ['elevation_m', 'distance_to_river_m']:
    donor_pool = combined.dropna(subset=['latitude', 'longitude', col])
    knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
    knn.fit(donor_pool[['latitude', 'longitude']], donor_pool[col])
    
    imputed_col_count = 0
    for df in [train_df, test_df]:
        missing_mask = df[col].isnull() & df['latitude'].notnull() & df['longitude'].notnull()
        if missing_mask.any():
            imputed_values = knn.predict(df.loc[missing_mask, ['latitude', 'longitude']])
            df.loc[missing_mask, col] = imputed_values
            imputed_col_count += len(imputed_values)
    print(f"      * Imputed missing {col} for {imputed_col_count} rows using spatial KNN.")

# C. Fallback: district medians for remaining missing values
print("   -> Filling remaining missing physical values with district medians...")
for col in ['elevation_m', 'distance_to_river_m', 'latitude', 'longitude']:
    for df in [train_df, test_df]:
        df[col] = df[col].fillna(df.groupby('district')[col].transform('median'))
        # Global fallback
        df[col] = df[col].fillna(train_df[col].median())

# Verify that no missing values remain in coordinates and elevation
assert train_df[['latitude', 'longitude', 'elevation_m', 'distance_to_river_m']].isnull().sum().sum() == 0, "Nulls remain in train spatial columns!"
assert test_df[['latitude', 'longitude', 'elevation_m', 'distance_to_river_m']].isnull().sum().sum() == 0, "Nulls remain in test spatial columns!"

# -----------------------------------------------------------------
# 3. FEATURE ENGINEERING ENGINE
# -----------------------------------------------------------------
print("[FEAT] Engineering features...")

# Pre-calculate district-level elevation standard deviation as flatness indicator
combined_imputed = pd.concat([
    train_df.drop(columns=['flood_risk_score'], errors='ignore'),
    test_df
], ignore_index=True)
district_elev_std = combined_imputed.groupby('district')['elevation_m'].std().to_dict()

# Soil infiltration rates mapping
soil_infilt_map = {
    'Sandy': 0.8,
    'Loamy': 0.6,
    'Silty': 0.4,
    'Clay': 0.2,
    'Peaty': 0.1
}

# Cyclone corridor districts
cyclone_districts = {'Batticaloa', 'Trincomalee', 'Ampara', 'Mullaitivu', 'Jaffna'}

# Wet Zone districts (zone_code = 1)
wet_zone_districts = {'Colombo', 'Gampaha', 'Kalutara', 'Galle', 'Matara', 'Ratnapura', 'Kegalle'}

def engineer_features(df):
    df = df.copy()
    
    # 1. Downstream signature key
    df['downstream_sig'] = (
        df['flood_occurrence_current_event'].astype(str).str.strip() + "_" +
        df['is_good_to_live'].astype(str).str.strip() + "_" +
        df['reason_not_good_to_live'].astype(str).str.strip()
    )
    
    # 2. Extract calendar month
    date_series = pd.to_datetime(df['generation_date'])
    df['month'] = date_series.dt.month
    
    # 3. Monsoon Switch Flag & Score
    df['is_yala'] = df['month'].isin([5, 6, 7, 8, 9]).astype(int)
    df['is_maha'] = df['month'].isin([11, 12, 1]).astype(int)
    df['zone_code'] = df['district'].astype(str).map(lambda x: 1 if x in wet_zone_districts else 2)
    df['monsoon_impact'] = df['rainfall_7d_mm'] * df['is_yala'] * (df['zone_code'] == 1).astype(int) + \
                           df['rainfall_7d_mm'] * df['is_maha'] * (df['zone_code'] == 2).astype(int)
                           
    # 4. Urban Pluvial vs. Rural Fluvial
    df['urban_runoff_potential'] = df['rainfall_7d_mm'] * df['built_up_percent'] * (1.0 / (df['drainage_index'] + 1e-5))
    df['fluvial_risk_score_feat'] = df['rainfall_7d_mm'] * (1.0 / (df['distance_to_river_m'] + 1.0))
    
    # 5. Soil saturation limit
    df['soil_infiltration'] = df['soil_type'].astype(str).map(soil_infilt_map).fillna(0.4)
    df['soil_saturation_limit'] = df['rainfall_7d_mm'] / (df['soil_infiltration'] + 0.1)
    
    # 6. Pseudo-TWI & Flatness
    df['pseudo_twi'] = np.log1p((df['distance_to_river_m'] + 1.0) / (df['elevation_m'].clip(lower=0.0) + 1.0))
    df['flatness_index'] = df['district'].astype(str).map(district_elev_std).fillna(df['elevation_m'].std())
    
    # 7. Cyclone vulnerability flag
    df['in_cyclone_path'] = df['district'].astype(str).map(lambda x: 1 if x in cyclone_districts else 0)
    df['cyclone_vulnerability'] = df['in_cyclone_path'] * df['extreme_weather_index']
    
    # 8. Slope Proxy
    df['slope_proxy'] = df['elevation_m'] / (df['distance_to_river_m'] + 1.0)
    
    # 9. Coordinate Multipliers
    df['lat_lon_multiply'] = df['latitude'] * df['longitude']
    df['lat_lon_divide'] = df['latitude'] / (df['longitude'] + 1e-5)
    
    # 10. Soil Saturation
    df['soil_saturation'] = df['rainfall_7d_mm'] * (1.0 - df['drainage_index'])
    
    # 11. Isolation indices
    df['isolation_index'] = np.log1p(df['nearest_hospital_km']) + np.log1p(df['nearest_evac_km'])
    df['vulnerability'] = df['isolation_index'] / (df['infrastructure_score'] + 1.0)
    df['isolation_multiplier'] = df['nearest_hospital_km'] * df['nearest_evac_km']
    
    # 12. Transformation Divergence
    df['elevation_divergence'] = df['elevation_m'] - df['elevation_m_yeojohnson']
    
    # 13. Infrastructure Deficit String
    df['infra_deficit_sig'] = (
        df['water_supply'].astype(str).str.strip() + "_" +
        df['electricity'].astype(str).str.strip() + "_" +
        df['road_quality'].astype(str).str.strip()
    )
    
    # --- Previous v10 baseline features ---
    df["inundation_area_log"] = np.log1p(df["inundation_area_sqm"])
    df["flood_occurrence_yes"] = (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes").astype(int)
    df["inundation_flood_interaction"] = df["flood_occurrence_yes"] * df["inundation_area_log"]
    df["river_rain_interaction"]  = df["distance_to_river_m_log1p"] * df["rainfall_7d_mm_log1p"]
    df["river_monthly_exposure"]  = df["distance_to_river_m_log1p"] * df["monthly_rainfall_mm_log1p"]
    df["elev_rain_risk"]          = df["elevation_m_yeojohnson"] / (df["rainfall_7d_mm_log1p"] + 1e-6)
    df["water_signal"]            = df["ndwi_qmap"].clip(lower=0)
    df["drainage_deficit"]        = (df["rainfall_7d_mm_log1p"] + 1) * (1.0 - df["drainage_index_yeojohnson"].clip(0, 1))
    df["infra_resilience"]        = df["infrastructure_score"] / (df["population_density_per_km2_log1p"] + 1e-6)
    df["evacuation_difficulty"]   = df["nearest_hospital_km_log1p"] + df["nearest_evac_km_log1p"]
    df["inundation_density_risk"] = df["inundation_area_log"] / (df["population_density_per_km2_log1p"] + 1e-6)
    df["terrain_veg_risk"]        = df["terrain_roughness_index"] * (1.0 - df["ndvi_qmap"].clip(-1, 1))
    df["flood_pressure"]          = df["extreme_weather_index"] * df["seasonal_index"].clip(lower=0)
    df["is_repeat_flood_zone"]    = (df["historical_flood_count"] > 2).astype(int)
    df["rain_spike_ratio"]        = df["rainfall_7d_mm"] / (df["monthly_rainfall_mm"] + 1e-6)
    df["confirmed_risk"]          = (
        (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes") &
        (df["is_good_to_live"].astype(str).str.strip().str.lower() == "no")
    ).astype(int)
    
    # Spatial bins for out-of-fold target encoding
    lat = df["latitude"].fillna(df["latitude"].median())
    lon = df["longitude"].fillna(df["longitude"].median())
    df["lat_bin"] = (lat / 0.5).astype(int)
    df["lon_bin"] = (lon / 0.5).astype(int)
    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
    
    # Remove raw skewed inundation feature
    df = df.drop(columns=["inundation_area_sqm"])
    
    return df

train_df = engineer_features(train_df)
test_df  = engineer_features(test_df)

# -----------------------------------------------------------------
# 4. COLUMN TAXONOMY & DTYPE CASTING
# -----------------------------------------------------------------
TARGET    = "flood_risk_score"
ID_COL    = "record_id"
DROP_COLS = [ID_COL, "place_name", "is_synthetic", "generation_date"]

CAT_FEATURES = [
    "district", "landcover", "soil_type", "water_supply",
    "electricity", "road_quality", "urban_rural",
    "water_presence_flag", "flood_occurrence_current_event",
    "is_good_to_live", "reason_not_good_to_live",
    "downstream_sig", "infra_deficit_sig"
]

IGNORE_COLS = DROP_COLS + [TARGET, "flood_occurrence_yes"]
SPATIAL_HELPERS = ["lat_bin", "lon_bin", "grid_id"]

BASE_FEATURES = [c for c in train_df.columns
                 if c not in IGNORE_COLS and c not in SPATIAL_HELPERS]

print("[PREP] Casting dtypes...")
cat_dtype_map = {}
for col in BASE_FEATURES:
    if col in CAT_FEATURES:
        train_df[col] = train_df[col].fillna("missing").astype(str)
        test_df[col]  = test_df[col].fillna("missing").astype(str)
        all_vals = sorted(set(train_df[col].unique()) | set(test_df[col].unique()))
        cdt = pd.CategoricalDtype(categories=all_vals, ordered=False)
        train_df[col] = train_df[col].astype(cdt)
        test_df[col]  = test_df[col].astype(cdt)
        cat_dtype_map[col] = cdt
    elif train_df[col].dtype in ["int64", "float64", "int32", "float32"]:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col]  = test_df[col].fillna(median_val)

print(f"   Number of Base Features: {len(BASE_FEATURES)}")

# -----------------------------------------------------------------
# 5. CROSS VALIDATION SETUP
# -----------------------------------------------------------------
N_FOLDS     = 5
y           = train_df[TARGET]
GLOBAL_MEAN = float(y.mean())
y_bins = pd.cut(y, bins=10, labels=False)
skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train_df))
oof_lgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
tst_xgb = np.zeros(len(test_df))
tst_lgb = np.zeros(len(test_df))
tst_cat = np.zeros(len(test_df))

fold_results     = []
cat_feature_names = [c for c in CAT_FEATURES if c in BASE_FEATURES]

# Helper: model-specific DataFrame converters
def to_xgb_fmt(df):
    df = df.copy()
    for col in df.columns:
        if hasattr(df[col], "cat"):
            df[col] = df[col].cat.codes.astype("int32")
    return df

def to_cat_fmt(df):
    df = df.copy()
    for col in cat_feature_names:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

# -----------------------------------------------------------------
# 6. TRAINING LOOP WITH IN-FOLD TARGET ENCODING
# -----------------------------------------------------------------
print("\n" + "="*65)
print("  5-FOLD STRATIFIED CV -- XGBoost + LightGBM + CatBoost (v11)")
print("="*65)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_bins)):
    t0 = time.time()
    print(f"\n>> Fold {fold+1}/{N_FOLDS}")

    tr_rows = train_df.iloc[tr_idx].copy()
    va_rows = train_df.iloc[va_idx].copy()

    # KFold-safe target encoding computed strictly on tr_rows to prevent leakage
    target_encoded_cols = ["district", "grid_id", "downstream_sig", "infra_deficit_sig"]
    
    for col in target_encoded_cols:
        enc_name = f"{col}_target_enc"
        mapping = tr_rows.groupby(col)[TARGET].mean()
        
        tr_rows[enc_name] = tr_rows[col].astype(str).map(mapping).fillna(GLOBAL_MEAN).astype(float)
        va_rows[enc_name] = va_rows[col].astype(str).map(mapping).fillna(GLOBAL_MEAN).astype(float)
        test_df[enc_name] = test_df[col].astype(str).map(mapping.to_dict()).fillna(GLOBAL_MEAN).astype(float)

    # Reconstruct final feature list to include target encodings
    FEATURES = BASE_FEATURES + [f"{c}_target_enc" for c in target_encoded_cols]

    y_tr = tr_rows[TARGET]
    y_va = va_rows[TARGET]

    X_tr = tr_rows[FEATURES].copy()
    X_va = va_rows[FEATURES].copy()
    X_te = test_df[FEATURES].copy()
    
    # Re-apply category categories to prevent slicing codes mismatch
    for col in cat_feature_names:
        if col in FEATURES:
            cdt = cat_dtype_map[col]
            X_tr[col] = X_tr[col].astype(str).astype(cdt)
            X_va[col] = X_va[col].astype(str).astype(cdt)
            X_te[col] = X_te[col].astype(str).astype(cdt)

    # Format converters
    X_tr_xgb = to_xgb_fmt(X_tr);  X_va_xgb = to_xgb_fmt(X_va);  X_te_xgb = to_xgb_fmt(X_te)
    X_tr_cat = to_cat_fmt(X_tr);   X_va_cat = to_cat_fmt(X_va);   X_te_cat = to_cat_fmt(X_te)

    # XGBoost Regressor
    xgb_model = xgb.XGBRegressor(
        n_estimators          = 3000,
        learning_rate         = 0.05,
        max_depth             = 7,
        min_child_weight      = 3,
        subsample             = 0.8,
        colsample_bytree      = 0.75,
        colsample_bylevel     = 0.75,
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        gamma                 = 0.05,
        tree_method           = "hist",
        enable_categorical    = False,
        early_stopping_rounds = 100,
        random_state          = 42,
        n_jobs                = -1
    )
    xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
    oof_xgb[va_idx] = xgb_model.predict(X_va_xgb)
    tst_xgb        += xgb_model.predict(X_te_xgb) / N_FOLDS
    print(f"   [XGB] best_iter={xgb_model.best_iteration}")

    # LightGBM Regressor
    lgb_model = lgb.LGBMRegressor(
        n_estimators       = 3000,
        learning_rate      = 0.05,
        num_leaves         = 127,
        max_depth          = -1,
        min_child_samples  = 20,
        subsample          = 0.8,
        subsample_freq     = 1,
        colsample_bytree   = 0.75,
        reg_alpha          = 0.1,
        reg_lambda         = 1.0,
        random_state       = 42,
        n_jobs             = -1,
        verbosity          = -1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set  = [(X_va, y_va)],
        callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
    )
    oof_lgb[va_idx] = lgb_model.predict(X_va)
    tst_lgb        += lgb_model.predict(X_te) / N_FOLDS
    print(f"   [LGB] best_iter={lgb_model.best_iteration_}")

    # CatBoost Regressor
    cat_model = cb.CatBoostRegressor(
        iterations            = 3000,
        learning_rate         = 0.05,
        depth                 = 7,
        l2_leaf_reg           = 3,
        bagging_temperature   = 0.5,
        random_strength       = 1,
        border_count          = 128,
        loss_function         = "RMSE",
        eval_metric           = "RMSE",
        task_type             = "CPU",
        random_seed           = 42,
        verbose               = False
    )
    cat_model.fit(
        X_tr_cat, y_tr,
        cat_features          = cat_feature_names,
        eval_set              = (X_va_cat, y_va),
        early_stopping_rounds = 100,
        verbose               = False
    )
    oof_cat[va_idx] = cat_model.predict(X_va_cat)
    tst_cat        += cat_model.predict(X_te_cat) / N_FOLDS
    print(f"   [CAT] best_iter={cat_model.best_iteration_}")

    # Fold Ensemble scoring
    oof_ens_fold = (oof_xgb[va_idx] + oof_lgb[va_idx] + oof_cat[va_idx]) / 3.0
    y_va_arr = y_va.values
    f_mae  = mean_absolute_error(y_va_arr, oof_ens_fold)
    f_rmse = root_mean_squared_error(y_va_arr, oof_ens_fold)
    f_ev   = explained_variance_score(y_va_arr, oof_ens_fold)
    fold_results.append({"fold": fold+1, "MAE": f_mae, "RMSE": f_rmse, "EV": f_ev})
    print(f"   [ENS] MAE={f_mae:.4f}  RMSE={f_rmse:.4f}  EV={f_ev:.4f}  [{time.time()-t0:.0f}s]")

# -----------------------------------------------------------------
# 7. ENSEMBLE COMBINATION & EXPORT
# -----------------------------------------------------------------
y_arr    = y.values
rmse_xgb = root_mean_squared_error(y_arr, oof_xgb)
rmse_lgb = root_mean_squared_error(y_arr, oof_lgb)
rmse_cat = root_mean_squared_error(y_arr, oof_cat)
w_xgb = 1.0 / rmse_xgb;  w_lgb = 1.0 / rmse_lgb;  w_cat = 1.0 / rmse_cat
total_w = w_xgb + w_lgb + w_cat

print(f"\n[WGHT] Model weights (inverse-RMSE):")
print(f"   XGB : {w_xgb/total_w:.3f}  (OOF RMSE={rmse_xgb:.5f})")
print(f"   LGB : {w_lgb/total_w:.3f}  (OOF RMSE={rmse_lgb:.5f})")
print(f"   CAT : {w_cat/total_w:.3f}  (OOF RMSE={rmse_cat:.5f})")

oof_ensemble = (w_xgb*oof_xgb + w_lgb*oof_lgb + w_cat*oof_cat) / total_w
tst_ensemble = (w_xgb*tst_xgb + w_lgb*tst_lgb + w_cat*tst_cat) / total_w

g_mae  = mean_absolute_error(y_arr, oof_ensemble)
g_rmse = root_mean_squared_error(y_arr, oof_ensemble)
g_ev   = explained_variance_score(y_arr, oof_ensemble)

print("\n" + "="*65)
print("  GLOBAL OOF RESULTS (Raw Ensemble v11)")
print("="*65)
print(f"    MAE            : {g_mae:.5f}")
print(f"    RMSE           : {g_rmse:.5f}")
print(f"    Explained Var. : {g_ev:.5f}")
print("="*65)

# Export OOF report
pd.DataFrame(fold_results).to_csv("submissions/fold_report_v11.csv", index=False)
print("\n[DONE] Saved fold_report_v11.csv")

# Final predictions exported with boundary preservation
tst_final = np.clip(tst_ensemble, 0.0, 1.0)
submission = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": tst_final
})
submission.to_csv("submissions/submission_v11.csv", index=False)
print(f"[DONE] Saved submission_v11.csv ({len(submission)} rows)")
print(f"       Pred range : [{tst_final.min():.4f}, {tst_final.max():.4f}]")
