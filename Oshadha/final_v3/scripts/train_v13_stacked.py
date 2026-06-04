"""
ML Opsidian: Genesis - Pruned + Stacked Pipeline v13
=====================================================
Strategy: Combine all three insights from v12 post-mortem:

  1. PRUNE & FOCUS: Revert to v11-proven hyperparameters (depth=7, LR=0.05,
     no sample weighting). Remove noise features via automatic importance
     thresholding after a quick pre-selection pass.

  2. STACKING: Add a second-level Ridge meta-learner that learns optimal
     blending weights from the 3 base models' OOF predictions. This replaces
     naive inverse-RMSE averaging with a learned combination.

  3. TARGET ENCODING EXPANSION: Add smoothed Bayesian target encodings for
     landcover, soil_type, water_supply, electricity, road_quality (beyond
     just district/grid/downstream/infra_deficit).

  4. INUNDATION RATIO: Keep the one v12 feature with solid physical grounding
     (normalized by landcover class mean). Drop the 4 noise features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
import time

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# 1. LOAD & DEDUPLICATE
# -----------------------------------------------------------------
print("=" * 70)
print("  ML OPSIDIAN v13 - PRUNED + STACKED PIPELINE")
print("=" * 70)
print("\n[LOAD] Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")
print(f"   Train shape      : {train_df.shape}")
print(f"   Test shape       : {test_df.shape}")
train_df = train_df.drop_duplicates()
print(f"   Train after dedup: {train_df.shape}")

# -----------------------------------------------------------------
# 2. GEOSPATIAL HOT-DECK IMPUTATION (same as v11)
# -----------------------------------------------------------------
print("\n[IMPUTE] Starting Geospatial Hot-Deck Imputation...")

combined = pd.concat([
    train_df.drop(columns=['flood_risk_score'], errors='ignore'),
    test_df
], ignore_index=True)

print("   -> Creating coordinate lookup maps from place_name and district...")
coords_lookup = combined.groupby(['place_name', 'district'])[['latitude', 'longitude']].median().to_dict('index')

imputed_coords_count = 0
for df in [train_df, test_df]:
    mask = df['latitude'].isnull() & df['place_name'].notnull() & df['district'].notnull()
    for idx in df[mask].index:
        key = (df.loc[idx, 'place_name'], df.loc[idx, 'district'])
        if key in coords_lookup and not np.isnan(coords_lookup[key]['latitude']):
            df.loc[idx, 'latitude'] = coords_lookup[key]['latitude']
            df.loc[idx, 'longitude'] = coords_lookup[key]['longitude']
            imputed_coords_count += 1
print(f"   -> Imputed missing coordinates for {imputed_coords_count} rows.")

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
    print(f"      * Imputed {col}: {imputed_col_count} rows via spatial KNN.")

print("   -> District median fallback for remaining nulls...")
for col in ['elevation_m', 'distance_to_river_m', 'latitude', 'longitude']:
    for df in [train_df, test_df]:
        df[col] = df[col].fillna(df.groupby('district')[col].transform('median'))
        df[col] = df[col].fillna(train_df[col].median())

assert train_df[['latitude', 'longitude', 'elevation_m', 'distance_to_river_m']].isnull().sum().sum() == 0
assert test_df[['latitude', 'longitude', 'elevation_m', 'distance_to_river_m']].isnull().sum().sum() == 0

# -----------------------------------------------------------------
# 3. FEATURE ENGINEERING (v11 proven set + inundation ratio)
# -----------------------------------------------------------------
print("\n[FEAT] Engineering features (v11 proven set + inundation ratio)...")

combined_imputed = pd.concat([
    train_df.drop(columns=['flood_risk_score'], errors='ignore'),
    test_df
], ignore_index=True)
district_elev_std = combined_imputed.groupby('district')['elevation_m'].std().to_dict()
landcover_mean_inundation = combined_imputed.groupby('landcover')['inundation_area_sqm'].mean().to_dict()

soil_infilt_map = {'Sandy': 0.8, 'Loamy': 0.6, 'Silty': 0.4, 'Clay': 0.2, 'Peaty': 0.1}
cyclone_districts = {'Batticaloa', 'Trincomalee', 'Ampara', 'Mullaitivu', 'Jaffna'}
wet_zone_districts = {'Colombo', 'Gampaha', 'Kalutara', 'Galle', 'Matara', 'Ratnapura', 'Kegalle'}

def engineer_features(df):
    df = df.copy()
    
    # --- Downstream Signature (proven top-3 driver via target enc) ---
    df['downstream_sig'] = (
        df['flood_occurrence_current_event'].astype(str).str.strip() + "_" +
        df['is_good_to_live'].astype(str).str.strip() + "_" +
        df['reason_not_good_to_live'].astype(str).str.strip()
    )
    
    # --- Calendar month ---
    date_series = pd.to_datetime(df['generation_date'])
    df['month'] = date_series.dt.month
    
    # --- Monsoon Switch (proven: is_maha in top 12) ---
    df['is_yala'] = df['month'].isin([5, 6, 7, 8, 9]).astype(int)
    df['is_maha'] = df['month'].isin([11, 12, 1]).astype(int)
    df['zone_code'] = df['district'].astype(str).map(lambda x: 1 if x in wet_zone_districts else 2)
    df['monsoon_impact'] = df['rainfall_7d_mm'] * df['is_yala'] * (df['zone_code'] == 1).astype(int) + \
                           df['rainfall_7d_mm'] * df['is_maha'] * (df['zone_code'] == 2).astype(int)
                           
    # --- Urban Pluvial vs Rural Fluvial (proven: fluvial in top 4) ---
    df['urban_runoff_potential'] = df['rainfall_7d_mm'] * df['built_up_percent'] * (1.0 / (df['drainage_index'] + 1e-5))
    df['fluvial_risk_score_feat'] = df['rainfall_7d_mm'] * (1.0 / (df['distance_to_river_m'] + 1.0))
    
    # --- Soil saturation physics ---
    df['soil_infiltration'] = df['soil_type'].astype(str).map(soil_infilt_map).fillna(0.4)
    df['soil_saturation_limit'] = df['rainfall_7d_mm'] / (df['soil_infiltration'] + 0.1)
    
    # --- TWI & Flatness (proven: flatness in top 11) ---
    df['pseudo_twi'] = np.log1p((df['distance_to_river_m'] + 1.0) / (df['elevation_m'].clip(lower=0.0) + 1.0))
    df['flatness_index'] = df['district'].astype(str).map(district_elev_std).fillna(df['elevation_m'].std())
    
    # --- Cyclone vulnerability (proven: in_cyclone_path in top 6) ---
    df['in_cyclone_path'] = df['district'].astype(str).map(lambda x: 1 if x in cyclone_districts else 0)
    df['cyclone_vulnerability'] = df['in_cyclone_path'] * df['extreme_weather_index']
    
    # --- Slope Proxy ---
    df['slope_proxy'] = df['elevation_m'] / (df['distance_to_river_m'] + 1.0)
    
    # --- Isolation indices (proven: in top 20 via vulnerability) ---
    df['isolation_index'] = np.log1p(df['nearest_hospital_km']) + np.log1p(df['nearest_evac_km'])
    df['vulnerability'] = df['isolation_index'] / (df['infrastructure_score'] + 1.0)
    
    # --- Elevation divergence (proven: in top 16) ---
    df['elevation_divergence'] = df['elevation_m'] - df['elevation_m_yeojohnson']
    
    # --- Infrastructure Deficit String ---
    df['infra_deficit_sig'] = (
        df['water_supply'].astype(str).str.strip() + "_" +
        df['electricity'].astype(str).str.strip() + "_" +
        df['road_quality'].astype(str).str.strip()
    )
    
    # --- v10 baseline interactions (all proven) ---
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
    
    # --- NEW v13: Inundation Deviational Ratio (solid physics) ---
    df['landcover_mean_inundation_val'] = df['landcover'].astype(str).map(landcover_mean_inundation).fillna(
        combined_imputed['inundation_area_sqm'].mean()
    )
    df['inundation_ratio'] = df['inundation_area_sqm'] / (df['landcover_mean_inundation_val'] + 1.0)
    
    # Spatial bins for target encoding
    lat = df["latitude"].fillna(df["latitude"].median())
    lon = df["longitude"].fillna(df["longitude"].median())
    df["lat_bin"] = (lat / 0.5).astype(int)
    df["lon_bin"] = (lon / 0.5).astype(int)
    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
    
    # Remove raw skewed & helper columns
    df = df.drop(columns=["inundation_area_sqm", "landcover_mean_inundation_val"])
    
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

# Expanded target encoding pool (Strategy 3)
TARGET_ENC_COLS = [
    "district", "grid_id", "downstream_sig", "infra_deficit_sig",
    "landcover", "soil_type", "water_supply", "electricity", "road_quality"
]

IGNORE_COLS = DROP_COLS + [TARGET, "flood_occurrence_yes"]
SPATIAL_HELPERS = ["lat_bin", "lon_bin", "grid_id"]

BASE_FEATURES = [c for c in train_df.columns
                 if c not in IGNORE_COLS and c not in SPATIAL_HELPERS]

print(f"\n[PREP] Casting dtypes...")
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

print(f"   Base Features : {len(BASE_FEATURES)}")

# -----------------------------------------------------------------
# 5. FEATURE IMPORTANCE PRE-SELECTION (Strategy 1: Prune)
# -----------------------------------------------------------------
print("\n[PRUNE] Running feature importance pre-selection pass...")
y_full = train_df[TARGET]
GLOBAL_MEAN = float(y_full.mean())

# Quick single-fold importance scan
np.random.seed(42)
idx_all = np.arange(len(train_df))
np.random.shuffle(idx_all)
split_pt = int(len(idx_all) * 0.8)
tr_scan_idx, va_scan_idx = idx_all[:split_pt], idx_all[split_pt:]

tr_scan = train_df.iloc[tr_scan_idx].copy()
va_scan = train_df.iloc[va_scan_idx].copy()

# Compute target encodings for the scan pass
for col in TARGET_ENC_COLS:
    enc_name = f"{col}_target_enc"
    mapping = tr_scan.groupby(col)[TARGET].mean()
    tr_scan[enc_name] = tr_scan[col].astype(str).map(mapping).fillna(GLOBAL_MEAN).astype(float)
    va_scan[enc_name] = va_scan[col].astype(str).map(mapping).fillna(GLOBAL_MEAN).astype(float)

SCAN_FEATURES = BASE_FEATURES + [f"{c}_target_enc" for c in TARGET_ENC_COLS]

X_scan_tr = tr_scan[SCAN_FEATURES].copy()
X_scan_va = va_scan[SCAN_FEATURES].copy()

# Convert categoricals for XGBoost scan
for col in X_scan_tr.columns:
    if hasattr(X_scan_tr[col], "cat"):
        X_scan_tr[col] = X_scan_tr[col].cat.codes.astype("int32")
        X_scan_va[col] = X_scan_va[col].cat.codes.astype("int32")

scan_model = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=7,
    min_child_weight=3, subsample=0.8, colsample_bytree=0.75,
    reg_lambda=1.0, tree_method="hist", enable_categorical=False,
    early_stopping_rounds=50, random_state=42, n_jobs=-1
)
scan_model.fit(X_scan_tr, tr_scan[TARGET], eval_set=[(X_scan_va, va_scan[TARGET])], verbose=False)

imp = pd.Series(scan_model.feature_importances_, index=SCAN_FEATURES)
imp_sorted = imp.sort_values(ascending=False)

# Keep top N features that capture meaningful importance
# Use cumulative importance threshold: keep features until 95% of total importance
cum_imp = imp_sorted.cumsum() / imp_sorted.sum()
n_keep = (cum_imp <= 0.95).sum() + 1  # +1 to include the one that crosses 95%
n_keep = max(n_keep, 30)  # Minimum 30 features
n_keep = min(n_keep, len(SCAN_FEATURES))

SELECTED_FEATURES = list(imp_sorted.head(n_keep).index)
PRUNED_FEATURES = list(imp_sorted.tail(len(SCAN_FEATURES) - n_keep).index)

print(f"   Scan model best_iter: {scan_model.best_iteration}")
print(f"   Total features scanned : {len(SCAN_FEATURES)}")
print(f"   Features KEPT (top 95%): {n_keep}")
print(f"   Features PRUNED        : {len(PRUNED_FEATURES)}")
print(f"\n   Top 25 selected features:")
for i, (feat, score) in enumerate(imp_sorted.head(25).items(), 1):
    marker = "[TE]" if "_target_enc" in feat else "[CAT]" if feat in CAT_FEATURES else "[NUM]"
    print(f"      {i:>2}. {marker} {feat:<45} {score:.4f}")
print(f"\n   Pruned features:")
for feat in PRUNED_FEATURES:
    print(f"      - {feat} ({imp[feat]:.5f})")

# -----------------------------------------------------------------
# 6. CROSS VALIDATION SETUP
# -----------------------------------------------------------------
N_FOLDS = 5
y = train_df[TARGET]
y_bins = pd.cut(y, bins=10, labels=False)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train_df))
oof_lgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
tst_xgb = np.zeros(len(test_df))
tst_lgb = np.zeros(len(test_df))
tst_cat = np.zeros(len(test_df))

fold_results = []
cat_feature_names = [c for c in CAT_FEATURES if c in SELECTED_FEATURES]

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
# 7. TRAINING LOOP (v11-proven hyperparameters + expanded target enc)
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("  5-FOLD STRATIFIED CV -- XGBoost + LightGBM + CatBoost (v13)")
print("  STRATEGY: Pruned features + Expanded TE + Stacking")
print(f"  FEATURES: {len(SELECTED_FEATURES)} selected (pruned {len(PRUNED_FEATURES)})")
print("=" * 70)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_bins)):
    t0 = time.time()
    print(f"\n>> Fold {fold+1}/{N_FOLDS}")

    tr_rows = train_df.iloc[tr_idx].copy()
    va_rows = train_df.iloc[va_idx].copy()

    # KFold-safe Bayesian-smoothed target encoding (expanded set)
    SMOOTHING = 10  # Bayesian smoothing strength
    for col in TARGET_ENC_COLS:
        enc_name = f"{col}_target_enc"
        if enc_name not in SELECTED_FEATURES:
            continue  # Skip if this TE was pruned
        
        group_stats = tr_rows.groupby(col)[TARGET].agg(['mean', 'count'])
        # Bayesian smoothing: blend group mean with global mean weighted by count
        smoothed = (group_stats['count'] * group_stats['mean'] + SMOOTHING * GLOBAL_MEAN) / (group_stats['count'] + SMOOTHING)
        
        tr_rows[enc_name] = tr_rows[col].astype(str).map(smoothed).fillna(GLOBAL_MEAN).astype(float)
        va_rows[enc_name] = va_rows[col].astype(str).map(smoothed).fillna(GLOBAL_MEAN).astype(float)
        test_df[enc_name] = test_df[col].astype(str).map(smoothed.to_dict()).fillna(GLOBAL_MEAN).astype(float)

    # Use only SELECTED features (pruned)
    FEATURES = [f for f in SELECTED_FEATURES if f in tr_rows.columns]

    y_tr = tr_rows[TARGET]
    y_va = va_rows[TARGET]
    X_tr = tr_rows[FEATURES].copy()
    X_va = va_rows[FEATURES].copy()
    X_te = test_df[FEATURES].copy()
    
    # Re-apply category dtype for fold slicing safety
    for col in cat_feature_names:
        if col in FEATURES and col in cat_dtype_map:
            cdt = cat_dtype_map[col]
            X_tr[col] = X_tr[col].astype(str).astype(cdt)
            X_va[col] = X_va[col].astype(str).astype(cdt)
            X_te[col] = X_te[col].astype(str).astype(cdt)

    X_tr_xgb = to_xgb_fmt(X_tr); X_va_xgb = to_xgb_fmt(X_va); X_te_xgb = to_xgb_fmt(X_te)
    X_tr_cat = to_cat_fmt(X_tr);  X_va_cat = to_cat_fmt(X_va);  X_te_cat = to_cat_fmt(X_te)

    # XGBoost (v11-proven hyperparameters)
    xgb_model = xgb.XGBRegressor(
        n_estimators=3000, learning_rate=0.05, max_depth=7,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.75,
        colsample_bylevel=0.75, reg_alpha=0.1, reg_lambda=1.0,
        gamma=0.05, tree_method="hist", enable_categorical=False,
        early_stopping_rounds=100, random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
    oof_xgb[va_idx] = xgb_model.predict(X_va_xgb)
    tst_xgb += xgb_model.predict(X_te_xgb) / N_FOLDS
    print(f"   [XGB] best_iter={xgb_model.best_iteration}")

    # LightGBM (v11-proven hyperparameters)
    lgb_model = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=0.05, num_leaves=127,
        max_depth=-1, min_child_samples=20, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.75, reg_alpha=0.1,
        reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1
    )
    lgb_model.fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
    )
    oof_lgb[va_idx] = lgb_model.predict(X_va)
    tst_lgb += lgb_model.predict(X_te) / N_FOLDS
    print(f"   [LGB] best_iter={lgb_model.best_iteration_}")

    # CatBoost (v11-proven hyperparameters)
    cat_model = cb.CatBoostRegressor(
        iterations=3000, learning_rate=0.05, depth=7,
        l2_leaf_reg=3, bagging_temperature=0.5, random_strength=1,
        border_count=128, loss_function="RMSE", eval_metric="RMSE",
        task_type="CPU", random_seed=42, verbose=False
    )
    cat_model.fit(
        X_tr_cat, y_tr, cat_features=cat_feature_names,
        eval_set=(X_va_cat, y_va), early_stopping_rounds=100, verbose=False
    )
    oof_cat[va_idx] = cat_model.predict(X_va_cat)
    tst_cat += cat_model.predict(X_te_cat) / N_FOLDS
    print(f"   [CAT] best_iter={cat_model.best_iteration_}")

    # Fold metrics (simple average for reporting)
    oof_ens_fold = (oof_xgb[va_idx] + oof_lgb[va_idx] + oof_cat[va_idx]) / 3.0
    y_va_arr = y_va.values
    f_mae  = mean_absolute_error(y_va_arr, oof_ens_fold)
    f_rmse = root_mean_squared_error(y_va_arr, oof_ens_fold)
    f_ev   = explained_variance_score(y_va_arr, oof_ens_fold)
    fold_results.append({"fold": fold+1, "MAE": f_mae, "RMSE": f_rmse, "EV": f_ev})
    print(f"   [ENS] MAE={f_mae:.4f}  RMSE={f_rmse:.4f}  EV={f_ev:.4f}  [{time.time()-t0:.0f}s]")

# -----------------------------------------------------------------
# 8. LEVEL-2 STACKING (Strategy 2: Ridge Meta-Learner)
# -----------------------------------------------------------------
print("\n" + "-" * 70)
print("  LEVEL-2 STACKING: Ridge Meta-Learner")
print("-" * 70)

y_arr = y.values

# Assemble OOF meta-features
oof_meta = np.column_stack([oof_xgb, oof_lgb, oof_cat])
tst_meta = np.column_stack([tst_xgb, tst_lgb, tst_cat])

# Train Ridge with cross-validation to find optimal blend
# Use same folds for consistency
oof_stacked = np.zeros(len(train_df))
tst_stacked_accum = np.zeros(len(test_df))

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_bins)):
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(oof_meta[tr_idx], y_arr[tr_idx])
    oof_stacked[va_idx] = ridge.predict(oof_meta[va_idx])
    tst_stacked_accum += ridge.predict(tst_meta) / N_FOLDS

# Final Ridge on all OOF data for test predictions
ridge_final = Ridge(alpha=1.0, fit_intercept=True)
ridge_final.fit(oof_meta, y_arr)
tst_stacked = ridge_final.predict(tst_meta)

print(f"   Ridge coefficients: XGB={ridge_final.coef_[0]:.4f}, LGB={ridge_final.coef_[1]:.4f}, CAT={ridge_final.coef_[2]:.4f}")
print(f"   Ridge intercept   : {ridge_final.intercept_:.4f}")

# Compare: simple average vs inverse-RMSE vs stacked
rmse_xgb = root_mean_squared_error(y_arr, oof_xgb)
rmse_lgb = root_mean_squared_error(y_arr, oof_lgb)
rmse_cat = root_mean_squared_error(y_arr, oof_cat)

# Method 1: Simple average
oof_avg = (oof_xgb + oof_lgb + oof_cat) / 3.0
tst_avg = (tst_xgb + tst_lgb + tst_cat) / 3.0

# Method 2: Inverse-RMSE weighted
w_xgb = 1.0/rmse_xgb; w_lgb = 1.0/rmse_lgb; w_cat = 1.0/rmse_cat
total_w = w_xgb + w_lgb + w_cat
oof_invw = (w_xgb*oof_xgb + w_lgb*oof_lgb + w_cat*oof_cat) / total_w
tst_invw = (w_xgb*tst_xgb + w_lgb*tst_lgb + w_cat*tst_cat) / total_w

# Method 3: Stacked (Ridge)
# oof_stacked already computed

print(f"\n   [BLEND COMPARISON]")
methods = {
    "Simple Average":  (oof_avg,     tst_avg),
    "Inverse-RMSE":    (oof_invw,    tst_invw),
    "Ridge Stacked":   (oof_stacked, tst_stacked),
    "Ridge Stacked CV":(oof_stacked, tst_stacked_accum),
}

best_method = None
best_ev = -999
for name, (oof_pred, tst_pred) in methods.items():
    m_mae  = mean_absolute_error(y_arr, oof_pred)
    m_rmse = root_mean_squared_error(y_arr, oof_pred)
    m_ev   = explained_variance_score(y_arr, oof_pred)
    winner = ""
    if m_ev > best_ev:
        best_ev = m_ev
        best_method = name
        best_oof = oof_pred
        best_tst = tst_pred
        winner = " <-- BEST"
    print(f"      {name:>20}: MAE={m_mae:.5f}  RMSE={m_rmse:.5f}  EV={m_ev:.5f}{winner}")

print(f"\n   [SELECTED] {best_method}")

# -----------------------------------------------------------------
# 9. FINAL RESULTS & EXPORT
# -----------------------------------------------------------------
g_mae  = mean_absolute_error(y_arr, best_oof)
g_rmse = root_mean_squared_error(y_arr, best_oof)
g_ev   = explained_variance_score(y_arr, best_oof)

print("\n" + "=" * 70)
print("  GLOBAL OOF RESULTS (v13 - Pruned + Stacked)")
print("=" * 70)
print(f"    MAE            : {g_mae:.5f}")
print(f"    RMSE           : {g_rmse:.5f}")
print(f"    Explained Var. : {g_ev:.5f}")
print(f"    Pred Range     : [{best_oof.min():.4f}, {best_oof.max():.4f}]")
print("=" * 70)

# Per-model stats
print(f"\n[MODEL] Individual OOF Performance:")
print(f"   XGB : RMSE={rmse_xgb:.5f}  EV={explained_variance_score(y_arr, oof_xgb):.5f}")
print(f"   LGB : RMSE={rmse_lgb:.5f}  EV={explained_variance_score(y_arr, oof_lgb):.5f}")
print(f"   CAT : RMSE={rmse_cat:.5f}  EV={explained_variance_score(y_arr, oof_cat):.5f}")

# Feature importance (top 20 from last XGBoost fold)
print(f"\n[FEAT IMPORTANCE] Top 20 Features (XGBoost, last fold):")
imp_final = pd.Series(xgb_model.feature_importances_, index=FEATURES)
for rank, (feat, score) in enumerate(imp_final.sort_values(ascending=False).head(20).items(), 1):
    print(f"   {rank:>2}. {feat:<45} {score:.4f}")

# Export fold report
fold_report = pd.DataFrame(fold_results)
fold_report.to_csv("submissions/fold_report_v13.csv", index=False)
print(f"\n[DONE] Saved fold_report_v13.csv")
print(fold_report.to_string(index=False))

# Comparison
print(f"\n[COMPARE] v11 Baseline : MAE=0.17984, RMSE=0.23539, EV=0.02737")
print(f"[COMPARE] v12 Regress  : MAE=0.18022, RMSE=0.23550, EV=0.02664")
print(f"[COMPARE] v13 Current  : MAE={g_mae:.5f}, RMSE={g_rmse:.5f}, EV={g_ev:.5f}")
ev_delta = g_ev - 0.02737
rmse_delta = g_rmse - 0.23539
print(f"[COMPARE] v13 vs v11 EV   : {ev_delta:+.5f} ({'IMPROVED' if ev_delta > 0 else 'REGRESSED'})")
print(f"[COMPARE] v13 vs v11 RMSE : {rmse_delta:+.5f} ({'IMPROVED' if rmse_delta < 0 else 'REGRESSED'})")

# Final submission with boundary preservation
tst_final = np.clip(best_tst, 0.0, 1.0)
submission = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": tst_final
})
submission.to_csv("submissions/submission_v13.csv", index=False)
print(f"\n[DONE] Saved submission_v13.csv ({len(submission)} rows)")

np.save("submissions/oof_v13.npy", best_oof)
print(f"       Pred range : [{tst_final.min():.4f}, {tst_final.max():.4f}]")
