"""
ML Opsidian: Genesis v34 - Multi-Seed Stacking, MAE-Only, & 5x Sample Weighting
========================================================================
Features of v34:
1. MAE-Only Ensemble: Drop all RMSE models.
2. Target Encodings: Restored grid_id back to TARGET_ENC_COLS.
3. Pseudo-Labeling: Restored v30 pseudo-labeling window (0.46-0.49).
4. Level-2 Stacking: Per-seed Ridge stacking (positive=True) to preserve error structure.
5. Upgrade 3 (Weighted Calibration): Added 5x sample weight on real rows and 1x on synthetic rows
   during Level-1 model training.
6. Output Paths: Saves results to submissions/ folder for evaluate.py.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import warnings
import time
import os

DATA_DIR = "/kaggle/input/competitions/ml-opsidian-genesis-initial-round-26"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "data" # Fallback local

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------
USE_PSEUDO = True
PSEUDO_WINDOW_LOW = 0.46
PSEUDO_WINDOW_HIGH = 0.49

print("=" * 70)
print("  ML OPSIDIAN v34 - MULTI-SEED MAE-ONLY ENSEMBLE + 5X WEIGHTING")
print("=" * 70)

# -----------------------------------------------------------------
# 1. LOAD & DEDUPLICATE
# -----------------------------------------------------------------
print("\n[LOAD] Loading data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
train_df = train_df.drop_duplicates()
print(f"   Train: {train_df.shape}  Test: {test_df.shape}")

# -----------------------------------------------------------------
# 1.3. PRECISION FINGERPRINTS
# -----------------------------------------------------------------
print("\n[FEAT] Extracting coordinate decimal precision fingerprints...")
for df in [train_df, test_df]:
    df['lat_decimal_len'] = df['latitude'].apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) and str(x).lower() != 'nan' else 0)
    df['lon_decimal_len'] = df['longitude'].apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) and str(x).lower() != 'nan' else 0)

# -----------------------------------------------------------------
# 1.6. SEMI-SUPERVISED PSEUDO-LABELING
# -----------------------------------------------------------------
print(f"\n[SEMI-SUPERVISED] Pseudo-Labeling configuration: USE_PSEUDO={USE_PSEUDO}")
train_df['is_pseudo'] = 0
test_df['is_pseudo'] = 0

if USE_PSEUDO:
    pseudo_path = "submission_v30.csv"
    if not os.path.exists(pseudo_path):
        pseudo_path = "submissions/submission_v30.csv"
    
    if os.path.exists(pseudo_path):
        sub_blend = pd.read_csv(pseudo_path)
        test_pseudo = test_df.merge(sub_blend, on="record_id", how="left")
        
        # Filter highly confident predictions (around median 0.46 - 0.49)
        mask = (test_pseudo['flood_risk_score'] >= PSEUDO_WINDOW_LOW) & (test_pseudo['flood_risk_score'] <= PSEUDO_WINDOW_HIGH)
        pseudo_rows = test_pseudo[mask].copy()
        pseudo_rows['is_pseudo'] = 1
        
        print(f"   Added {len(pseudo_rows)} pseudo-labeled rows from test set.")
        train_df = pd.concat([train_df, pseudo_rows], ignore_index=True)
    else:
        print("   [WARNING] submission_v30.csv not found. Skipping pseudo-labeling.")
else:
    print("   Pseudo-labeling is disabled for this run.")

# -----------------------------------------------------------------
# 2. GEOSPATIAL IMPUTATION
# -----------------------------------------------------------------
print("\n[IMPUTE] Geospatial Hot-Deck Imputation...")
combined = pd.concat([
    train_df.drop(columns=['flood_risk_score'], errors='ignore'), test_df
], ignore_index=True)

coords_lookup = combined.groupby(['place_name', 'district'])[['latitude', 'longitude']].median().to_dict('index')
for df in [train_df, test_df]:
    mask = df['latitude'].isnull() & df['place_name'].notnull() & df['district'].notnull()
    for idx in df[mask].index:
        key = (df.loc[idx, 'place_name'], df.loc[idx, 'district'])
        if key in coords_lookup and not np.isnan(coords_lookup[key]['latitude']):
            df.loc[idx, 'latitude'] = coords_lookup[key]['latitude']
            df.loc[idx, 'longitude'] = coords_lookup[key]['longitude']

for col in ['elevation_m', 'distance_to_river_m']:
    donor_pool = combined.dropna(subset=['latitude', 'longitude', col])
    knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
    knn.fit(donor_pool[['latitude', 'longitude']], donor_pool[col])
    for df in [train_df, test_df]:
        mm = df[col].isnull() & df['latitude'].notnull() & df['longitude'].notnull()
        if mm.any():
            df.loc[mm, col] = knn.predict(df.loc[mm, ['latitude', 'longitude']])

for col in ['elevation_m', 'distance_to_river_m', 'latitude', 'longitude']:
    for df in [train_df, test_df]:
        df[col] = df[col].fillna(df.groupby('district')[col].transform('median'))
        df[col] = df[col].fillna(train_df[col].median())
print("   Done.")

# -----------------------------------------------------------------
# 3. FEATURE ENGINEERING
# -----------------------------------------------------------------
print("\n[FEAT] Engineering features...")
district_elev_std = combined.groupby('district')['elevation_m'].std().to_dict()
landcover_mean_inundation = combined.groupby('landcover')['inundation_area_sqm'].mean().to_dict()
soil_infilt_map = {'Sandy': 0.8, 'Loamy': 0.6, 'Silty': 0.4, 'Clay': 0.2, 'Peaty': 0.1}
cyclone_districts = {'Batticaloa', 'Trincomalee', 'Ampara', 'Mullaitivu', 'Jaffna'}
wet_zone_districts = {'Colombo', 'Gampaha', 'Kalutara', 'Galle', 'Matara', 'Ratnapura', 'Kegalle'}

def engineer_features(df):
    df = df.copy()
    
    df['downstream_sig'] = (
        df['flood_occurrence_current_event'].astype(str).str.strip() + "_" +
        df['is_good_to_live'].astype(str).str.strip() + "_" +
        df['reason_not_good_to_live'].astype(str).str.strip()
    )
    date_series = pd.to_datetime(df['generation_date'])
    df['month'] = date_series.dt.month
    df['is_yala'] = df['month'].isin([5, 6, 7, 8, 9]).astype(int)
    df['is_maha'] = df['month'].isin([11, 12, 1]).astype(int)
    df['zone_code'] = df['district'].astype(str).map(lambda x: 1 if x in wet_zone_districts else 2)
    df['monsoon_impact'] = df['rainfall_7d_mm'] * df['is_yala'] * (df['zone_code'] == 1).astype(int) + \
                           df['rainfall_7d_mm'] * df['is_maha'] * (df['zone_code'] == 2).astype(int)
    df['urban_runoff_potential'] = df['rainfall_7d_mm'] * df['built_up_percent'] * (1.0 / (df['drainage_index'] + 1e-5))
    df['fluvial_risk_score_feat'] = df['rainfall_7d_mm'] * (1.0 / (df['distance_to_river_m'] + 1.0))
    df['soil_infiltration'] = df['soil_type'].astype(str).map(soil_infilt_map).fillna(0.4)
    df['soil_saturation_limit'] = df['rainfall_7d_mm'] / (df['soil_infiltration'] + 0.1)
    df['pseudo_twi'] = np.log1p((df['distance_to_river_m'] + 1.0) / (df['elevation_m'].clip(lower=0.0) + 1.0))
    df['flatness_index'] = df['district'].astype(str).map(district_elev_std).fillna(df['elevation_m'].std())
    df['in_cyclone_path'] = df['district'].astype(str).map(lambda x: 1 if x in cyclone_districts else 0)
    df['cyclone_vulnerability'] = df['in_cyclone_path'] * df['extreme_weather_index']
    df['slope_proxy'] = df['elevation_m'] / (df['distance_to_river_m'] + 1.0)
    df['isolation_index'] = np.log1p(df['nearest_hospital_km']) + np.log1p(df['nearest_evac_km'])
    df['vulnerability'] = df['isolation_index'] / (df['infrastructure_score'] + 1.0)
    df['elevation_divergence'] = df['elevation_m'] - df['elevation_m_yeojohnson']
    df['infra_deficit_sig'] = (
        df['water_supply'].astype(str).str.strip() + "_" +
        df['electricity'].astype(str).str.strip() + "_" +
        df['road_quality'].astype(str).str.strip()
    )
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
    df['landcover_mean_inundation_val'] = df['landcover'].astype(str).map(landcover_mean_inundation).fillna(
        combined['inundation_area_sqm'].mean()
    )
    df['inundation_ratio'] = df['inundation_area_sqm'] / (df['landcover_mean_inundation_val'] + 1.0)
    
    # 2D Grid Helper
    lat = df["latitude"].fillna(df["latitude"].median())
    lon = df["longitude"].fillna(df["longitude"].median())
    df["lat_bin"] = (lat / 0.5).astype(int)
    df["lon_bin"] = (lon / 0.5).astype(int)
    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
    
    df = df.drop(columns=["inundation_area_sqm", "landcover_mean_inundation_val"])
    return df

train_df = engineer_features(train_df)
test_df  = engineer_features(test_df)

# -----------------------------------------------------------------
# 4. PREP & DTYPES
# -----------------------------------------------------------------
TARGET    = "flood_risk_score"
ID_COL    = "record_id"
DROP_COLS = [ID_COL, "place_name", "is_synthetic", "generation_date", "is_pseudo"]
CAT_FEATURES = [
    "district", "landcover", "soil_type", "water_supply",
    "electricity", "road_quality", "urban_rural",
    "water_presence_flag", "flood_occurrence_current_event",
    "is_good_to_live", "reason_not_good_to_live"
]

TARGET_ENC_COLS = [
    "district", "grid_id", "downstream_sig", "infra_deficit_sig",
    "landcover", "soil_type", "water_supply", "electricity", "road_quality"
]

COMPOSITE_ENC_COLS = []
STD_ENC_COLS = [
    "district", "downstream_sig"
]

IGNORE_COLS = DROP_COLS + [
    TARGET, "flood_occurrence_yes", "downstream_sig", "infra_deficit_sig",
    "lat_decimal_len", "lon_decimal_len"
]
SPATIAL_HELPERS = ["lat_bin", "lon_bin", "grid_id"]
BASE_FEATURES = [c for c in train_df.columns if c not in IGNORE_COLS and c not in SPATIAL_HELPERS]

print("\n[PREP] Casting dtypes...")
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

cat_feature_names = [c for c in CAT_FEATURES if c in BASE_FEATURES]
print(f"   Base features: {len(BASE_FEATURES)}")

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
# 5. CROSS-VALIDATION SETUP & LEVEL-1 MODEL TRAINING
# -----------------------------------------------------------------
MODEL_NAMES = [
    "XGB-MAE (d7)",
    "CAT-MAE-1 (d5)",
    "CAT-MAE-2 (d5)",
    "LGB-MAE (d5)"
]

N_FOLDS     = 5
y           = train_df[TARGET]
GLOBAL_MEAN = float(y[train_df['is_pseudo'] == 0].mean())
GLOBAL_STD  = float(y[train_df['is_pseudo'] == 0].std())
GLOBAL_Q25  = float(y[train_df['is_pseudo'] == 0].quantile(0.25))
GLOBAL_Q75  = float(y[train_df['is_pseudo'] == 0].quantile(0.75))
GLOBAL_MEDIAN = float(y[train_df['is_pseudo'] == 0].median())

# GroupKFold on grid_id (spatial partitioning to block coordinate leaks)
gkf = GroupKFold(n_splits=N_FOLDS)
groups = train_df['grid_id'].values

SMOOTHING   = 10
SMOOTHING_COMPOSITE = 15
y_arr = y.values
real_mask = train_df['is_pseudo'] == 0

# Multi-seed with per-seed Ridge stacking (seeds 42, 123, 456, 789)
SEEDS = [42, 123, 456, 789]

original_y = y_arr[real_mask]
original_df = train_df[real_mask].reset_index(drop=True)
original_groups = original_df['grid_id'].values
gkf_l2 = GroupKFold(n_splits=N_FOLDS)

all_oof_stacked = np.zeros(len(original_y))
all_tst_stacked = []

print("\n" + "=" * 70)
print(f"  5-FOLD SPATIAL GROUP CV - MULTI-SEED MAE-ONLY ENSEMBLE + 5X WEIGHTING (v34)")
print("=" * 70)

t_start_global = time.time()

for seed in SEEDS:
    print(f"\n==================== RUNNING SEED {seed} ====================")
    oof_preds = {m: np.zeros(len(train_df)) for m in MODEL_NAMES}
    tst_preds = {m: np.zeros(len(test_df))  for m in MODEL_NAMES}
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, groups)):
        t0 = time.time()
        
        # Exclude pseudo rows from validation split
        va_is_pseudo = train_df.iloc[va_idx]['is_pseudo'] == 1
        va_idx_clean = va_idx[~va_is_pseudo] if va_is_pseudo.any() else va_idx
        
        tr_rows = train_df.iloc[tr_idx].copy()
        va_rows = train_df.iloc[va_idx_clean].copy()

        # Target encodings (strictly mapping statistics from REAL training rows only)
        real_tr_rows = tr_rows[tr_rows['is_pseudo'] == 0]
        all_te_cols = TARGET_ENC_COLS + COMPOSITE_ENC_COLS
        for col in all_te_cols:
            group_stats = real_tr_rows.groupby(col)[TARGET].agg(
                median='median', count='count', mean='mean', std='std',
                q25=lambda x: x.quantile(0.25), 
                q75=lambda x: x.quantile(0.75)
            )
            group_stats['std'] = group_stats['std'].fillna(0.0)
            
            s = SMOOTHING_COMPOSITE if col in COMPOSITE_ENC_COLS else SMOOTHING
            
            smoothed_median = (group_stats['count'] * group_stats['median'] + s * GLOBAL_MEDIAN) / (group_stats['count'] + s)
            smoothed_mean = (group_stats['count'] * group_stats['mean'] + s * GLOBAL_MEAN) / (group_stats['count'] + s)
            smoothed_std  = (group_stats['count'] * group_stats['std'] + s * GLOBAL_STD) / (group_stats['count'] + s)
            smoothed_q25  = (group_stats['count'] * group_stats['q25'] + s * GLOBAL_Q25) / (group_stats['count'] + s)
            smoothed_q75  = (group_stats['count'] * group_stats['q75'] + s * GLOBAL_Q75) / (group_stats['count'] + s)
            log_count = np.log1p(group_stats['count'])
            
            for tgt_df in [tr_rows, va_rows, test_df]:
                tgt_df[f"{col}_target_enc"] = tgt_df[col].astype(str).map(smoothed_median).fillna(GLOBAL_MEDIAN).astype(float)
                if col not in COMPOSITE_ENC_COLS:
                    tgt_df[f"{col}_target_q25"] = tgt_df[col].astype(str).map(smoothed_q25).fillna(GLOBAL_Q25).astype(float)
                    tgt_df[f"{col}_target_q75"] = tgt_df[col].astype(str).map(smoothed_q75).fillna(GLOBAL_Q75).astype(float)
                    tgt_df[f"{col}_target_cnt"] = tgt_df[col].astype(str).map(log_count).fillna(0.0).astype(float)
                
                if col in STD_ENC_COLS:
                    tgt_df[f"{col}_target_std"] = tgt_df[col].astype(str).map(smoothed_std).fillna(GLOBAL_STD).astype(float)

        te_features = []
        for col in all_te_cols:
            te_features.append(f"{col}_target_enc")
            if col not in COMPOSITE_ENC_COLS:
                te_features.extend([f"{col}_target_q25", f"{col}_target_q75", f"{col}_target_cnt"])
            if col in STD_ENC_COLS:
                te_features.append(f"{col}_target_std")
                
        FEATURES = BASE_FEATURES + te_features

        y_tr, y_va = tr_rows[TARGET], va_rows[TARGET]
        X_tr, X_va, X_te = tr_rows[FEATURES].copy(), va_rows[FEATURES].copy(), test_df[FEATURES].copy()

        for col in cat_feature_names:
            if col in FEATURES and col in cat_dtype_map:
                cdt = cat_dtype_map[col]
                X_tr[col] = X_tr[col].astype(str).astype(cdt)
                X_va[col] = X_va[col].astype(str).astype(cdt)
                X_te[col] = X_te[col].astype(str).astype(cdt)

        X_tr_xgb, X_va_xgb, X_te_xgb = to_xgb_fmt(X_tr), to_xgb_fmt(X_va), to_xgb_fmt(X_te)
        X_tr_cat, X_va_cat, X_te_cat  = to_cat_fmt(X_tr), to_cat_fmt(X_va), to_cat_fmt(X_te)
        
        # Upgrade 3: 5x Sample Weighting on real rows
        real_row_mask_tr = tr_rows['is_synthetic'].isna() & (tr_rows['is_pseudo'] == 0)
        sample_weights_tr = np.where(real_row_mask_tr, 5.0, 1.0)
        
        cat_pool_tr = cb.Pool(X_tr_cat, y_tr, cat_features=cat_feature_names, weight=sample_weights_tr)
        cat_pool_va = cb.Pool(X_va_cat, y_va, cat_features=cat_feature_names)

        # === 1. XGBoost with MAE objective ===
        xgb_mae = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.05, max_depth=7,
            objective='reg:absoluteerror', 
            min_child_weight=3, subsample=0.8, colsample_bytree=0.75,
            tree_method="hist", early_stopping_rounds=100, random_state=seed, n_jobs=-1,
            eval_metric='mae'
        )
        xgb_mae.fit(X_tr_xgb, y_tr, sample_weight=sample_weights_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
        oof_preds["XGB-MAE (d7)"][va_idx_clean] = xgb_mae.predict(X_va_xgb)
        tst_preds["XGB-MAE (d7)"] += xgb_mae.predict(X_te_xgb) / N_FOLDS

        # === 2. CatBoost-MAE-1 ===
        cat_mae1 = cb.CatBoostRegressor(
            iterations=5000, learning_rate=0.03, depth=5,
            l2_leaf_reg=5, loss_function="MAE", eval_metric="MAE",
            max_ctr_complexity=2,
            random_seed=seed, verbose=False
        )
        cat_mae1.fit(cat_pool_tr, eval_set=cat_pool_va, early_stopping_rounds=100, verbose=False)
        oof_preds["CAT-MAE-1 (d5)"][va_idx_clean] = cat_mae1.predict(X_va_cat)
        tst_preds["CAT-MAE-1 (d5)"] += cat_mae1.predict(X_te_cat) / N_FOLDS

        # === 3. CatBoost-MAE-2 ===
        cat_mae2 = cb.CatBoostRegressor(
            iterations=5000, learning_rate=0.05, depth=5,
            l2_leaf_reg=5, loss_function="MAE", eval_metric="MAE",
            max_ctr_complexity=2,
            random_seed=seed + 100, verbose=False
        )
        cat_mae2.fit(cat_pool_tr, eval_set=cat_pool_va, early_stopping_rounds=100, verbose=False)
        oof_preds["CAT-MAE-2 (d5)"][va_idx_clean] = cat_mae2.predict(X_va_cat)
        tst_preds["CAT-MAE-2 (d5)"] += cat_mae2.predict(X_te_cat) / N_FOLDS

        # === 4. LightGBM with MAE loss ===
        lgb_mae = lgb.LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            objective='regression_l1',
            random_state=seed,
            n_jobs=-1,
            verbosity=-1
        )
        lgb_mae.fit(
            X_tr, y_tr,
            sample_weight=sample_weights_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        oof_preds["LGB-MAE (d5)"][va_idx_clean] = lgb_mae.predict(X_va)
        tst_preds["LGB-MAE (d5)"] += lgb_mae.predict(X_te) / N_FOLDS

        oof_avg_fold = np.mean([oof_preds[m][va_idx_clean] for m in MODEL_NAMES], axis=0)
        y_va_arr = y_va.values
        f_mae  = mean_absolute_error(y_va_arr, oof_avg_fold)
        f_rmse = root_mean_squared_error(y_va_arr, oof_avg_fold)
        f_ev   = explained_variance_score(y_va_arr, oof_avg_fold)
        print(f"      Fold {fold+1}/{N_FOLDS} | XGB_M_it={xgb_mae.best_iteration:<4} CAT_M1_it={cat_mae1.best_iteration_:<4} CAT_M2_it={cat_mae2.best_iteration_:<4} LGB_M_it={lgb_mae.best_iteration_:<4} | [ENS MAE={f_mae:.4f}] [{time.time() - t0:.0f}s]")

    # Stacking Setup per Seed
    oof_meta_seed = np.column_stack([oof_preds[m][real_mask] for m in MODEL_NAMES])
    tst_meta_seed = np.column_stack([tst_preds[m] for m in MODEL_NAMES])
    
    # 1. Stacker Alpha Grid Search per Seed
    best_alpha, best_score = 1.0, np.inf
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        ridge_cv = Ridge(alpha=alpha, fit_intercept=True, positive=True)
        scores = cross_val_score(
            ridge_cv, oof_meta_seed, original_y,
            cv=GroupKFold(n_splits=5).split(original_df, original_y, original_groups),
            scoring='neg_mean_absolute_error'
        )
        if -scores.mean() < best_score:
            best_score = -scores.mean()
            best_alpha = alpha
            
    print(f"   [RIDGE SEED {seed}] Best alpha: {best_alpha}  (OOF MAE: {best_score:.5f})")
    
    # 2. Leak-Free Level-2 OOF stacking per seed (GroupKFold)
    oof_stacked_seed = np.zeros(len(original_y))
    for fold, (tr_idx, va_idx) in enumerate(gkf_l2.split(original_df, original_y, original_groups)):
        stacker_seed = Ridge(alpha=best_alpha, fit_intercept=True, positive=True)
        stacker_seed.fit(oof_meta_seed[tr_idx], original_y[tr_idx])
        oof_stacked_seed[va_idx] = stacker_seed.predict(oof_meta_seed[va_idx])
        
    all_oof_stacked += oof_stacked_seed / len(SEEDS)
    
    # 3. Fit final stacker on full meta-features of this seed to predict test
    final_stacker_seed = Ridge(alpha=best_alpha, fit_intercept=True, positive=True)
    final_stacker_seed.fit(oof_meta_seed, original_y)
    all_tst_stacked.append(final_stacker_seed.predict(tst_meta_seed))
    
    print("   [FINAL SEED COEFFICIENTS]")
    for i, name in enumerate(MODEL_NAMES):
        print(f"      {name:<18}: {final_stacker_seed.coef_[i]:.4f}")
    print(f"      {'Intercept':<18}: {final_stacker_seed.intercept_:.4f}")

# -----------------------------------------------------------------
# 7. GLOBAL ENSEMBLE RESULTS & METRICS
# -----------------------------------------------------------------
# Newly calibrated simulator coefficients (optimized over all 9 historical points)
c_mae, c_rmse, c_ev = 0.392696, 0.875527, 0.406963

g_mae  = mean_absolute_error(original_y, all_oof_stacked)
g_rmse = root_mean_squared_error(original_y, all_oof_stacked)
g_ev   = explained_variance_score(original_y, all_oof_stacked)
g_lb   = (c_mae * g_mae + c_rmse * g_rmse) * (1.0 + c_ev * (1.0 - g_ev))

# Organic real-only stats (is_synthetic is NaN)
real_organic_mask = original_df['is_synthetic'].isna().values
y_organic = original_y[real_organic_mask]
oof_organic = all_oof_stacked[real_organic_mask]
r_mae  = mean_absolute_error(y_organic, oof_organic)
r_rmse = root_mean_squared_error(y_organic, oof_organic)
r_ev   = explained_variance_score(y_organic, oof_organic)
r_lb   = (c_mae * r_mae + c_rmse * r_rmse) * (1.0 + c_ev * (1.0 - r_ev))

print("\n" + "=" * 70)
print("  GLOBAL OOF RESULTS (v34 - Weighted Multi-Seed Ensemble)")
print("=" * 70)
print(f"    [ALL ROWS]")
print(f"      MAE            : {g_mae:.5f}")
print(f"      RMSE           : {g_rmse:.5f}")
print(f"      Explained Var. : {g_ev:.5f}")
print(f"      Est. LB Score  : {g_lb:.5f}")
print(f"    [REAL ROWS ONLY]")
print(f"      MAE            : {r_mae:.5f}")
print(f"      RMSE           : {r_rmse:.5f}")
print(f"      Explained Var. : {r_ev:.5f}")
print(f"      Est. LB Score  : {r_lb:.5f}")
print(f"    Pred Range       : [{all_oof_stacked.min():.4f}, {all_oof_stacked.max():.4f}]")
print(f"    Total Time       : {time.time() - t_start_global:.1f}s")
print("=" * 70)

# Save Fold Report (aggregate results)
fold_results_all = []
for fold, (tr_idx, va_idx) in enumerate(gkf_l2.split(original_df, original_y, original_groups)):
    f_mae  = mean_absolute_error(original_y[va_idx], all_oof_stacked[va_idx])
    f_rmse = root_mean_squared_error(original_y[va_idx], all_oof_stacked[va_idx])
    f_ev   = explained_variance_score(original_y[va_idx], all_oof_stacked[va_idx])
    fold_results_all.append({"fold": fold+1, "MAE": f_mae, "RMSE": f_rmse, "EV": f_ev})

fold_report = pd.DataFrame(fold_results_all)
fold_report.to_csv("fold_report_v34.csv", index=False)
fold_report.to_csv("submissions/fold_report_v34.csv", index=False)
print(f"\n[DONE] Saved fold reports.")
print(fold_report.to_string(index=False))

# Boundary Preservation and Submission Save
tst_final = np.clip(np.mean(all_tst_stacked, axis=0), 0.0, 1.0)
submission = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": tst_final
})
submission.to_csv("submission_v34.csv", index=False)
submission.to_csv("submissions/submission_v34.csv", index=False)
print(f"[DONE] Saved submissions ({len(submission)} rows)")

# Save OOF (all_oof_stacked has exactly 20881 rows corresponding to train.csv)
np.save("oof_v34.npy", all_oof_stacked)
np.save("submissions/oof_v34.npy", all_oof_stacked)
print(f"[DONE] Saved oof_v34.npy and submissions/oof_v34.npy (for evaluate.py)")
