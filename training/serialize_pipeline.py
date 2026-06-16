"""
FloodGuard SL — Serialization Pipeline
================================================================================
Based on v703 (6-Model, Single-Seed, 5-Fold CV)
This script trains the production model and serializes all artifacts needed
by the production inference engine.

Outputs (to production/models/prod_v1/):
  model_metadata.json   — version, OOF scores, architecture info
  preprocessing.pkl     — freq_maps, KNN imputers, district stats, global stats
  te_maps.pkl           — full-dataset target encoding maps
  feature_lists.json    — BASE_FEATURES, FEATURES, cat feature names
  stacker.json          — stacking weights + bias
  posthoc.json          — power transform params (a, b, c)
  xgb1.json             — XGB-MAE-1 model (from last fold)
  xgb2.json             — XGB-MAE-2 model (from last fold)
  cat1.cbm              — CAT-MAE-1 model (from last fold)
  cat2.cbm              — CAT-MAE-2 model (from last fold)
  catrmse.cbm           — CAT-RMSE model (from last fold)
  lgb1.txt              — LGB-MAE model (from last fold)

Also outputs:
  production/data/district_reference.json  — per-district median features

Usage:
  cd KruthimaOps
  python training/serialize_pipeline.py

Runtime: ~60-90 minutes on CPU (1 seed, 5 folds)
"""

import json
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import explained_variance_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
DATA_DIR    = "data"
OUTPUT_DIR  = os.path.join("production", "models", "prod_v1000")
DIST_REF    = os.path.join("production", "data", "district_reference.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DIST_REF), exist_ok=True)

# ============================================================
# CONFIGURATION (same as v703)
# ============================================================
SEED       = 42         # Single seed for speed. Change to list for full mode.
USE_PSEUDO = True
N_FOLDS    = 5
SMOOTHING            = 10
SMOOTHING_COMPOSITE  = 15
c_mae, c_rmse, c_ev  = 0.539328, 1.152263, 0.048467

MODEL_NAMES = [
    "XGB-MAE-1 (d7)",
    "CAT-MAE-1 (d5)",
    "CAT-MAE-2 (d5)",
    "CAT-RMSE (d5)",
    "LGB-MAE (d5)",
    "XGB-MAE-2 (d5)",
]

print("=" * 75)
print("  FloodGuard SL — Production Serialization Pipeline")
print("  Based on v703 | Output ->", OUTPUT_DIR)
print("=" * 75)

# ============================================================
# 1. LOAD & DEDUPLICATE
# ============================================================
print("\n[LOAD] Loading data...")
train_df = pd.read_csv("C:/KruthimaOps/data/train_v1001_gee.csv")
train_df = train_df[train_df['is_synthetic'].isna()].reset_index(drop=True)
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
for col in ["hand_metric", "slope_deg", "water_occurrence_pct"]:
    if col not in test_df.columns:
        test_df[col] = 0.0
train_df = train_df.drop_duplicates()
print(f"   Train: {train_df.shape}  Test: {test_df.shape}")

# ============================================================
# 1.3. PRECISION FINGERPRINTS
# ============================================================
print("\n[FEAT] Extracting coordinate decimal precision fingerprints...")
for df in [train_df, test_df]:
    df['lat_decimal_len'] = df['latitude'].apply(
        lambda x: len(str(x).split('.')[1]) if '.' in str(x) and str(x).lower() != 'nan' else 0
    )
    df['lon_decimal_len'] = df['longitude'].apply(
        lambda x: len(str(x).split('.')[1]) if '.' in str(x) and str(x).lower() != 'nan' else 0
    )

# ============================================================
# 1.4. VALUE FREQUENCY COUNT FEATURES
# ============================================================
print("\n[FEAT] Computing value frequency count features...")
FREQ_COLS = [
    'latitude', 'longitude', 'elevation_m', 'distance_to_river_m',
    'rainfall_7d_mm', 'monthly_rainfall_mm', 'inundation_area_sqm'
]
combined_raw = pd.concat([
    train_df[[c for c in FREQ_COLS if c in train_df.columns]],
    test_df[[c for c in FREQ_COLS if c in test_df.columns]]
], ignore_index=True)

freq_maps = {}
for col in FREQ_COLS:
    if col in combined_raw.columns:
        freq_maps[col] = combined_raw[col].value_counts().to_dict()
        for df in [train_df, test_df]:
            if col in df.columns:
                df[f'{col}_freq'] = df[col].map(freq_maps[col]).fillna(0).astype(float)
        print(f"   {col}_freq: {len(freq_maps[col])} unique values")

# ============================================================
# 1.6. PSEUDO-LABELING
# ============================================================
print(f"\n[SEMI-SUPERVISED] Pseudo-Labeling: USE_PSEUDO={USE_PSEUDO}")
train_df['is_pseudo'] = 0
test_df['is_pseudo']  = 0

if USE_PSEUDO:
    pseudo_path = "submissions/submission_v30.csv"
    if not os.path.exists(pseudo_path):
        pseudo_path = "submission_v30.csv"
    if os.path.exists(pseudo_path):
        sub_blend   = pd.read_csv(pseudo_path)
        test_pseudo = test_df.merge(sub_blend, on="record_id", how="left")
        pseudo_rows = test_pseudo.copy()
        pseudo_rows['is_pseudo'] = 1
        print(f"   Added {len(pseudo_rows)} pseudo-labeled rows.")
        train_df = pd.concat([train_df, pseudo_rows], ignore_index=True)
    else:
        print("   [WARNING] submission_v30.csv not found. Skipping pseudo-labeling.")

# ============================================================
# 2. GEOSPATIAL IMPUTATION — [SERIALIZE] Save KNN objects
# ============================================================
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
            df.loc[idx, 'latitude']  = coords_lookup[key]['latitude']
            df.loc[idx, 'longitude'] = coords_lookup[key]['longitude']

knn_models = {}
for col in ['elevation_m', 'distance_to_river_m']:
    donor_pool = combined.dropna(subset=['latitude', 'longitude', col])
    knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
    knn.fit(donor_pool[['latitude', 'longitude']], donor_pool[col])
    knn_models[col] = knn                                  # [SERIALIZE]
    for df in [train_df, test_df]:
        mm = df[col].isnull() & df['latitude'].notnull() & df['longitude'].notnull()
        if mm.any():
            df.loc[mm, col] = knn.predict(df.loc[mm, ['latitude', 'longitude']])

# [SERIALIZE] District medians for fallback imputation
district_medians_by_col = {}
for col in ['elevation_m', 'distance_to_river_m', 'latitude', 'longitude']:
    district_medians_by_col[col] = combined.groupby('district')[col].median().to_dict()
    for df in [train_df, test_df]:
        df[col] = df[col].fillna(df.groupby('district')[col].transform('median'))
        df[col] = df[col].fillna(train_df[col].median())

global_fallback_medians = {col: float(train_df[col].median())
                           for col in ['elevation_m', 'distance_to_river_m', 'latitude', 'longitude']}
print("   Done.")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n[FEAT] Engineering features...")
district_elev_std        = combined.groupby('district')['elevation_m'].std().to_dict()
landcover_mean_inundation = combined.groupby('landcover')['inundation_area_sqm'].mean().to_dict()
combined_inundation_mean  = float(combined['inundation_area_sqm'].mean())      # [SERIALIZE]
soil_infilt_map    = {'Sandy': 0.8, 'Loamy': 0.6, 'Silty': 0.4, 'Clay': 0.2, 'Peaty': 0.1}
cyclone_districts  = {'Batticaloa', 'Trincomalee', 'Ampara', 'Mullaitivu', 'Jaffna'}
wet_zone_districts = {'Colombo', 'Gampaha', 'Kalutara', 'Galle', 'Matara', 'Ratnapura', 'Kegalle'}


def engineer_features(df, dist_elev_std, lc_inund_mean, comb_inund_mean, soil_map, cyc_d, wet_d):
    df = df.copy()
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
    has_reason = (~df["reason_not_good_to_live"].astype(str).str.strip().str.lower().isin(
        ["nan", "none", "", "missing", "n/a"])).astype(int)
    df["downstream_risk_count"] = (
        (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes").astype(int) +
        (df["is_good_to_live"].astype(str).str.strip().str.lower() == "no").astype(int) +
        has_reason +
        (df["inundation_area_sqm"] > 0).astype(int)
    )
    df['downstream_sig'] = (
        df['flood_occurrence_current_event'].astype(str).str.strip() + "_" +
        df['is_good_to_live'].astype(str).str.strip() + "_" +
        df['reason_not_good_to_live'].astype(str).str.strip()
    )
    has_inundation = (df["inundation_area_sqm"] > 0).astype(int)
    df["downstream_quad_sig"] = (
        df["flood_occurrence_current_event"].astype(str).str.strip() + "_" +
        df["is_good_to_live"].astype(str).str.strip() + "_" +
        df["reason_not_good_to_live"].astype(str).str.strip() + "_" +
        has_inundation.astype(str)
    )
    date_series = pd.to_datetime(df['generation_date'])
    df['month']   = date_series.dt.month
    df['is_yala'] = df['month'].isin([5, 6, 7, 8, 9]).astype(int)
    df['is_maha'] = df['month'].isin([11, 12, 1]).astype(int)
    df['zone_code'] = df['district'].astype(str).map(lambda x: 1 if x in wet_d else 2)
    df['monsoon_impact'] = (
        df['rainfall_7d_mm'] * df['is_yala'] * (df['zone_code'] == 1).astype(int) +
        df['rainfall_7d_mm'] * df['is_maha'] * (df['zone_code'] == 2).astype(int)
    )
    df['urban_runoff_potential']  = df['rainfall_7d_mm'] * df['built_up_percent'] * (1.0 / (df['drainage_index'] + 1e-5))
    df['fluvial_risk_score_feat'] = df['rainfall_7d_mm'] * (1.0 / (df['distance_to_river_m'] + 1.0))
    df['soil_infiltration']       = df['soil_type'].astype(str).map(soil_map).fillna(0.4)
    df['soil_saturation_limit']   = df['rainfall_7d_mm'] / (df['soil_infiltration'] + 0.1)
    df['pseudo_twi']              = np.log1p((df['distance_to_river_m'] + 1.0) / (df['elevation_m'].clip(lower=0.0) + 1.0))
    df['flatness_index']          = df['district'].astype(str).map(dist_elev_std).fillna(df['elevation_m'].std())
    df['in_cyclone_path']         = df['district'].astype(str).map(lambda x: 1 if x in cyc_d else 0)
    df['cyclone_vulnerability']   = df['in_cyclone_path'] * df['extreme_weather_index']
    df['slope_proxy']             = df['elevation_m'] / (df['distance_to_river_m'] + 1.0)
    df['isolation_index']         = np.log1p(df['nearest_hospital_km']) + np.log1p(df['nearest_evac_km'])
    df['vulnerability']           = df['isolation_index'] / (df['infrastructure_score'] + 1.0)
    df['elevation_divergence']    = df['elevation_m'] - df['elevation_m_yeojohnson']
    df['infra_deficit_sig'] = (
        df['water_supply'].astype(str).str.strip() + "_" +
        df['electricity'].astype(str).str.strip() + "_" +
        df['road_quality'].astype(str).str.strip()
    )
    df["inundation_area_log"]         = np.log1p(df["inundation_area_sqm"])
    df["flood_occurrence_yes"]        = (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes").astype(int)
    df["inundation_flood_interaction"] = df["flood_occurrence_yes"] * df["inundation_area_log"]
    df["river_rain_interaction"]      = df["distance_to_river_m_log1p"] * df["rainfall_7d_mm_log1p"]
    df["river_monthly_exposure"]      = df["distance_to_river_m_log1p"] * df["monthly_rainfall_mm_log1p"]
    df["elev_rain_risk"]              = df["elevation_m_yeojohnson"] / (df["rainfall_7d_mm_log1p"] + 1e-6)
    df["water_signal"]                = df["ndwi_qmap"].clip(lower=0)
    df["drainage_deficit"]            = (df["rainfall_7d_mm_log1p"] + 1) * (1.0 - df["drainage_index_yeojohnson"].clip(0, 1))
    df["infra_resilience"]            = df["infrastructure_score"] / (df["population_density_per_km2_log1p"] + 1e-6)
    df["evacuation_difficulty"]       = df["nearest_hospital_km_log1p"] + df["nearest_evac_km_log1p"]
    df["inundation_density_risk"]     = df["inundation_area_log"] / (df["population_density_per_km2_log1p"] + 1e-6)
    df["terrain_veg_risk"]            = df["terrain_roughness_index"] * (1.0 - df["ndvi_qmap"].clip(-1, 1))
    df["flood_pressure"]              = df["extreme_weather_index"] * df["seasonal_index"].clip(lower=0)
    df["is_repeat_flood_zone"]        = (df["historical_flood_count"] > 2).astype(int)
    df["rain_spike_ratio"]            = df["rainfall_7d_mm"] / (df["monthly_rainfall_mm"] + 1e-6)
    df["confirmed_risk"]              = (
        (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes") &
        (df["is_good_to_live"].astype(str).str.strip().str.lower() == "no")
    ).astype(int)

    df["is_historical_water"]     = (df.get("water_occurrence_pct", 0.0) > 5.0).astype(int)
    df["water_elevation_ratio"]   = df.get("water_occurrence_pct", 0.0) / (df["elevation_m"] + 1.0)

    df['landcover_mean_inundation_val'] = df['landcover'].astype(str).map(lc_inund_mean).fillna(comb_inund_mean)
    df['inundation_ratio']            = df['inundation_area_sqm'] / (df['landcover_mean_inundation_val'] + 1.0)
    ndwi_clip = df["ndwi_qmap"].clip(lower=0.0)
    ndvi_clip = df["ndvi_qmap"].clip(-1.0, 1.0).clip(lower=0.0)
    df["pooling_vulnerability"]        = ndwi_clip * (1.0 - ndvi_clip)
    df["soil_drainage_saturation"]     = df["soil_saturation_limit"] * (1.0 - df["drainage_index_yeojohnson"].clip(0.0, 1.0))
    lat = df["latitude"].fillna(df["latitude"].median())
    lon = df["longitude"].fillna(df["longitude"].median())
    df["grid_id_100"] = (lat / 1.0).astype(int).astype(str) + "_" + (lon / 1.0).astype(int).astype(str)
    df["grid_id_050"] = (lat / 0.5).astype(int).astype(str) + "_" + (lon / 0.5).astype(int).astype(str)
    df["grid_id_025"] = (lat / 0.25).astype(int).astype(str) + "_" + (lon / 0.25).astype(int).astype(str)
    df["grid_id_012"] = (lat / 0.125).astype(int).astype(str) + "_" + (lon / 0.125).astype(int).astype(str)
    df["lat_bin"]     = (lat / 0.5).astype(int)
    df["lon_bin"]     = (lon / 0.5).astype(int)
    df["grid_id"]     = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
    df = df.drop(columns=["inundation_area_sqm", "landcover_mean_inundation_val"])
    return df


# [SERIALIZE] Save a raw snapshot of test_df BEFORE engineer_features for district_reference
test_df_raw_snap = test_df.copy()

train_df = engineer_features(train_df, district_elev_std, landcover_mean_inundation,
                              combined_inundation_mean, soil_infilt_map, cyclone_districts, wet_zone_districts)
test_df  = engineer_features(test_df, district_elev_std, landcover_mean_inundation,
                              combined_inundation_mean, soil_infilt_map, cyclone_districts, wet_zone_districts)

# ============================================================
# 4. PREP & DTYPES
# ============================================================
TARGET    = "flood_risk_score"
ID_COL    = "record_id"
DROP_COLS = [ID_COL, "place_name", "is_synthetic", "generation_date", "is_pseudo"]
CAT_FEATURES = [
    "district", "landcover", "soil_type", "water_supply",
    "electricity", "road_quality", "urban_rural",
    "water_presence_flag", "flood_occurrence_current_event",
    "is_good_to_live", "reason_not_good_to_live", "downstream_risk_count"
]
TARGET_ENC_COLS = [
    "district", "grid_id", "downstream_sig", "downstream_quad_sig", "infra_deficit_sig",
    "landcover", "soil_type", "water_supply", "electricity", "road_quality",
    "downstream_risk_count", "grid_id_100", "grid_id_050", "grid_id_025", "grid_id_012"
]
COMPOSITE_ENC_COLS = []
STD_ENC_COLS = ["district", "downstream_sig"]
IGNORE_COLS = DROP_COLS + [TARGET, "flood_occurrence_yes", "downstream_sig",
                            "downstream_quad_sig", "infra_deficit_sig"]
SPATIAL_HELPERS = ["lat_bin", "lon_bin", "grid_id", "grid_id_100", "grid_id_050", "grid_id_025", "grid_id_012"]
BASE_FEATURES = [c for c in train_df.columns if c not in IGNORE_COLS and c not in SPATIAL_HELPERS]

print(f"\n[PREP] Base features: {len(BASE_FEATURES)}")

cat_dtype_map = {}
for col in BASE_FEATURES:
    if col in CAT_FEATURES:
        train_df[col] = train_df[col].fillna("missing").astype(str)
        test_df[col]  = test_df[col].fillna("missing").astype(str)
        all_vals = sorted(set(train_df[col].unique()) | set(test_df[col].unique()))
        cdt = pd.CategoricalDtype(categories=all_vals, ordered=False)
        train_df[col] = train_df[col].astype(cdt)
        test_df[col]  = test_df[col].astype(cdt)
        cat_dtype_map[col] = all_vals   # [SERIALIZE] save category lists
    elif train_df[col].dtype in ["int64", "float64", "int32", "float32"]:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col]  = test_df[col].fillna(median_val)

cat_feature_names = [c for c in CAT_FEATURES if c in BASE_FEATURES]

# Global stats from real rows only
y          = train_df[TARGET]
real_mask  = train_df['is_pseudo'] == 0
GLOBAL_MEAN   = float(y[real_mask].mean())
GLOBAL_STD    = float(y[real_mask].std())
GLOBAL_Q25    = float(y[real_mask].quantile(0.25))
GLOBAL_Q75    = float(y[real_mask].quantile(0.75))
GLOBAL_MEDIAN = float(y[real_mask].median())

downstream_cols = [
    "confirmed_severe_risk", "no_flood_confirmed", "inundation_per_capita", "downstream_risk_count",
    "downstream_sig", "downstream_quad_sig", "confirmed_risk", "inundation_ratio", "flood_occurrence_yes",
    "inundation_flood_interaction", "inundation_density_risk"
]
conflict_key_cols = [c for c in BASE_FEATURES if c not in downstream_cols]

def to_xgb_fmt(df):
    df = df.copy()
    for c in df.columns:
        if hasattr(df[c], "cat"):
            df[c] = df[c].cat.codes.astype("int32")
    return df

def to_cat_fmt_local(df, cat_cols):
    df = df.copy()
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

# ============================================================
# 5. STACKING UTILITIES
# ============================================================
gkf        = GroupKFold(n_splits=N_FOLDS)
groups     = train_df['grid_id'].values
y_arr      = y.values
original_y = y_arr[real_mask]
original_df = train_df[real_mask].reset_index(drop=True)
original_groups = original_df['grid_id'].values
gkf_l2     = GroupKFold(n_splits=N_FOLDS)


def fit_metric_stacker(X_meta, y_true, alpha=0.1):
    n_models = X_meta.shape[1]
    def loss_fn(params):
        w         = params[:n_models]
        intercept = params[n_models]
        pred      = np.clip(np.dot(X_meta, w) + intercept, 0.0, 1.0)
        mae  = mean_absolute_error(y_true, pred)
        rmse = root_mean_squared_error(y_true, pred)
        ev   = explained_variance_score(y_true, pred)
        score = (c_mae * mae + c_rmse * rmse) * (1.0 + c_ev * (1.0 - ev))
        return score + alpha * np.sum(w**2)
    init_guess = np.append(np.ones(n_models) / n_models, 0.0)
    bounds     = [(0.0, None)] * n_models + [(None, None)]
    res        = minimize(loss_fn, init_guess, bounds=bounds, method='L-BFGS-B')
    return res.x[:-1], res.x[-1]


# ============================================================
# 6. TRAINING LOOP  [SERIALIZE] Save last-fold models
# ============================================================
print(f"\n{'=' * 75}")
print(f"  5-FOLD SPATIAL GROUP CV — SEED {SEED}")
print(f"{'=' * 75}")

oof_preds = {m: np.zeros(len(train_df)) for m in MODEL_NAMES}
tst_preds = {m: np.zeros(len(test_df))  for m in MODEL_NAMES}
te_features_ref = []   # [SERIALIZE] captured from last fold
FEATURES_ref    = []   # [SERIALIZE] captured from last fold

t_start = time.time()

for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, groups)):
    t0 = time.time()
    va_is_pseudo  = train_df.iloc[va_idx]['is_pseudo'] == 1
    va_idx_clean  = va_idx[~va_is_pseudo] if va_is_pseudo.any() else va_idx
    tr_rows = train_df.iloc[tr_idx].copy()
    va_rows = train_df.iloc[va_idx_clean].copy()

    # Conflict resolution
    temp_tr = tr_rows[conflict_key_cols].copy()
    for col in temp_tr.columns:
        if temp_tr[col].dtype in ['object', 'category']:
            temp_tr[col] = temp_tr[col].astype(str).fillna('missing')
        else:
            temp_tr[col] = temp_tr[col].fillna(-999)
    group_medians_tr = tr_rows.groupby([temp_tr[c] for c in temp_tr.columns])[TARGET].transform('median')
    tr_rows[TARGET]  = group_medians_tr

    # Target encodings
    real_tr_rows = tr_rows[tr_rows['is_pseudo'] == 0]
    all_te_cols  = TARGET_ENC_COLS + COMPOSITE_ENC_COLS
    for col in all_te_cols:
        group_stats = real_tr_rows.groupby(col)[TARGET].agg(
            median='median', count='count', mean='mean', std='std',
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75)
        )
        group_stats['std'] = group_stats['std'].fillna(0.0)
        s = SMOOTHING_COMPOSITE if col in COMPOSITE_ENC_COLS else SMOOTHING
        smoothed_median = (group_stats['count'] * group_stats['median'] + s * GLOBAL_MEDIAN) / (group_stats['count'] + s)
        smoothed_mean   = (group_stats['count'] * group_stats['mean']   + s * GLOBAL_MEAN)   / (group_stats['count'] + s)
        smoothed_std    = (group_stats['count'] * group_stats['std']    + s * GLOBAL_STD)    / (group_stats['count'] + s)
        smoothed_q25    = (group_stats['count'] * group_stats['q25']    + s * GLOBAL_Q25)    / (group_stats['count'] + s)
        smoothed_q75    = (group_stats['count'] * group_stats['q75']    + s * GLOBAL_Q75)    / (group_stats['count'] + s)
        log_count       = np.log1p(group_stats['count'])
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

    # Capture from last fold  [SERIALIZE]
    if fold == N_FOLDS - 1:
        te_features_ref = te_features
        FEATURES_ref    = FEATURES

    y_tr, y_va = tr_rows[TARGET], va_rows[TARGET]
    X_tr  = tr_rows[FEATURES].copy()
    X_va  = va_rows[FEATURES].copy()
    X_te  = test_df[FEATURES].copy()
    cat_cols = [c for c in CAT_FEATURES if c in FEATURES]

    X_tr_xgb = to_xgb_fmt(X_tr);  X_va_xgb = to_xgb_fmt(X_va);  X_te_xgb = to_xgb_fmt(X_te)
    X_tr_cat = to_cat_fmt_local(X_tr, cat_cols)
    X_va_cat = to_cat_fmt_local(X_va, cat_cols)
    X_te_cat = to_cat_fmt_local(X_te, cat_cols)

    for col in cat_cols:
        cdt = pd.CategoricalDtype(categories=cat_dtype_map[col], ordered=False)
        X_tr[col] = X_tr[col].astype(str).astype(cdt)
        X_va[col] = X_va[col].astype(str).astype(cdt)
        X_te[col] = X_te[col].astype(str).astype(cdt)

    # Model 1: XGB-MAE-1 (d7)
    xgb_m1 = xgb.XGBRegressor(
        n_estimators=4000, learning_rate=0.03, max_depth=7, min_child_weight=4,
        subsample=0.85, colsample_bytree=0.6, colsample_bylevel=0.6,
        reg_alpha=2.0, reg_lambda=4.0, gamma=0.1, max_delta_step=1,
        objective="reg:absoluteerror", eval_metric="mae", tree_method="hist",
        enable_categorical=False, early_stopping_rounds=100, random_state=SEED, n_jobs=-1
    )
    xgb_m1.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)

    # Model 2: CAT-MAE-1 (d5)
    cat_m1 = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.03, depth=5, l2_leaf_reg=5.0,
        bagging_temperature=0.7, random_strength=2.0, border_count=254,
        loss_function="MAE", eval_metric="MAE", task_type="CPU",
        random_seed=SEED, verbose=False
    )
    cat_m1.fit(X_tr_cat, y_tr, cat_features=cat_cols,
               eval_set=(X_va_cat, y_va), early_stopping_rounds=150, verbose=False)

    # Model 3: CAT-MAE-2 (d5)
    cat_m2 = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.03, depth=5, l2_leaf_reg=12.0,
        bagging_temperature=0.4, random_strength=5.0, border_count=254,
        loss_function="MAE", eval_metric="MAE", task_type="CPU",
        random_seed=SEED + 1, verbose=False
    )
    cat_m2.fit(X_tr_cat, y_tr, cat_features=cat_cols,
               eval_set=(X_va_cat, y_va), early_stopping_rounds=150, verbose=False)

    # Model 4: CAT-RMSE (d5)
    cat_rmse = cb.CatBoostRegressor(
        iterations=4000, learning_rate=0.03, depth=5, l2_leaf_reg=8.0,
        bagging_temperature=0.6, random_strength=3.0, border_count=254,
        loss_function="RMSE", eval_metric="RMSE", task_type="CPU",
        random_seed=SEED + 2, verbose=False
    )
    cat_rmse.fit(X_tr_cat, y_tr, cat_features=cat_cols,
                 eval_set=(X_va_cat, y_va), early_stopping_rounds=150, verbose=False)

    # Model 5: LGB-MAE (d5)
    lgb_m1 = lgb.LGBMRegressor(
        n_estimators=4000, learning_rate=0.03, num_leaves=31, max_depth=5,
        min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
        reg_alpha=2.0, reg_lambda=5.0, objective="regression_l1",
        random_state=SEED, n_jobs=-1, verbosity=-1
    )
    lgb_m1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
               callbacks=[lgb.early_stopping(150, verbose=False)])

    # Model 6: XGB-MAE-2 (d5)
    xgb_m2 = xgb.XGBRegressor(
        n_estimators=4000, learning_rate=0.03, max_depth=5, min_child_weight=6,
        subsample=0.75, colsample_bytree=0.5, colsample_bylevel=0.8,
        reg_alpha=5.0, reg_lambda=10.0, gamma=0.2, max_delta_step=1,
        objective="reg:absoluteerror", eval_metric="mae", tree_method="hist",
        enable_categorical=False, early_stopping_rounds=100, random_state=SEED + 3, n_jobs=-1
    )
    xgb_m2.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)

    # OOF predictions
    oof_preds["XGB-MAE-1 (d7)"][va_idx_clean] = xgb_m1.predict(X_va_xgb)
    oof_preds["CAT-MAE-1 (d5)"][va_idx_clean] = cat_m1.predict(X_va_cat)
    oof_preds["CAT-MAE-2 (d5)"][va_idx_clean] = cat_m2.predict(X_va_cat)
    oof_preds["CAT-RMSE (d5)"][va_idx_clean]  = cat_rmse.predict(X_va_cat)
    oof_preds["LGB-MAE (d5)"][va_idx_clean]   = lgb_m1.predict(X_va)
    oof_preds["XGB-MAE-2 (d5)"][va_idx_clean] = xgb_m2.predict(X_va_xgb)

    # Test predictions (averaged across folds)
    tst_preds["XGB-MAE-1 (d7)"] += xgb_m1.predict(X_te_xgb) / N_FOLDS
    tst_preds["CAT-MAE-1 (d5)"] += cat_m1.predict(X_te_cat)  / N_FOLDS
    tst_preds["CAT-MAE-2 (d5)"] += cat_m2.predict(X_te_cat)  / N_FOLDS
    tst_preds["CAT-RMSE (d5)"]  += cat_rmse.predict(X_te_cat) / N_FOLDS
    tst_preds["LGB-MAE (d5)"]   += lgb_m1.predict(X_te)       / N_FOLDS
    tst_preds["XGB-MAE-2 (d5)"] += xgb_m2.predict(X_te_xgb)  / N_FOLDS

    oof_avg_fold = np.mean([oof_preds[m][va_idx_clean] for m in MODEL_NAMES], axis=0)
    f_mae  = mean_absolute_error(va_rows[TARGET].values, oof_avg_fold)
    f_rmse = root_mean_squared_error(va_rows[TARGET].values, oof_avg_fold)
    print(f"   Fold {fold+1}/5 | MAE={f_mae:.4f} RMSE={f_rmse:.4f} | [{time.time() - t0:.0f}s]")

    # [SERIALIZE] Save last fold models
    if fold == N_FOLDS - 1:
        print(f"\n[SERIALIZE] Saving last-fold models to {OUTPUT_DIR}/...")
        xgb_m1.save_model(os.path.join(OUTPUT_DIR, "xgb1.json"))
        cat_m1.save_model(os.path.join(OUTPUT_DIR, "cat1.cbm"))
        cat_m2.save_model(os.path.join(OUTPUT_DIR, "cat2.cbm"))
        cat_rmse.save_model(os.path.join(OUTPUT_DIR, "catrmse.cbm"))
        lgb_m1.booster_.save_model(os.path.join(OUTPUT_DIR, "lgb1.txt"))
        xgb_m2.save_model(os.path.join(OUTPUT_DIR, "xgb2.json"))
        print("   Models saved.")

# ============================================================
# 7. STACKING
# ============================================================
print("\n[STACK] Running L2 alpha grid search...")
oof_meta = np.column_stack([oof_preds[m][real_mask] for m in MODEL_NAMES])
tst_meta = np.column_stack([tst_preds[m] for m in MODEL_NAMES])

best_alpha, best_score = 0.1, np.inf
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
    oof_cv = np.zeros(len(original_y))
    for fold, (tr_l2, va_l2) in enumerate(gkf_l2.split(original_df, original_y, original_groups)):
        w_cv, b_cv = fit_metric_stacker(oof_meta[tr_l2], original_y[tr_l2], alpha=alpha)
        oof_cv[va_l2] = np.clip(np.dot(oof_meta[va_l2], w_cv) + b_cv, 0.0, 1.0)
    cv_mae  = mean_absolute_error(original_y, oof_cv)
    cv_rmse = root_mean_squared_error(original_y, oof_cv)
    cv_ev   = explained_variance_score(original_y, oof_cv)
    cv_score = (c_mae * cv_mae + c_rmse * cv_rmse) * (1.0 + c_ev * (1.0 - cv_ev))
    if cv_score < best_score:
        best_score = cv_score
        best_alpha = alpha
print(f"   Best alpha: {best_alpha} | Nested CV Score: {best_score:.5f}")

oof_stacked = np.zeros(len(original_y))
for fold, (tr_l2, va_l2) in enumerate(gkf_l2.split(original_df, original_y, original_groups)):
    w_fold, b_fold = fit_metric_stacker(oof_meta[tr_l2], original_y[tr_l2], alpha=best_alpha)
    oof_stacked[va_l2] = np.clip(np.dot(oof_meta[va_l2], w_fold) + b_fold, 0.0, 1.0)

w_final, b_final = fit_metric_stacker(oof_meta, original_y, alpha=best_alpha)
tst_stacked = np.clip(np.dot(tst_meta, w_final) + b_final, 0.0, 1.0)

g_mae  = mean_absolute_error(original_y, oof_stacked)
g_rmse = root_mean_squared_error(original_y, oof_stacked)
g_ev   = explained_variance_score(original_y, oof_stacked)
g_lb   = (c_mae * g_mae + c_rmse * g_rmse) * (1.0 + c_ev * (1.0 - g_ev))
print(f"\n[OOF] MAE={g_mae:.5f} RMSE={g_rmse:.5f} EV={g_ev:.5f} Est.LB={g_lb:.5f}")

# ============================================================
# 8. POST-HOC POWER TRANSFORMATION
# ============================================================
print("\n[POSTHOC] Optimizing power transform...")
def transform_loss(params):
    a, b, c = params
    pred = np.clip(a * np.power(np.clip(oof_stacked, 1e-6, None), b) + c, 0.0, 1.0)
    mae  = mean_absolute_error(original_y, pred)
    rmse = root_mean_squared_error(original_y, pred)
    ev   = explained_variance_score(original_y, pred)
    return (c_mae * mae + c_rmse * rmse) * (1.0 + c_ev * (1.0 - ev))

res_opt = minimize(transform_loss, [1.0, 1.0, 0.0],
                   bounds=[(0.5, 1.5), (0.5, 2.0), (-0.15, 0.15)], method='L-BFGS-B')
a_opt, b_opt, c_opt = res_opt.x

opt_oof  = np.clip(a_opt * np.power(np.clip(oof_stacked, 1e-6, None), b_opt) + c_opt, 0.0, 1.0)
opt_mae  = mean_absolute_error(original_y, opt_oof)
opt_rmse = root_mean_squared_error(original_y, opt_oof)
opt_ev   = explained_variance_score(original_y, opt_oof)
opt_lb   = (c_mae * opt_mae + c_rmse * opt_rmse) * (1.0 + c_ev * (1.0 - opt_ev))
print(f"   a={a_opt:.5f} b={b_opt:.5f} c={c_opt:.5f}")
print(f"   Opt.LB={opt_lb:.5f} | MAE={opt_mae:.5f} RMSE={opt_rmse:.5f} EV={opt_ev:.5f}")

# ============================================================
# 9. [SERIALIZE] FULL-DATASET TE MAPS (for production inference)
# ============================================================
print("\n[SERIALIZE] Computing full-dataset target encoding maps...")
real_train_full = train_df[real_mask].copy()
te_maps = {}
all_te_cols_full = TARGET_ENC_COLS + COMPOSITE_ENC_COLS

for col in all_te_cols_full:
    group_stats = real_train_full.groupby(col)[TARGET].agg(
        median='median', count='count', mean='mean', std='std',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    )
    group_stats['std'] = group_stats['std'].fillna(0.0)
    s = SMOOTHING_COMPOSITE if col in COMPOSITE_ENC_COLS else SMOOTHING

    sm = (group_stats['count'] * group_stats['median'] + s * GLOBAL_MEDIAN) / (group_stats['count'] + s)
    ss = (group_stats['count'] * group_stats['std']    + s * GLOBAL_STD)    / (group_stats['count'] + s)
    sq25 = (group_stats['count'] * group_stats['q25']  + s * GLOBAL_Q25)    / (group_stats['count'] + s)
    sq75 = (group_stats['count'] * group_stats['q75']  + s * GLOBAL_Q75)    / (group_stats['count'] + s)
    lc   = np.log1p(group_stats['count'])

    te_maps[col] = {
        'smoothed_median': sm.to_dict(),
        'smoothed_q25':    sq25.to_dict(),
        'smoothed_q75':    sq75.to_dict(),
        'log_count':       lc.to_dict(),
        'smoothed_std':    ss.to_dict(),
    }

print(f"   TE maps computed for {len(te_maps)} columns.")

# ============================================================
# 10. [SERIALIZE] SAVE ALL ARTIFACTS
# ============================================================
print("\n[SERIALIZE] Saving preprocessing artifacts...")

preprocessing = {
    'freq_maps':               freq_maps,
    'coords_lookup':           {str(k): v for k, v in coords_lookup.items()},
    'knn_models':              knn_models,
    'district_medians':        district_medians_by_col,
    'global_fallback_medians': global_fallback_medians,
    'district_elev_std':       district_elev_std,
    'landcover_mean_inundation': landcover_mean_inundation,
    'combined_inundation_mean':  combined_inundation_mean,
    'cat_dtype_map':           cat_dtype_map,
    'global_stats': {
        'GLOBAL_MEAN':   GLOBAL_MEAN,
        'GLOBAL_STD':    GLOBAL_STD,
        'GLOBAL_Q25':    GLOBAL_Q25,
        'GLOBAL_Q75':    GLOBAL_Q75,
        'GLOBAL_MEDIAN': GLOBAL_MEDIAN,
    },
    'soil_infilt_map':    soil_infilt_map,
    'cyclone_districts':  list(cyclone_districts),
    'wet_zone_districts': list(wet_zone_districts),
    'freq_cols':          FREQ_COLS,
}

with open(os.path.join(OUTPUT_DIR, "preprocessing.pkl"), "wb") as f:
    pickle.dump(preprocessing, f)
print("   preprocessing.pkl saved.")

with open(os.path.join(OUTPUT_DIR, "te_maps.pkl"), "wb") as f:
    pickle.dump(te_maps, f)
print("   te_maps.pkl saved.")

feature_lists = {
    'BASE_FEATURES':      BASE_FEATURES,
    'TARGET_ENC_COLS':    TARGET_ENC_COLS,
    'COMPOSITE_ENC_COLS': COMPOSITE_ENC_COLS,
    'STD_ENC_COLS':       STD_ENC_COLS,
    'CAT_FEATURES':       CAT_FEATURES,
    'cat_feature_names':  cat_feature_names,
    'FEATURES':           FEATURES_ref,
    'te_features':        te_features_ref,
    'SMOOTHING':          SMOOTHING,
    'SMOOTHING_COMPOSITE': SMOOTHING_COMPOSITE,
    'FREQ_COLS':          FREQ_COLS,
}
with open(os.path.join(OUTPUT_DIR, "feature_lists.json"), "w") as f:
    json.dump(feature_lists, f, indent=2)
print("   feature_lists.json saved.")

stacker_config = {
    'weights':     w_final.tolist(),
    'bias':        float(b_final),
    'model_names': MODEL_NAMES,
    'best_alpha':  float(best_alpha),
}
with open(os.path.join(OUTPUT_DIR, "stacker.json"), "w") as f:
    json.dump(stacker_config, f, indent=2)
print("   stacker.json saved.")

posthoc = {'a': float(a_opt), 'b': float(b_opt), 'c': float(c_opt)}
with open(os.path.join(OUTPUT_DIR, "posthoc.json"), "w") as f:
    json.dump(posthoc, f, indent=2)
print("   posthoc.json saved.")

metadata = {
    'version':          'prod_v1000',
    'base_pipeline':    'v1000',
    'seed':             SEED,
    'n_folds':          N_FOLDS,
    'training_date':    datetime.now().isoformat(),
    'oof_mae':          round(g_mae, 6),
    'oof_rmse':         round(g_rmse, 6),
    'oof_ev':           round(g_ev, 6),
    'est_lb_score':     round(g_lb, 6),
    'opt_mae':          round(opt_mae, 6),
    'opt_rmse':         round(opt_rmse, 6),
    'opt_ev':           round(opt_ev, 6),
    'opt_lb_score':     round(opt_lb, 6),
    'model_names':      MODEL_NAMES,
    'n_base_features':  len(BASE_FEATURES),
    'n_total_features': len(FEATURES_ref),
    'model_files': {
        'xgb1':   'xgb1.json',
        'cat1':   'cat1.cbm',
        'cat2':   'cat2.cbm',
        'catrmse': 'catrmse.cbm',
        'lgb1':   'lgb1.txt',
        'xgb2':   'xgb2.json',
    }
}
with open(os.path.join(OUTPUT_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print("   model_metadata.json saved.")

# ============================================================
# 11. [SERIALIZE] BUILD DISTRICT REFERENCE JSON
# ============================================================
print("\n[SERIALIZE] Building district_reference.json...")

# Load existing reference json to preserve correct map coordinates
existing_ref = {}
if os.path.exists(DIST_REF):
    try:
        with open(DIST_REF, 'r') as f:
            existing_ref = json.load(f)
        print(f"   Loaded existing district_reference.json with {len(existing_ref)} districts to preserve coordinates.")
    except Exception as e:
        print(f"   Warning: could not load existing reference file: {e}")

# Use the raw test snapshot (before engineer_features)
# Merge with test_df to add freq features that were computed on test_df
freq_feat_cols = [f'{col}_freq' for col in FREQ_COLS if f'{col}_freq' in test_df_raw_snap.columns]

# Identify numeric vs categorical columns
num_cols = test_df_raw_snap.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_raw = test_df_raw_snap.select_dtypes(exclude=[np.number]).columns.tolist()
exclude_from_ref = ['record_id', 'is_synthetic', 'flood_risk_score']

district_ref = {}
for district, group in test_df_raw_snap.groupby('district'):
    ref = {}
    for col in num_cols:
        if col not in exclude_from_ref:
            ref[col] = float(group[col].median()) if not group[col].isna().all() else 0.0
    for col in cat_cols_raw:
        if col not in exclude_from_ref:
            mode_val = group[col].mode()
            ref[col] = str(mode_val.iloc[0]) if len(mode_val) > 0 else "missing"
    # Ensure generation_date is set to a representative value
    ref['generation_date'] = "2024-06-15"  # Fixed representative date
    # Coordinates for map centering (preserve original if they exist)
    if str(district) in existing_ref and 'center_lat' in existing_ref[str(district)] and 'center_lon' in existing_ref[str(district)]:
        ref['center_lat'] = existing_ref[str(district)]['center_lat']
        ref['center_lon'] = existing_ref[str(district)]['center_lon']
    else:
        ref['center_lat'] = float(group['latitude'].median())
        ref['center_lon'] = float(group['longitude'].median())
    district_ref[str(district)] = ref

with open(DIST_REF, "w") as f:
    json.dump(district_ref, f, indent=2)
print(f"   district_reference.json saved with {len(district_ref)} districts.")

# ============================================================
# 12. [SERIALIZE] PROD V1000 COMPATIBLE MODELS (XGB, LGB, CAT)
# ============================================================
print("\n[SERIALIZE] Training production compatible baseline models...")

# Define baseline features list expected by v1000_engine
prod_features = [
    "district", "latitude", "longitude", "elevation_m", "distance_to_river_m",
    "landcover", "soil_type", "water_supply", "electricity", "road_quality",
    "population_density_per_km2", "built_up_percent", "urban_rural",
    "rainfall_7d_mm", "monthly_rainfall_mm", "drainage_index", "ndvi", "ndwi",
    "water_presence_flag", "historical_flood_count", "infrastructure_score",
    "nearest_hospital_km", "nearest_evac_km", "flood_occurrence_current_event",
    "inundation_area_sqm", "is_good_to_live", "reason_not_good_to_live",
    "seasonal_index", "terrain_roughness_index", "socioeconomic_status_index",
    "extreme_weather_index", "hand_metric", "slope_deg", "water_occurrence_pct"
]

prod_cat_cols = [
    "district", "landcover", "soil_type", "water_supply", "electricity",
    "road_quality", "urban_rural", "water_presence_flag",
    "flood_occurrence_current_event", "is_good_to_live", "reason_not_good_to_live"
]

# Get the clean training data (no synthetic rows, no pseudo labels)
train_real = train_df[train_df['is_pseudo'] == 0].copy()

# Restore dropped inundation_area_sqm column from raw CSV
raw_inund = pd.read_csv("C:/KruthimaOps/data/train_v1001_gee.csv")[["record_id", "inundation_area_sqm"]]
train_real = train_real.merge(raw_inund, on="record_id", how="left")

# Fill missing categoricals as 'missing' and continuous as median
medians = {}
for col in prod_features:
    if col in prod_cat_cols:
        train_real[col] = train_real[col].fillna("missing").astype(str)
        medians[col] = "missing"
    else:
        median_val = float(train_real[col].median()) if not train_real[col].isna().all() else 0.0
        train_real[col] = train_real[col].fillna(median_val)
        medians[col] = median_val

# Prepare training data (category dtypes)
X_prod = train_real[prod_features].copy()
for col in prod_cat_cols:
    X_prod[col] = X_prod[col].astype("category")

y_prod = train_real[TARGET]

# Train production XGBoost on all real rows
xgb_prod = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.03, max_depth=5,
    objective="reg:absoluteerror", tree_method="hist",
    enable_categorical=True, random_state=SEED, n_jobs=-1
)
xgb_prod.fit(X_prod, y_prod, verbose=False)

# Train production LightGBM on all real rows
lgb_prod = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.03, max_depth=5,
    objective="regression_l1", random_state=SEED, n_jobs=-1, verbosity=-1
)
lgb_prod.fit(X_prod, y_prod)

# Train production CatBoost on all real rows
cat_prod = cb.CatBoostRegressor(
    iterations=500, learning_rate=0.03, depth=5,
    loss_function="MAE", random_seed=SEED, verbose=False
)
cat_prod.fit(X_prod, y_prod, cat_features=prod_cat_cols, verbose=False)

# Save the models in the exact filenames expected by v1000_engine
xgb_prod.save_model(os.path.join(OUTPUT_DIR, "xgb.json"))
lgb_prod.booster_.save_model(os.path.join(OUTPUT_DIR, "lgb.txt"))
cat_prod.save_model(os.path.join(OUTPUT_DIR, "cat.cbm"))

# Save feature_info.json
feature_info = {
    "features": prod_features,
    "cat_cols": prod_cat_cols,
    "medians": medians,
    "categories": cat_dtype_map
}
with open(os.path.join(OUTPUT_DIR, "feature_info.json"), "w") as f:
    json.dump(feature_info, f, indent=2)

print("   xgb.json, lgb.txt, cat.cbm, and feature_info.json saved successfully for v1000_engine.")

total_time = time.time() - t_start
print(f"\n{'=' * 75}")
print(f"  SERIALIZATION COMPLETE — Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
print(f"  Artifacts saved to: {OUTPUT_DIR}/")
print(f"  District reference: {DIST_REF}")
print(f"{'=' * 75}")
