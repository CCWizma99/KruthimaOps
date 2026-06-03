"""
ML Opsidian: Genesis v57 - EV-Optimized & Stable Custom Stacking Pipeline
==========================================================================
Upgrades and Fixes from v54:
1. Global-Only Calibration (Reverting Group-Specific Power Transforms):
   - Group-specific calibration caused severe overfitting on small slices. Reverted to a single global calibration curve.
2. EV-Prioritizing Stacker & Calibration:
   - Configures L2 stacker and global calibration loss function with weights from v42:
     c_mae = 0.392696, c_rmse = 0.875527, c_ev = 0.406963.
   - Puts a massive penalty on Explained Variance (EV) to align with actual LB scoring and act as a powerful regularizer.
3. Stable Feature & Model Core:
   - Keeps the 6 stable base models from v42/v45.
   - Removes experimental target conflict resolution (target smoothing) to preserve downstream signals.
   - Removes quantile dispersion features and LOO target encodings to prevent stacker overfitting.
4. High-Quality Pseudo-Labeling:
   - Uses submission_v42_optimized.csv (our best Public LB score at 0.38245) as the soft pseudo-label source.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from scipy.optimize import minimize
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
SEED = 42

# Automatic GPU Detection
HAS_GPU = False
try:
    import torch
    if torch.cuda.is_available():
        HAS_GPU = True
except ImportError:
    pass

CB_TASK_TYPE = "GPU" if HAS_GPU else "CPU"
XGB_DEVICE = "cuda" if HAS_GPU else "cpu"
print(f"GPU Configured: {HAS_GPU} (CatBoost: {CB_TASK_TYPE}, XGBoost: {XGB_DEVICE})")

print("=" * 75)
print("  ML OPSIDIAN v57 - EV-OPTIMIZED & STABLE CUSTOM STACKING")
print("=" * 75)

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
    # Use submission_v42_optimized.csv (our best Public LB score at 0.38245)
    pseudo_path = "submission_v42_optimized.csv"
    if not os.path.exists(pseudo_path):
        pseudo_path = "submissions/submission_v42_optimized.csv"
    
    # Fallback chains
    if not os.path.exists(pseudo_path):
        pseudo_path = "submission_v42.csv"
        if not os.path.exists(pseudo_path):
            pseudo_path = "submissions/submission_v42.csv"
            
    if not os.path.exists(pseudo_path):
        pseudo_path = "submission_v30.csv"
        if not os.path.exists(pseudo_path):
            pseudo_path = "submissions/submission_v30.csv"
    
    if os.path.exists(pseudo_path):
        sub_blend = pd.read_csv(pseudo_path)
        test_pseudo = test_df.merge(sub_blend, on="record_id", how="left")
        
        # Soft pseudo-labeling of all test rows
        pseudo_rows = test_pseudo.copy()
        pseudo_rows['is_pseudo'] = 1
        
        print(f"   Added {len(pseudo_rows)} soft pseudo-labeled rows from {pseudo_path}.")
        train_df = pd.concat([train_df, pseudo_rows], ignore_index=True)
    else:
        print("   [WARNING] No pseudo-label source found. Skipping pseudo-labeling.")

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
    
    # Downstream features (Track B)
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

    df['downstream_sig'] = (
        df['flood_occurrence_current_event'].fillna('missing').astype(str).str.strip() + "_" +
        df['is_good_to_live'].fillna('missing').astype(str).str.strip() + "_" +
        df['reason_not_good_to_live'].fillna('missing').astype(str).str.strip()
    )
    
    # Quad-state composite interaction
    has_inundation = (df["inundation_area_sqm"] > 0).astype(int)
    df["downstream_quad_sig"] = (
        df["flood_occurrence_current_event"].fillna('missing').astype(str).str.strip() + "_" +
        df["is_good_to_live"].fillna('missing').astype(str).str.strip() + "_" +
        df["reason_not_good_to_live"].fillna('missing').astype(str).str.strip() + "_" +
        has_inundation.astype(str)
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
        df['water_supply'].fillna('missing').astype(str).str.strip() + "_" +
        df['electricity'].fillna('missing').astype(str).str.strip() + "_" +
        df['road_quality'].fillna('missing').astype(str).str.strip()
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
    
    # environmental interactions
    ndwi_clip = df["ndwi_qmap"].clip(lower=0.0)
    ndvi_clip = df["ndvi_qmap"].clip(-1.0, 1.0).clip(lower=0.0)
    df["pooling_vulnerability"] = ndwi_clip * (1.0 - ndvi_clip)
    df["soil_drainage_saturation"] = df["soil_saturation_limit"] * (1.0 - df["drainage_index_yeojohnson"].clip(0.0, 1.0))
    
    # Hierarchical Spatial Grids
    lat = df["latitude"].fillna(df["latitude"].median())
    lon = df["longitude"].fillna(df["longitude"].median())
    
    df["grid_id_100"] = (lat / 1.0).astype(int).astype(str) + "_" + (lon / 1.0).astype(int).astype(str)
    df["grid_id_050"] = (lat / 0.5).astype(int).astype(str) + "_" + (lon / 0.5).astype(int).astype(str)
    df["grid_id_025"] = (lat / 0.25).astype(int).astype(str) + "_" + (lon / 0.25).astype(int).astype(str)
    df["grid_id_012"] = (lat / 0.125).astype(int).astype(str) + "_" + (lon / 0.125).astype(int).astype(str)
    
    # 2D Grid Helper
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
    "is_good_to_live", "reason_not_good_to_live", "downstream_risk_count"
]

TARGET_ENC_COLS = [
    "district", "grid_id", "downstream_sig", "downstream_quad_sig", "infra_deficit_sig",
    "landcover", "soil_type", "water_supply", "electricity", "road_quality",
    "downstream_risk_count", "grid_id_100", "grid_id_050", "grid_id_025", "grid_id_012"
]

COMPOSITE_ENC_COLS = []
STD_ENC_COLS = [
    "district", "downstream_sig"
]

IGNORE_COLS = DROP_COLS + [
    TARGET, "flood_occurrence_yes", "downstream_sig", "downstream_quad_sig", "infra_deficit_sig"
]
SPATIAL_HELPERS = ["lat_bin", "lon_bin", "grid_id", "grid_id_100", "grid_id_050", "grid_id_025", "grid_id_012"]
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

# -----------------------------------------------------------------
# 5. MODEL DEFINITIONS & BASE ESTIMATORS
# -----------------------------------------------------------------
MODEL_NAMES = [
    "XGB-MAE-1 (d7)",
    "CAT-MAE-1 (d5)",
    "CAT-MAE-2 (d5)",
    "CAT-RMSE (d5)",
    "LGB-MAE (d5)",
    "XGB-MAE-2 (d5)"
]

N_FOLDS     = 5
y           = train_df[TARGET]
GLOBAL_MEAN = float(y[train_df['is_pseudo'] == 0].mean())
GLOBAL_STD  = float(y[train_df['is_pseudo'] == 0].std())
GLOBAL_Q25  = float(y[train_df['is_pseudo'] == 0].quantile(0.25))
GLOBAL_Q75  = float(y[train_df['is_pseudo'] == 0].quantile(0.75))
GLOBAL_MEDIAN = float(y[train_df['is_pseudo'] == 0].median())

# GroupKFold on grid_id (spatial CV)
gkf = GroupKFold(n_splits=N_FOLDS)
groups = train_df['grid_id'].values

SMOOTHING   = 10
SMOOTHING_COMPOSITE = 15
y_arr = y.values
real_mask = train_df['is_pseudo'] == 0

original_y = y_arr[real_mask]
original_df = train_df[real_mask].reset_index(drop=True)
original_groups = original_df['grid_id'].values
gkf_l2 = GroupKFold(n_splits=N_FOLDS)

all_oof_stacked = np.zeros(len(original_y))
all_tst_stacked = []

print("\n" + "=" * 75)
print(f"  5-FOLD SPATIAL GROUP CV - SINGLE-SEED v57 PIPELINE")
print("=" * 75)

t_start_global = time.time()

# -----------------------------------------------------------------
# 6. L2 REGULARIZED CUSTOM METRIC-DRIVEN LEVEL-2 STACKER
# -----------------------------------------------------------------
# EV-prioritizing simulator weights from v42:
c_mae, c_rmse, c_ev = 0.392696, 0.875527, 0.406963

def fit_metric_stacker(X_meta, y_true, alpha=0.1):
    n_meta_features = X_meta.shape[1]
    
    def loss_fn(params):
        w = params[:n_meta_features]
        intercept = params[n_meta_features]
        pred = np.dot(X_meta, w) + intercept
        pred = np.clip(pred, 0.0, 1.0)
        
        mae = mean_absolute_error(y_true, pred)
        rmse = root_mean_squared_error(y_true, pred)
        ev = explained_variance_score(y_true, pred)
        
        # Target metric
        score = (c_mae * mae + c_rmse * rmse) * (1.0 + c_ev * (1.0 - ev))
        # L2 Penalty to preserve ensemble diversity
        reg = alpha * np.sum(w**2)
        return score + reg
    
    init_guess = np.zeros(n_meta_features)
    init_guess[:len(MODEL_NAMES)] = 1.0 / len(MODEL_NAMES)
    init_guess = np.append(init_guess, 0.0) # Intercept
    
    bounds = (
        [(0.0, None) for _ in range(len(MODEL_NAMES))] + 
        [(None, None)]
    )
    
    res = minimize(loss_fn, init_guess, bounds=bounds, method='L-BFGS-B')
    return res.x[:-1], res.x[-1]

# -----------------------------------------------------------------
# 7. TRAINING LOOP
# -----------------------------------------------------------------
oof_preds = {m: np.zeros(len(train_df)) for m in MODEL_NAMES}
tst_preds = {m: np.zeros(len(test_df))  for m in MODEL_NAMES}

for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, groups)):
    t0 = time.time()
    
    va_is_pseudo = train_df.iloc[va_idx]['is_pseudo'] == 1
    va_idx_clean = va_idx[~va_is_pseudo] if va_is_pseudo.any() else va_idx
    
    tr_rows = train_df.iloc[tr_idx].copy()
    va_rows = train_df.iloc[va_idx_clean].copy()

    # 1. Standard Target Encodings (mapping statistics from REAL training rows only)
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
            
    # Combine all features (Reverted to stable v45 feature core)
    FEATURES = BASE_FEATURES + te_features

    y_tr, y_va = tr_rows[TARGET], va_rows[TARGET]
    X_tr, X_va, X_te = tr_rows[FEATURES].copy(), va_rows[FEATURES].copy(), test_df[FEATURES].copy()

    # Dtype converters
    cat_cols = [c for c in CAT_FEATURES if c in FEATURES]
    
    def to_xgb_fmt(df):
        df = df.copy()
        for c in df.columns:
            if hasattr(df[c], "cat"):
                df[c] = df[c].cat.codes.astype("int32")
        return df

    def to_cat_fmt(df):
        df = df.copy()
        for c in cat_cols:
            if c in df.columns:
                df[c] = df[c].astype(str)
        return df

    X_tr_xgb = to_xgb_fmt(X_tr); X_va_xgb = to_xgb_fmt(X_va); X_te_xgb = to_xgb_fmt(X_te)
    X_tr_cat = to_cat_fmt(X_tr); X_va_cat = to_cat_fmt(X_va); X_te_cat = to_cat_fmt(X_te)

    for col in cat_cols:
        cdt = cat_dtype_map[col]
        X_tr[col] = X_tr[col].astype(str).astype(cdt)
        X_va[col] = X_va[col].astype(str).astype(cdt)
        X_te[col] = X_te[col].astype(str).astype(cdt)

    # --- Base Level-1 Model Training ---

    # 1. XGB-MAE-1 (d7) - Unconstrained, MAE loss
    xgb_m1 = xgb.XGBRegressor(
        n_estimators=4000, learning_rate=0.03, max_depth=7, min_child_weight=4,
        subsample=0.85, colsample_bytree=0.6, colsample_bylevel=0.6,
        reg_alpha=2.0, reg_lambda=4.0, gamma=0.1, max_delta_step=1,
        objective="reg:absoluteerror", eval_metric="mae", tree_method="hist",
        device=XGB_DEVICE,
        enable_categorical=False, early_stopping_rounds=100, random_state=SEED, n_jobs=-1
    )
    xgb_m1.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
    
    # 2. CAT-MAE-1 (d5)
    cat_m1 = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.03, depth=5, l2_leaf_reg=5.0,
        bagging_temperature=0.7, random_strength=2.0, border_count=254,
        loss_function="MAE", eval_metric="MAE", task_type=CB_TASK_TYPE,
        random_seed=SEED, max_ctr_complexity=2, verbose=False
    )
    cat_m1.fit(X_tr_cat, y_tr, cat_features=cat_cols, eval_set=(X_va_cat, y_va), early_stopping_rounds=150, verbose=False)

    # 3. CAT-MAE-2 (d5)
    cat_m2 = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.03, depth=5, l2_leaf_reg=12.0,
        bagging_temperature=0.4, random_strength=5.0, border_count=254,
        loss_function="MAE", eval_metric="MAE", task_type=CB_TASK_TYPE,
        random_seed=SEED + 1, max_ctr_complexity=2, verbose=False
    )
    cat_m2.fit(X_tr_cat, y_tr, cat_features=cat_cols, eval_set=(X_va_cat, y_va), early_stopping_rounds=150, verbose=False)

    # 4. CAT-RMSE (d5)
    cat_rmse = cb.CatBoostRegressor(
        iterations=4000, learning_rate=0.03, depth=5, l2_leaf_reg=8.0,
        bagging_temperature=0.6, random_strength=3.0, border_count=254,
        loss_function="RMSE", eval_metric="RMSE", task_type=CB_TASK_TYPE,
        random_seed=SEED + 2, max_ctr_complexity=2, verbose=False
    )
    cat_rmse.fit(X_tr_cat, y_tr, cat_features=cat_cols, eval_set=(X_va_cat, y_va), early_stopping_rounds=150, verbose=False)

    # 5. LGB-MAE (d5)
    lgb_m1 = lgb.LGBMRegressor(
        n_estimators=4000, learning_rate=0.03, num_leaves=31, max_depth=5,
        min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
        reg_alpha=2.0, reg_lambda=5.0, objective="regression_l1",
        random_state=SEED, n_jobs=-1, verbosity=-1
    )
    lgb_m1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], categorical_feature=cat_cols, callbacks=[lgb.early_stopping(150, verbose=False)])

    # 6. XGB-MAE-2 (d5) - Unconstrained, MAE loss
    xgb_m2 = xgb.XGBRegressor(
        n_estimators=4000, learning_rate=0.03, max_depth=5, min_child_weight=6,
        subsample=0.75, colsample_bytree=0.5, colsample_bylevel=0.8,
        reg_alpha=5.0, reg_lambda=10.0, gamma=0.2, max_delta_step=1,
        objective="reg:absoluteerror", eval_metric="mae", tree_method="hist",
        device=XGB_DEVICE,
        enable_categorical=False, early_stopping_rounds=100, random_state=SEED + 3, n_jobs=-1
    )
    xgb_m2.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)

    # --- Save Out-Of-Fold Predictions ---
    oof_preds["XGB-MAE-1 (d7)"][va_idx_clean] = xgb_m1.predict(X_va_xgb)
    oof_preds["CAT-MAE-1 (d5)"][va_idx_clean] = cat_m1.predict(X_va_cat)
    oof_preds["CAT-MAE-2 (d5)"][va_idx_clean] = cat_m2.predict(X_va_cat)
    oof_preds["CAT-RMSE (d5)"][va_idx_clean] = cat_rmse.predict(X_va_cat)
    oof_preds["LGB-MAE (d5)"][va_idx_clean] = lgb_m1.predict(X_va)
    oof_preds["XGB-MAE-2 (d5)"][va_idx_clean] = xgb_m2.predict(X_va_xgb)

    # --- Accumulate Test Predictions ---
    tst_preds["XGB-MAE-1 (d7)"] += xgb_m1.predict(X_te_xgb) / N_FOLDS
    tst_preds["CAT-MAE-1 (d5)"] += cat_m1.predict(X_te_cat) / N_FOLDS
    tst_preds["CAT-MAE-2 (d5)"] += cat_m2.predict(X_te_cat) / N_FOLDS
    tst_preds["CAT-RMSE (d5)"] += cat_rmse.predict(X_te_cat) / N_FOLDS
    tst_preds["LGB-MAE (d5)"]  += lgb_m1.predict(X_te) / N_FOLDS
    tst_preds["XGB-MAE-2 (d5)"] += xgb_m2.predict(X_te_xgb) / N_FOLDS

    print(f"      Fold {fold+1}/5 | XGB1_it={xgb_m1.best_iteration}  CAT1_it={cat_m1.best_iteration_}  CAT2_it={cat_m2.best_iteration_}  LGB_it={lgb_m1.best_iteration_}  [{time.time()-t0:.0f}s]")

# Construct Stacking Meta-Feature Matrices
oof_meta = np.column_stack([oof_preds[m][real_mask] for m in MODEL_NAMES])
tst_meta = np.column_stack([tst_preds[m] for m in MODEL_NAMES])

# Inner-Loop CV Grid Search for best custom L2 alpha
print("   [STACK] Running nested CV grid search for L2 alpha...")
best_alpha_l2, best_score_l2 = 0.1, np.inf
alphas_to_test = [0.001, 0.01, 0.1, 1.0, 10.0]

for alpha in alphas_to_test:
    oof_cv = np.zeros(len(original_y))
    for fold_l2, (tr_idx_l2, va_idx_l2) in enumerate(gkf_l2.split(original_df, original_y, original_groups)):
        w_cv, b_cv = fit_metric_stacker(oof_meta[tr_idx_l2], original_y[tr_idx_l2], alpha=alpha)
        oof_cv[va_idx_l2] = np.clip(np.dot(oof_meta[va_idx_l2], w_cv) + b_cv, 0.0, 1.0)
        
    cv_mae = mean_absolute_error(original_y, oof_cv)
    cv_rmse = root_mean_squared_error(original_y, oof_cv)
    cv_ev = explained_variance_score(original_y, oof_cv)
    cv_score = (c_mae * cv_mae + c_rmse * cv_rmse) * (1.0 + c_ev * (1.0 - cv_ev))
    
    if cv_score < best_score_l2:
        best_score_l2 = cv_score
        best_alpha_l2 = alpha
        
print(f"   [L2 SEARCH] Best alpha: {best_alpha_l2} (Nested CV Metric Score: {best_score_l2:.5f})")

# Level-2 OOF stacking using the best alpha found
oof_stacked = np.zeros(len(original_y))
for fold, (tr_idx, va_idx) in enumerate(gkf_l2.split(original_df, original_y, original_groups)):
    w_fold, b_fold = fit_metric_stacker(oof_meta[tr_idx], original_y[tr_idx], alpha=best_alpha_l2)
    oof_stacked[va_idx] = np.clip(np.dot(oof_meta[va_idx], w_fold) + b_fold, 0.0, 1.0)
    
# Fit final stacker on full predictions to predict test
w_final, b_final = fit_metric_stacker(oof_meta, original_y, alpha=best_alpha_l2)
tst_stacked = np.clip(np.dot(tst_meta, w_final) + b_final, 0.0, 1.0)

print(f"   [FINAL COEFFICIENTS]")
for i, name in enumerate(MODEL_NAMES):
    print(f"      {name:<18}: {w_final[i]:.4f}")
print(f"      {'Intercept':<18}: {b_final:.4f}")

# -----------------------------------------------------------------
# 7. GLOBAL ENSEMBLE RESULTS & METRICS
# -----------------------------------------------------------------
g_mae  = mean_absolute_error(original_y, oof_stacked)
g_rmse = root_mean_squared_error(original_y, oof_stacked)
g_ev   = explained_variance_score(original_y, oof_stacked)
g_lb   = (c_mae * g_mae + c_rmse * g_rmse) * (1.0 + c_ev * (1.0 - g_ev))

print("\n" + "=" * 75)
print("  GLOBAL OOF RESULTS (v57 - Raw Stacking)")
print("=" * 75)
print(f"      MAE            : {g_mae:.5f}")
print(f"      RMSE           : {g_rmse:.5f}")
print(f"      Explained Var. : {g_ev:.5f}")
print(f"      Est. LB Score  : {g_lb:.5f}")
print("=" * 75)

# Save Fold Report (aggregate results)
fold_results_all = []
for fold, (tr_idx, va_idx) in enumerate(gkf_l2.split(original_df, original_y, original_groups)):
    f_mae  = mean_absolute_error(original_y[va_idx], oof_stacked[va_idx])
    f_rmse = root_mean_squared_error(original_y[va_idx], oof_stacked[va_idx])
    f_ev   = explained_variance_score(original_y[va_idx], oof_stacked[va_idx])
    fold_results_all.append({"fold": fold+1, "MAE": f_mae, "RMSE": f_rmse, "EV": f_ev})

fold_report = pd.DataFrame(fold_results_all)
fold_report.to_csv("fold_report_v57.csv", index=False)
fold_report.to_csv("submissions/fold_report_v57.csv", index=False)
print(f"\n[DONE] Saved fold reports.")

submission = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": tst_stacked
})
submission.to_csv("submission_v57.csv", index=False)
submission.to_csv("submissions/submission_v57.csv", index=False)
print(f"[DONE] Saved submission_v57.csv ({len(submission)} rows)")

np.save("oof_v57.npy", oof_stacked)
np.save("submissions/oof_v57.npy", oof_stacked)
print(f"[DONE] Saved oof_v57.npy")

# -----------------------------------------------------------------
# 8. INTEGRATED POST-HOC GLOBAL POWER TRANSFORMATION
# -----------------------------------------------------------------
print("\n" + "=" * 75)
print("  POST-HOC GLOBAL POWER TRANSFORMATION OPTIMIZATION (v57)")
print("=" * 75)

# Fit global calibration parameters
def global_transform_loss(params):
    a, b, c = params
    pred = a * np.power(np.clip(oof_stacked, 1e-6, None), b) + c
    pred = np.clip(pred, 0.0, 1.0)
    
    mae = mean_absolute_error(original_y, pred)
    rmse = root_mean_squared_error(original_y, pred)
    ev = explained_variance_score(original_y, pred)
    
    return (c_mae * mae + c_rmse * rmse) * (1.0 + c_ev * (1.0 - ev))

initial_guess = [1.0, 1.0, 0.0]
bounds = [(0.5, 1.5), (0.5, 2.0), (-0.20, 0.20)]

res_glob = minimize(global_transform_loss, initial_guess, bounds=bounds, method='L-BFGS-B')
a_glob, b_glob, c_glob = res_glob.x
print(f"Global parameters: a={a_glob:.5f}, b={b_glob:.5f}, c={c_glob:.5f}")

# Compute globally calibrated predictions
final_oof = a_glob * np.power(np.clip(oof_stacked, 1e-6, None), b_glob) + c_glob
final_oof = np.clip(final_oof, 0.0, 1.0)

final_test_preds = a_glob * np.power(np.clip(tst_stacked, 1e-6, None), b_glob) + c_glob
final_test_preds = np.clip(final_test_preds, 0.0, 1.0)

opt_mae = mean_absolute_error(original_y, final_oof)
opt_rmse = root_mean_squared_error(original_y, final_oof)
opt_ev = explained_variance_score(original_y, final_oof)
opt_lb = (c_mae * opt_mae + c_rmse * opt_rmse) * (1.0 + c_ev * (1.0 - opt_ev))

print(f"\nOptimized OOF LB Score: {opt_lb:.5f}")
print(f"  MAE: {opt_mae:.5f}, RMSE: {opt_rmse:.5f}, EV: {opt_ev:.5f}")

np.save("oof_v57_optimized.npy", final_oof)
np.save("submissions/oof_v57_optimized.npy", final_oof)
print(f"[DONE] Saved oof_v57_optimized.npy")

# Save final calibrated test predictions
submission_opt = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": final_test_preds
})
submission_opt.to_csv("submission_v57_optimized.csv", index=False)
submission_opt.to_csv("submissions/submission_v57_optimized.csv", index=False)
print(f"[DONE] Saved submission_v57_optimized.csv ({len(submission_opt)} rows)")
print(f"  Optimized range  : [{final_test_preds.min():.4f}, {final_test_preds.max():.4f}]")
print(f"  Total Time       : {time.time() - t_start_global:.1f}s")
print("=" * 75)
