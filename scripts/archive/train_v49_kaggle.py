"""
ML Opsidian: Genesis v49 - Psychic Mode 🔮
===========================================
Built from v48 (LB 0.38245) with THREE targeted novel additions:

1. KNN Real-Row Consensus (7th Level-1 Base Model):
   - For each fold, train KNN on REAL training rows only (802 rows).
   - Predicts flood_risk_score from normalized numeric feature space.
   - Ground-truth label propagation from real Sri Lanka coords to synthetic test.

2. Monotonic Constraints on XGBoost (Domain-Physics Enforcement):
   - rainfall ↑ → risk ↑, elevation ↑ → risk ↓, distance_to_river ↑ → risk ↓
   - Prevents physically impossible gradient directions in sparse regions.
   - Applied to XGB-Huber and XGB-MAE-2 via monotone_constraints dict.

3. Geospatial Residual Kriging (Post-Hoc Spatial Error Correction):
   - After L2 stacking, fit KNN on (lat, lon) → OOF residuals.
   - Apply weighted spatial correction to test predictions.
   - Corrects spatially-correlated systematic model errors.

REMOVED: Multi-seed (single seed=42 only, per design decision).
RETAINED: Huber loss (fixed sklearn API signature from v48 hotfix).
RETAINED: v42 raw pseudo-labels, 16-point simulator, post-hoc power transform.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import warnings
import time
import os

DATA_DIR = "/kaggle/input/competitions/ml-opsidian-genesis-initial-round-26"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "data"

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------
SEED         = 42          # Single seed — no multi-seed
HUBER_DELTA  = 0.1
KRIGING_WEIGHT = 0.3       # Weight for spatial residual correction (tuned on OOF)
KRIGING_K      = 7         # K nearest neighbors for kriging

# Physical domain knowledge: feature → monotone direction
# +1 = feature ↑ → flood risk ↑, -1 = feature ↑ → flood risk ↓, 0 = unconstrained
MONOTONE_MAP = {
    'rainfall_7d_mm':           1,
    'rainfall_7d_mm_log1p':     1,
    'inundation_area_log':      1,
    'flood_occurrence_yes':     1,
    'confirmed_severe_risk':    1,
    'confirmed_risk':           1,
    'historical_flood_count':   1,
    'extreme_weather_index':    1,
    'inundation_flood_interaction': 1,
    'elevation_m':             -1,
    'elevation_m_yeojohnson':  -1,
    'distance_to_river_m':     -1,
    'distance_to_river_m_log1p': -1,
    'drainage_index':          -1,
    'drainage_index_yeojohnson': -1,
    'infrastructure_score':    -1,
    'ndvi_qmap':               -1,
    'infra_resilience':        -1,
    'slope_proxy':             -1,
}

print("=" * 75)
print("  ML OPSIDIAN v49 - PSYCHIC MODE 🔮")
print("=" * 75)
print(f"  Seed: {SEED}")
print(f"  Kriging weight: {KRIGING_WEIGHT}, K={KRIGING_K}")
print(f"  Monotone constraints: {len(MONOTONE_MAP)} features")

# -----------------------------------------------------------------
# CUSTOM HUBER LOSS FOR XGBOOST (sklearn API — y_true, y_pred)
# -----------------------------------------------------------------
def huber_loss(y_true, y_pred):
    """Custom Huber loss for XGBRegressor sklearn API.
    Signature is (y_true, y_pred) — NOT (y_pred, dtrain).
    Quadratic for |r| <= delta, linear for |r| > delta.
    """
    residual = y_pred - y_true
    delta = HUBER_DELTA
    abs_r = np.abs(residual)
    mask_small = abs_r <= delta
    grad = np.where(mask_small, residual, delta * np.sign(residual))
    hess = np.where(mask_small, np.ones_like(residual), np.zeros_like(residual))
    hess = np.clip(hess, 0.01, None)
    return grad, hess

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
    df['lat_decimal_len'] = df['latitude'].apply(
        lambda x: len(str(x).split('.')[1]) if '.' in str(x) and str(x).lower() != 'nan' else 0)
    df['lon_decimal_len'] = df['longitude'].apply(
        lambda x: len(str(x).split('.')[1]) if '.' in str(x) and str(x).lower() != 'nan' else 0)

# -----------------------------------------------------------------
# 1.6. SEMI-SUPERVISED PSEUDO-LABELING (v42 raw, no double-calibration)
# -----------------------------------------------------------------
print(f"\n[SEMI-SUPERVISED] Pseudo-Labeling (USE_PSEUDO=True, source=v42 raw)")
train_df['is_pseudo'] = 0
test_df['is_pseudo'] = 0

pseudo_path = "submission_v42.csv"
if not os.path.exists(pseudo_path):
    pseudo_path = "submissions/submission_v42.csv"

if os.path.exists(pseudo_path):
    sub_blend = pd.read_csv(pseudo_path)
    test_pseudo = test_df.merge(sub_blend, on="record_id", how="left")
    pseudo_rows = test_pseudo.copy()
    pseudo_rows['is_pseudo'] = 1
    print(f"   Added {len(pseudo_rows)} soft pseudo-labeled rows from v42 raw predictions.")
    train_df = pd.concat([train_df, pseudo_rows], ignore_index=True)
else:
    print("   [WARNING] submission_v42.csv not found. Falling back to v30...")
    for fallback in ["submission_v30.csv", "submissions/submission_v30.csv"]:
        if os.path.exists(fallback):
            sub_blend = pd.read_csv(fallback)
            test_pseudo = test_df.merge(sub_blend, on="record_id", how="left")
            pseudo_rows = test_pseudo.copy()
            pseudo_rows['is_pseudo'] = 1
            print(f"   Added {len(pseudo_rows)} soft pseudo-labeled rows from v30 (fallback).")
            train_df = pd.concat([train_df, pseudo_rows], ignore_index=True)
            break

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
district_elev_std       = combined.groupby('district')['elevation_m'].std().to_dict()
landcover_mean_inundation = combined.groupby('landcover')['inundation_area_sqm'].mean().to_dict()
soil_infilt_map         = {'Sandy': 0.8, 'Loamy': 0.6, 'Silty': 0.4, 'Clay': 0.2, 'Peaty': 0.1}
cyclone_districts       = {'Batticaloa', 'Trincomalee', 'Ampara', 'Mullaitivu', 'Jaffna'}
wet_zone_districts      = {'Colombo', 'Gampaha', 'Kalutara', 'Galle', 'Matara', 'Ratnapura', 'Kegalle'}

def engineer_features(df):
    df = df.copy()

    # Downstream features
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
    df['month'] = date_series.dt.month
    df['is_yala'] = df['month'].isin([5, 6, 7, 8, 9]).astype(int)
    df['is_maha'] = df['month'].isin([11, 12, 1]).astype(int)
    df['zone_code'] = df['district'].astype(str).map(lambda x: 1 if x in wet_zone_districts else 2)
    df['monsoon_impact'] = (df['rainfall_7d_mm'] * df['is_yala'] * (df['zone_code'] == 1).astype(int) +
                            df['rainfall_7d_mm'] * df['is_maha'] * (df['zone_code'] == 2).astype(int))
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

    ndwi_clip = df["ndwi_qmap"].clip(lower=0.0)
    ndvi_clip = df["ndvi_qmap"].clip(-1.0, 1.0).clip(lower=0.0)
    df["pooling_vulnerability"] = ndwi_clip * (1.0 - ndvi_clip)
    df["soil_drainage_saturation"] = df["soil_saturation_limit"] * (1.0 - df["drainage_index_yeojohnson"].clip(0.0, 1.0))

    lat = df["latitude"].fillna(df["latitude"].median())
    lon = df["longitude"].fillna(df["longitude"].median())

    df["grid_id_100"] = (lat / 1.0).astype(int).astype(str) + "_" + (lon / 1.0).astype(int).astype(str)
    df["grid_id_050"] = (lat / 0.5).astype(int).astype(str) + "_" + (lon / 0.5).astype(int).astype(str)
    df["grid_id_025"] = (lat / 0.25).astype(int).astype(str) + "_" + (lon / 0.25).astype(int).astype(str)
    df["grid_id_012"] = (lat / 0.125).astype(int).astype(str) + "_" + (lon / 0.125).astype(int).astype(str)

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
STD_ENC_COLS = ["district", "downstream_sig"]

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
NUMERIC_BASE_FEATS = [c for c in BASE_FEATURES if c not in cat_feature_names]
print(f"   Base features: {len(BASE_FEATURES)} ({len(NUMERIC_BASE_FEATS)} numeric, {len(cat_feature_names)} categorical)")

def to_cat_fmt(df):
    df = df.copy()
    for col in cat_feature_names:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

# -----------------------------------------------------------------
# 5. CROSS-VALIDATION SETUP
# -----------------------------------------------------------------
# v49: 7 base models (added KNN-RealRow)
MODEL_NAMES = [
    "XGB-Huber (d7)",       # Idea 2: + monotonic constraints
    "CAT-MAE-1 (d5)",
    "CAT-MAE-2 (d5)",
    "CAT-RMSE (d5)",
    "LGB-MAE (d5)",
    "XGB-MAE-2 (d5)",       # Idea 2: + monotonic constraints
    "KNN-RealRow",          # Idea 1: NEW — KNN real-row consensus
]

N_FOLDS      = 5
y            = train_df[TARGET]
GLOBAL_MEAN  = float(y[train_df['is_pseudo'] == 0].mean())
GLOBAL_STD   = float(y[train_df['is_pseudo'] == 0].std())
GLOBAL_Q25   = float(y[train_df['is_pseudo'] == 0].quantile(0.25))
GLOBAL_Q75   = float(y[train_df['is_pseudo'] == 0].quantile(0.75))
GLOBAL_MEDIAN = float(y[train_df['is_pseudo'] == 0].median())

gkf    = GroupKFold(n_splits=N_FOLDS)
groups = train_df['grid_id'].values

SMOOTHING           = 10
SMOOTHING_COMPOSITE = 15
y_arr     = y.values
real_mask = train_df['is_pseudo'] == 0

original_y      = y_arr[real_mask]
original_df     = train_df[real_mask].reset_index(drop=True)
original_groups = original_df['grid_id'].values
gkf_l2          = GroupKFold(n_splits=N_FOLDS)

oof_stacked_single = np.zeros(len(original_y))
tst_stacked_single = None

print("\n" + "=" * 75)
print(f"  5-FOLD SPATIAL GROUP CV — SEED {SEED} — v49 PSYCHIC PIPELINE 🔮")
print("=" * 75)

t_start_global = time.time()

# -----------------------------------------------------------------
# 6. LEVEL-2 CUSTOM METRIC STACKER (16-point simulator)
# -----------------------------------------------------------------
c_mae, c_rmse, c_ev = 0.583210, 1.122681, 0.045804

def fit_metric_stacker(X_meta, y_true, alpha=0.1):
    n_models = X_meta.shape[1]

    def loss_fn(params):
        w = params[:n_models]
        intercept = params[n_models]
        pred = np.dot(X_meta, w) + intercept
        pred = np.clip(pred, 0.0, 1.0)
        mae  = mean_absolute_error(y_true, pred)
        rmse = root_mean_squared_error(y_true, pred)
        ev   = explained_variance_score(y_true, pred)
        score = (c_mae * mae + c_rmse * rmse) * (1.0 + c_ev * (1.0 - ev))
        reg   = alpha * np.sum(w**2)
        return score + reg

    init_guess = np.append(np.ones(n_models) / n_models, 0.0)
    bounds     = [(0.0, None)] * n_models + [(None, None)]
    res = minimize(loss_fn, init_guess, bounds=bounds, method='L-BFGS-B')
    return res.x[:-1], res.x[-1]

# -----------------------------------------------------------------
# 7. TRAINING LOOP (Single seed=42)
# -----------------------------------------------------------------
print(f"\n==================== RUNNING SEED {SEED} ====================")
oof_preds = {m: np.zeros(len(train_df)) for m in MODEL_NAMES}
tst_preds = {m: np.zeros(len(test_df))  for m in MODEL_NAMES}

for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, groups)):
    t0 = time.time()

    va_is_pseudo  = train_df.iloc[va_idx]['is_pseudo'] == 1
    va_idx_clean  = va_idx[~va_is_pseudo] if va_is_pseudo.any() else va_idx

    tr_rows = train_df.iloc[tr_idx].copy()
    va_rows = train_df.iloc[va_idx_clean].copy()

    # ---- Target encodings (from REAL training rows only) ----
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

    # ---- Build monotone constraint vector aligned to FEATURES ----
    monotone_dict = {f: MONOTONE_MAP[f] for f in FEATURES if f in MONOTONE_MAP}

    y_tr, y_va     = tr_rows[TARGET], va_rows[TARGET]
    X_tr, X_va, X_te = tr_rows[FEATURES].copy(), va_rows[FEATURES].copy(), test_df[FEATURES].copy()

    for col in cat_feature_names:
        if col in FEATURES and col in cat_dtype_map:
            cdt = cat_dtype_map[col]
            X_tr[col] = X_tr[col].astype(str).astype(cdt)
            X_va[col] = X_va[col].astype(str).astype(cdt)
            X_te[col] = X_te[col].astype(str).astype(cdt)

    X_tr_cat = to_cat_fmt(X_tr)
    X_va_cat = to_cat_fmt(X_va)
    X_te_cat = to_cat_fmt(X_te)

    cat_pool_tr = cb.Pool(X_tr_cat, y_tr, cat_features=cat_feature_names)
    cat_pool_va = cb.Pool(X_va_cat, y_va, cat_features=cat_feature_names)

    # === 1. XGBoost-Huber (d7) + MONOTONIC CONSTRAINTS (Idea 2) ===
    xgb_huber = xgb.XGBRegressor(
        n_estimators=3000, learning_rate=0.05, max_depth=7,
        objective=huber_loss,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.75,
        tree_method="hist", early_stopping_rounds=100,
        random_state=SEED, n_jobs=-1,
        eval_metric='mae', enable_categorical=True, max_delta_step=1,
        disable_default_eval_metric=True,
        monotone_constraints=monotone_dict,        # <-- PSYCHIC MODE: domain physics
    )
    xgb_huber.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds["XGB-Huber (d7)"][va_idx_clean] = xgb_huber.predict(X_va)
    tst_preds["XGB-Huber (d7)"] += xgb_huber.predict(X_te) / N_FOLDS

    # === 2. CatBoost-MAE-1 (d5) ===
    cat_mae1 = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.03, depth=5,
        l2_leaf_reg=5, loss_function="MAE", eval_metric="MAE",
        max_ctr_complexity=2, random_seed=SEED, verbose=False
    )
    cat_mae1.fit(cat_pool_tr, eval_set=cat_pool_va, early_stopping_rounds=100, verbose=False)
    oof_preds["CAT-MAE-1 (d5)"][va_idx_clean] = cat_mae1.predict(X_va_cat)
    tst_preds["CAT-MAE-1 (d5)"] += cat_mae1.predict(X_te_cat) / N_FOLDS

    # === 3. CatBoost-MAE-2 (d5) ===
    cat_mae2 = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.05, depth=5,
        l2_leaf_reg=5, loss_function="MAE", eval_metric="MAE",
        max_ctr_complexity=2, random_seed=SEED + 100, verbose=False
    )
    cat_mae2.fit(cat_pool_tr, eval_set=cat_pool_va, early_stopping_rounds=100, verbose=False)
    oof_preds["CAT-MAE-2 (d5)"][va_idx_clean] = cat_mae2.predict(X_va_cat)
    tst_preds["CAT-MAE-2 (d5)"] += cat_mae2.predict(X_te_cat) / N_FOLDS

    # === 4. CatBoost-RMSE (d5) ===
    cat_rmse = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.03, depth=5,
        l2_leaf_reg=5, loss_function="RMSE", eval_metric="RMSE",
        max_ctr_complexity=2, random_seed=SEED + 200, verbose=False
    )
    cat_rmse.fit(cat_pool_tr, eval_set=cat_pool_va, early_stopping_rounds=100, verbose=False)
    oof_preds["CAT-RMSE (d5)"][va_idx_clean] = cat_rmse.predict(X_va_cat)
    tst_preds["CAT-RMSE (d5)"] += cat_rmse.predict(X_te_cat) / N_FOLDS

    # === 5. LightGBM-MAE (d5) ===
    lgb_mae = lgb.LGBMRegressor(
        n_estimators=5000, learning_rate=0.03, num_leaves=15, max_depth=5,
        objective='regression_l1', random_state=SEED, n_jobs=-1, verbosity=-1
    )
    lgb_mae.fit(X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_preds["LGB-MAE (d5)"][va_idx_clean] = lgb_mae.predict(X_va)
    tst_preds["LGB-MAE (d5)"] += lgb_mae.predict(X_te) / N_FOLDS

    # === 6. XGBoost-MAE-2 (d5) + MONOTONIC CONSTRAINTS (Idea 2) ===
    xgb_mae2 = xgb.XGBRegressor(
        n_estimators=3000, learning_rate=0.05, max_depth=5,
        objective='reg:absoluteerror',
        min_child_weight=3, subsample=0.8, colsample_bytree=0.5,
        tree_method="hist", early_stopping_rounds=100,
        random_state=SEED + 300, n_jobs=-1,
        eval_metric='mae', enable_categorical=True, max_delta_step=1,
        monotone_constraints=monotone_dict,        # <-- PSYCHIC MODE: domain physics
    )
    xgb_mae2.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds["XGB-MAE-2 (d5)"][va_idx_clean] = xgb_mae2.predict(X_va)
    tst_preds["XGB-MAE-2 (d5)"] += xgb_mae2.predict(X_te) / N_FOLDS

    # === 7. KNN Real-Row Consensus (Idea 1 — NEW) ===
    real_tr_knn = tr_rows[tr_rows['is_pseudo'] == 0].copy()
    X_real_num = real_tr_knn[NUMERIC_BASE_FEATS].fillna(0).values
    y_real_knn = real_tr_knn[TARGET].values

    knn_scaler = StandardScaler()
    X_real_scaled = knn_scaler.fit_transform(X_real_num)
    X_va_scaled   = knn_scaler.transform(X_va[NUMERIC_BASE_FEATS].fillna(0).values)
    X_te_scaled   = knn_scaler.transform(X_te[NUMERIC_BASE_FEATS].fillna(0).values)

    knn_consensus = KNeighborsRegressor(n_neighbors=7, weights='distance', metric='euclidean')
    knn_consensus.fit(X_real_scaled, y_real_knn)

    oof_preds["KNN-RealRow"][va_idx_clean] = np.clip(knn_consensus.predict(X_va_scaled), 0.0, 1.0)
    tst_preds["KNN-RealRow"] += np.clip(knn_consensus.predict(X_te_scaled), 0.0, 1.0) / N_FOLDS

    # ---- Per-fold ensemble diagnostic ----
    oof_avg_fold = np.mean([oof_preds[m][va_idx_clean] for m in MODEL_NAMES], axis=0)
    f_mae  = mean_absolute_error(y_va.values, oof_avg_fold)
    f_rmse = root_mean_squared_error(y_va.values, oof_avg_fold)
    f_ev   = explained_variance_score(y_va.values, oof_avg_fold)
    hub_it = xgb_huber.best_iteration if hasattr(xgb_huber, 'best_iteration') else '?'
    print(f"      Fold {fold+1}/{N_FOLDS} | HUB_it={hub_it:<4} CAT1_it={cat_mae1.best_iteration_:<4} "
          f"CAT2_it={cat_mae2.best_iteration_:<4} CAT3_it={cat_rmse.best_iteration_:<4} "
          f"LGB_it={lgb_mae.best_iteration_:<4} XGB2_it={xgb_mae2.best_iteration:<4} "
          f"| [ENS MAE={f_mae:.4f} EV={f_ev:.4f}] [{time.time() - t0:.0f}s]")

# -----------------------------------------------------------------
# 8. LEVEL-2 STACKING
# -----------------------------------------------------------------
oof_meta = np.column_stack([oof_preds[m][real_mask] for m in MODEL_NAMES])
tst_meta = np.column_stack([tst_preds[m] for m in MODEL_NAMES])

# Nested CV grid search for best L2 alpha
print("   [STACK] Running nested CV grid search for L2 alpha...")
best_alpha, best_score = 0.1, np.inf
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
    oof_cv = np.zeros(len(original_y))
    for fold_l2, (tr_idx_l2, va_idx_l2) in enumerate(gkf_l2.split(original_df, original_y, original_groups)):
        w_cv, b_cv = fit_metric_stacker(oof_meta[tr_idx_l2], original_y[tr_idx_l2], alpha=alpha)
        oof_cv[va_idx_l2] = np.clip(np.dot(oof_meta[va_idx_l2], w_cv) + b_cv, 0.0, 1.0)
    cv_mae  = mean_absolute_error(original_y, oof_cv)
    cv_rmse = root_mean_squared_error(original_y, oof_cv)
    cv_ev   = explained_variance_score(original_y, oof_cv)
    cv_score = (c_mae * cv_mae + c_rmse * cv_rmse) * (1.0 + c_ev * (1.0 - cv_ev))
    if cv_score < best_score:
        best_score = cv_score
        best_alpha = alpha
print(f"   [L2 SEARCH] Best alpha: {best_alpha} (Nested CV Score: {best_score:.5f})")

# Level-2 OOF stacking with best alpha
oof_stacked_single = np.zeros(len(original_y))
for fold, (tr_idx, va_idx) in enumerate(gkf_l2.split(original_df, original_y, original_groups)):
    w_fold, b_fold = fit_metric_stacker(oof_meta[tr_idx], original_y[tr_idx], alpha=best_alpha)
    oof_stacked_single[va_idx] = np.clip(np.dot(oof_meta[va_idx], w_fold) + b_fold, 0.0, 1.0)

# Fit final stacker on full predictions for test
w_final, b_final = fit_metric_stacker(oof_meta, original_y, alpha=best_alpha)
tst_stacked_single = np.clip(np.dot(tst_meta, w_final) + b_final, 0.0, 1.0)

print(f"\n   [FINAL STACKER COEFFICIENTS]")
for i, name in enumerate(MODEL_NAMES):
    print(f"      {name:<20}: {w_final[i]:.4f}")
print(f"      {'Intercept':<20}: {b_final:.4f}")

# -----------------------------------------------------------------
# 9. RAW STACKING RESULTS
# -----------------------------------------------------------------
g_mae  = mean_absolute_error(original_y, oof_stacked_single)
g_rmse = root_mean_squared_error(original_y, oof_stacked_single)
g_ev   = explained_variance_score(original_y, oof_stacked_single)
g_lb   = (c_mae * g_mae + c_rmse * g_rmse) * (1.0 + c_ev * (1.0 - g_ev))

print("\n" + "=" * 75)
print("  GLOBAL OOF RESULTS (v49 — Raw Stacking)")
print("=" * 75)
print(f"      MAE            : {g_mae:.5f}")
print(f"      RMSE           : {g_rmse:.5f}")
print(f"      Explained Var. : {g_ev:.5f}")
print(f"      Est. LB Score  : {g_lb:.5f}")
print("=" * 75)

# Save fold report
fold_results_raw = []
for fold, (tr_idx, va_idx) in enumerate(gkf_l2.split(original_df, original_y, original_groups)):
    f_mae  = mean_absolute_error(original_y[va_idx], oof_stacked_single[va_idx])
    f_rmse = root_mean_squared_error(original_y[va_idx], oof_stacked_single[va_idx])
    f_ev   = explained_variance_score(original_y[va_idx], oof_stacked_single[va_idx])
    fold_results_raw.append({"fold": fold+1, "MAE": f_mae, "RMSE": f_rmse, "EV": f_ev})

fold_report = pd.DataFrame(fold_results_raw)
for p in ["fold_report_v49.csv", "submissions/fold_report_v49.csv"]:
    fold_report.to_csv(p, index=False)

tst_raw = np.clip(tst_stacked_single, 0.0, 1.0)
sub_raw = pd.DataFrame({"record_id": test_df[ID_COL], "flood_risk_score": tst_raw})
for p in ["submission_v49.csv", "submissions/submission_v49.csv"]:
    sub_raw.to_csv(p, index=False)
print(f"[DONE] Saved submission_v49.csv ({len(sub_raw)} rows)")

for p in ["oof_v49.npy", "submissions/oof_v49.npy"]:
    np.save(p, oof_stacked_single)
print(f"[DONE] Saved oof_v49.npy")

# -----------------------------------------------------------------
# 10. IDEA 3: GEOSPATIAL RESIDUAL KRIGING 🔮
# -----------------------------------------------------------------
print("\n" + "=" * 75)
print("  GEOSPATIAL RESIDUAL KRIGING (Idea 3)")
print("=" * 75)

train_coords_krig = original_df[['latitude', 'longitude']].fillna(0).values
test_coords_krig  = test_df[['latitude', 'longitude']].fillna(0).values
residuals_krig    = original_y - oof_stacked_single

print(f"   Residual stats: mean={residuals_krig.mean():.5f}, std={residuals_krig.std():.5f}, "
      f"max_abs={np.abs(residuals_krig).max():.5f}")

# Fold-isolated OOF kriging correction (no self-leakage)
kriged_oof_correction = np.zeros(len(original_y))
for fold, (tr_idx, va_idx) in enumerate(gkf_l2.split(original_df, original_y, original_groups)):
    knn_krig = KNeighborsRegressor(n_neighbors=KRIGING_K, weights='distance')
    knn_krig.fit(train_coords_krig[tr_idx], residuals_krig[tr_idx])
    kriged_oof_correction[va_idx] = knn_krig.predict(train_coords_krig[va_idx])

# Full kriging for test (fit on all OOF residuals)
knn_krig_full = KNeighborsRegressor(n_neighbors=KRIGING_K, weights='distance')
knn_krig_full.fit(train_coords_krig, residuals_krig)
kriged_test_correction = knn_krig_full.predict(test_coords_krig)

print(f"   Test kriging correction: mean={kriged_test_correction.mean():.5f}, "
      f"std={kriged_test_correction.std():.5f}")

# Evaluate kriging at different weights on OOF
print(f"\n   [KRIGING WEIGHT SEARCH] Evaluating correction weights on OOF...")
best_krig_weight, best_krig_lb = 0.0, g_lb
for kw in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    krig_oof_cand = np.clip(oof_stacked_single + kw * kriged_oof_correction, 0.0, 1.0)
    kw_mae  = mean_absolute_error(original_y, krig_oof_cand)
    kw_rmse = root_mean_squared_error(original_y, krig_oof_cand)
    kw_ev   = explained_variance_score(original_y, krig_oof_cand)
    kw_lb   = (c_mae * kw_mae + c_rmse * kw_rmse) * (1.0 + c_ev * (1.0 - kw_ev))
    flag    = " ✅ BETTER" if kw_lb < best_krig_lb else ""
    print(f"      w={kw:.2f} → LB={kw_lb:.5f} MAE={kw_mae:.5f} RMSE={kw_rmse:.5f} EV={kw_ev:.5f}{flag}")
    if kw_lb < best_krig_lb:
        best_krig_lb = kw_lb
        best_krig_weight = kw

print(f"\n   Best kriging weight: {best_krig_weight} (OOF LB: {best_krig_lb:.5f} vs raw: {g_lb:.5f})")

# Apply best kriging weight
kriged_oof = np.clip(oof_stacked_single + best_krig_weight * kriged_oof_correction, 0.0, 1.0)
kriged_tst = np.clip(tst_raw + best_krig_weight * kriged_test_correction, 0.0, 1.0)

krig_mae  = mean_absolute_error(original_y, kriged_oof)
krig_rmse = root_mean_squared_error(original_y, kriged_oof)
krig_ev   = explained_variance_score(original_y, kriged_oof)
krig_lb   = (c_mae * krig_mae + c_rmse * krig_rmse) * (1.0 + c_ev * (1.0 - krig_ev))

print(f"\n   [KRIGED OOF] MAE={krig_mae:.5f} RMSE={krig_rmse:.5f} EV={krig_ev:.5f} LB={krig_lb:.5f}")

sub_kriged = pd.DataFrame({"record_id": test_df[ID_COL], "flood_risk_score": kriged_tst})
for p in ["submission_v49_kriged.csv", "submissions/submission_v49_kriged.csv"]:
    sub_kriged.to_csv(p, index=False)
print(f"[DONE] Saved submission_v49_kriged.csv")

for p in ["oof_v49_kriged.npy", "submissions/oof_v49_kriged.npy"]:
    np.save(p, kriged_oof)

# -----------------------------------------------------------------
# 11. POST-HOC POWER TRANSFORMATION
# -----------------------------------------------------------------
print("\n" + "=" * 75)
print("  POST-HOC POWER TRANSFORMATION OPTIMIZATION (v49)")
print("=" * 75)

def transform_loss(params, oof_src):
    a, b, c = params
    pred = a * np.power(np.clip(oof_src, 1e-6, None), b) + c
    pred = np.clip(pred, 0.0, 1.0)
    mae  = mean_absolute_error(original_y, pred)
    rmse = root_mean_squared_error(original_y, pred)
    ev   = explained_variance_score(original_y, pred)
    return (c_mae * mae + c_rmse * rmse) * (1.0 + c_ev * (1.0 - ev))

bounds_transform = [(0.5, 1.5), (0.5, 2.0), (-0.15, 0.15)]

# Optimize on raw stacked OOF
res_raw = minimize(lambda p: transform_loss(p, oof_stacked_single),
                   [1.0, 1.0, 0.0], bounds=bounds_transform, method='L-BFGS-B')
a_r, b_r, c_r = res_raw.x

opt_oof_raw = np.clip(a_r * np.power(np.clip(oof_stacked_single, 1e-6, None), b_r) + c_r, 0.0, 1.0)
opt_tst_raw = np.clip(a_r * np.power(np.clip(tst_raw, 1e-6, None), b_r) + c_r, 0.0, 1.0)

opt_mae  = mean_absolute_error(original_y, opt_oof_raw)
opt_rmse = root_mean_squared_error(original_y, opt_oof_raw)
opt_ev   = explained_variance_score(original_y, opt_oof_raw)
opt_lb   = (c_mae * opt_mae + c_rmse * opt_rmse) * (1.0 + c_ev * (1.0 - opt_ev))

print(f"Raw → Optimized: a={a_r:.5f} b={b_r:.5f} c={c_r:.5f}")
print(f"   MAE={opt_mae:.5f} RMSE={opt_rmse:.5f} EV={opt_ev:.5f} LB={opt_lb:.5f}")

sub_opt = pd.DataFrame({"record_id": test_df[ID_COL], "flood_risk_score": opt_tst_raw})
for p in ["submission_v49_optimized.csv", "submissions/submission_v49_optimized.csv"]:
    sub_opt.to_csv(p, index=False)
print(f"[DONE] Saved submission_v49_optimized.csv")

for p in ["oof_v49_optimized.npy", "submissions/oof_v49_optimized.npy"]:
    np.save(p, opt_oof_raw)

# Optimize on kriged OOF (if kriging helped)
if best_krig_weight > 0.0:
    res_krig = minimize(lambda p: transform_loss(p, kriged_oof),
                        [1.0, 1.0, 0.0], bounds=bounds_transform, method='L-BFGS-B')
    a_k, b_k, c_k = res_krig.x

    opt_oof_krig = np.clip(a_k * np.power(np.clip(kriged_oof, 1e-6, None), b_k) + c_k, 0.0, 1.0)
    opt_tst_krig = np.clip(a_k * np.power(np.clip(kriged_tst, 1e-6, None), b_k) + c_k, 0.0, 1.0)

    ok_mae  = mean_absolute_error(original_y, opt_oof_krig)
    ok_rmse = root_mean_squared_error(original_y, opt_oof_krig)
    ok_ev   = explained_variance_score(original_y, opt_oof_krig)
    ok_lb   = (c_mae * ok_mae + c_rmse * ok_rmse) * (1.0 + c_ev * (1.0 - ok_ev))

    print(f"\nKriged → Optimized: a={a_k:.5f} b={b_k:.5f} c={c_k:.5f}")
    print(f"   MAE={ok_mae:.5f} RMSE={ok_rmse:.5f} EV={ok_ev:.5f} LB={ok_lb:.5f}")

    sub_ok = pd.DataFrame({"record_id": test_df[ID_COL], "flood_risk_score": opt_tst_krig})
    for p in ["submission_v49_kriged_optimized.csv", "submissions/submission_v49_kriged_optimized.csv"]:
        sub_ok.to_csv(p, index=False)
    print(f"[DONE] Saved submission_v49_kriged_optimized.csv")

    for p in ["oof_v49_kriged_optimized.npy", "submissions/oof_v49_kriged_optimized.npy"]:
        np.save(p, opt_oof_krig)

# -----------------------------------------------------------------
# 12. FINAL SUMMARY
# -----------------------------------------------------------------
print("\n" + "=" * 75)
print("  v49 PSYCHIC MODE — FINAL SUMMARY")
print("=" * 75)
print(f"  Raw Stack          : MAE={g_mae:.5f}  RMSE={g_rmse:.5f}  EV={g_ev:.5f}  LB={g_lb:.5f}")
if best_krig_weight > 0.0:
    print(f"  + Kriging (w={best_krig_weight:.2f})  : MAE={krig_mae:.5f}  RMSE={krig_rmse:.5f}  EV={krig_ev:.5f}  LB={krig_lb:.5f}")
print(f"  + Power Calib.     : MAE={opt_mae:.5f}  RMSE={opt_rmse:.5f}  EV={opt_ev:.5f}  LB={opt_lb:.5f}")
if best_krig_weight > 0.0:
    print(f"  + Krig + Calib.    : MAE={ok_mae:.5f}  RMSE={ok_rmse:.5f}  EV={ok_ev:.5f}  LB={ok_lb:.5f}")
print(f"  Total Time         : {time.time() - t_start_global:.1f}s")
print("=" * 75)
print(f"\n  Outputs generated:")
print(f"    submission_v49.csv                 (raw)")
if best_krig_weight > 0.0:
    print(f"    submission_v49_kriged.csv          (+ spatial kriging)")
print(f"    submission_v49_optimized.csv       (+ power calibration)")
if best_krig_weight > 0.0:
    print(f"    submission_v49_kriged_optimized.csv (+ both)")
