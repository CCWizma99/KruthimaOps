"""
ML Opsidian: Genesis v56 - EV-Objective, Conflict Resolution & Spatial Residuals
==============================================================================
Upgrades from v55:
1. Custom EV-MAE Objective on XGBoost models.
2. Pre-processing Conflict Resolution (target smoothing) for feature-duplicate contradictions.
3. Hierarchical Spatial Encoding Residuals (replacing raw fine grid target encodings with deltas).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer
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
QUANTILE_ALPHAS = [0.40, 0.45, 0.50, 0.55, 0.60]
USE_BLEND = True  # Enabled for v56
USE_ADVERSARIAL_WEIGHTS = False  # Toggle to correct spatial covariate shift

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
print("  ML OPSIDIAN v56 - EV-OBJECTIVE, CONFLICT RESOLUTION & SPATIAL RESIDUALS")
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
# 1.6. SEMI-SUPERVISED PSEUDO-LABELING (v56 priority)
# -----------------------------------------------------------------
print(f"\n[SEMI-SUPERVISED] Pseudo-Labeling configuration: USE_PSEUDO={USE_PSEUDO}")
train_df['is_pseudo'] = 0
test_df['is_pseudo'] = 0

if USE_PSEUDO:
    # Priority chain: v49 -> v48 -> v45 -> v42 -> v30
    pseudo_paths_to_try = [
        "submission_v49.csv", "submissions/submission_v49.csv",
        "submission_v48.csv", "submissions/submission_v48.csv",
        "submission_v45.csv", "submissions/submission_v45.csv",
        "submission_v42.csv", "submissions/submission_v42.csv",
        "submissions/submission_v30.csv", "submission_v30.csv"
    ]
    
    pseudo_path_used = None
    for path in pseudo_paths_to_try:
        if os.path.exists(path):
            pseudo_path_used = path
            break
            
    if pseudo_path_used:
        sub_blend = pd.read_csv(pseudo_path_used)
        test_pseudo = test_df.merge(sub_blend, on="record_id", how="left")
        
        # Soft pseudo-labeling of all test rows
        pseudo_rows = test_pseudo.copy()
        pseudo_rows['is_pseudo'] = 1
        
        print(f"   Added {len(pseudo_rows)} soft pseudo-labeled rows from {pseudo_path_used}.")
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
# 3. FEATURE ENGINEERING (v56 Additions)
# -----------------------------------------------------------------
print("\n[FEAT] Engineering features...")
district_elev_std = combined.groupby('district')['elevation_m'].std().to_dict()
landcover_mean_inundation = combined.groupby('landcover')['inundation_area_sqm'].mean().to_dict()
soil_infilt_map = {'Sandy': 0.8, 'Loamy': 0.6, 'Silty': 0.4, 'Clay': 0.2, 'Peaty': 0.1}
cyclone_districts = {'Batticaloa', 'Trincomalee', 'Ampara', 'Mullaitivu', 'Jaffna'}
wet_zone_districts = {'Colombo', 'Gampaha', 'Kalutara', 'Galle', 'Matara', 'Ratnapura', 'Kegalle'}

# Fit inundation quantiles globally for consistency
combined_inun = pd.concat([train_df['inundation_area_sqm'], test_df['inundation_area_sqm']])
inun_bins = np.unique(np.percentile(combined_inun[combined_inun > 0], np.linspace(0, 100, 11)))
if len(inun_bins) < 2:
    inun_bins = [0, 1e9]

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
    
    # [v56] Compound downstream district signature
    df['downstream_district_sig'] = df['downstream_sig'] + "_" + df['district'].fillna('missing').astype(str).str.strip()
    
    # [v56] Inundation quantile bucket feature
    df['inundation_qbin'] = pd.cut(df['inundation_area_sqm'], bins=[-np.inf] + list(inun_bins) + [np.inf], labels=False, duplicates='drop').fillna(-1).astype(int)
    df['inundation_sig'] = df['inundation_qbin'].astype(str) + "_" + df['flood_occurrence_current_event'].fillna('missing').astype(str).str.strip()
    
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
    "is_good_to_live", "reason_not_good_to_live", "downstream_risk_count",
    "downstream_district_sig", "inundation_sig"
]

TARGET_ENC_COLS = [
    "district", "grid_id", "downstream_sig", "downstream_quad_sig", "infra_deficit_sig",
    "landcover", "soil_type", "water_supply", "electricity", "road_quality",
    "downstream_risk_count", "grid_id_100", "grid_id_050", "grid_id_025", "grid_id_012",
    "inundation_sig"
]

COMPOSITE_ENC_COLS = ["downstream_district_sig"] # High smoothing
STD_ENC_COLS = [
    "district", "downstream_sig", "downstream_district_sig"
]

IGNORE_COLS = DROP_COLS + [
    TARGET, "flood_occurrence_yes", "downstream_sig", "downstream_quad_sig", "infra_deficit_sig",
    "downstream_district_sig", "inundation_sig"
]
SPATIAL_HELPERS = ["lat_bin", "lon_bin", "grid_id", "grid_id_100", "grid_id_050", "grid_id_025", "grid_id_012"]
BASE_FEATURES = [c for c in train_df.columns if c not in IGNORE_COLS and c not in SPATIAL_HELPERS]

# [v56] Pre-processing Conflict Resolution (Target Smoothing on duplicate physical features)
print("\n[PREP] Resolving target conflicts for duplicate physical/geographical features...")
downstream_cols = [
    "flood_occurrence_current_event", "is_good_to_live", "reason_not_good_to_live",
    "downstream_risk_count", "downstream_sig", "downstream_district_sig",
    "downstream_quad_sig", "confirmed_severe_risk", "no_flood_confirmed",
    "inundation_per_capita", "inundation_qbin", "inundation_sig", "confirmed_risk",
    "inundation_flood_interaction", "inundation_density_risk", "inundation_ratio",
    "flood_occurrence_yes"
]
conflict_key_cols = [c for c in BASE_FEATURES if c not in downstream_cols]
# Fill NaNs temporarily for grouping consistency
temp_df = train_df[conflict_key_cols].fillna(-999)
group_medians = temp_df.assign(TARGET=train_df[TARGET]).groupby(list(temp_df.columns))['TARGET'].transform('median')
train_df[TARGET] = group_medians
print(f"   Conflict resolution complete. Adjusted target risk scores across {len(train_df)} rows.")

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
numeric_feature_names = [c for c in BASE_FEATURES if c not in cat_feature_names]
print(f"   Base features: {len(BASE_FEATURES)}")

# -----------------------------------------------------------------
# 4.3. ADVERSARIAL VALIDATION SAMPLE WEIGHTING (COVARIATE SHIFT CORRECTION)
# -----------------------------------------------------------------
train_df['sample_weight'] = 1.0

if USE_ADVERSARIAL_WEIGHTS:
    print("\n[ADVERSARIAL] Computing adversarial validation sample weights...")
    from sklearn.ensemble import RandomForestClassifier
    
    # Isolate real train rows vs test rows
    real_mask = train_df['is_pseudo'] == 0
    adv_train = train_df[real_mask][numeric_feature_names].copy()
    adv_test  = test_df[numeric_feature_names].copy()
    
    adv_train['is_test'] = 0
    adv_test['is_test']  = 1
    
    adv_combined = pd.concat([adv_train, adv_test], ignore_index=True)
    X_adv = adv_combined[numeric_feature_names].fillna(0).values
    y_adv = adv_combined['is_test'].values
    
    # Train classifier to distinguish train vs test
    adv_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED, n_jobs=-1)
    
    # We use a KFold prediction scheme to get out-of-fold probability estimates for train rows
    from sklearn.model_selection import KFold
    kf_adv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    train_probs = np.zeros(len(adv_train))
    
    for tr_idx_a, va_idx_a in kf_adv.split(adv_train):
        X_tr_a = adv_train.iloc[tr_idx_a][numeric_feature_names].fillna(0).values
        y_tr_a = [0] * len(tr_idx_a)
        
        # Mix in test rows during training to learn the boundary
        X_tr_mixed = np.vstack([X_tr_a, adv_test[numeric_feature_names].fillna(0).values])
        y_tr_mixed = np.append(y_tr_a, [1] * len(adv_test))
        
        clf_fold = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED, n_jobs=-1)
        clf_fold.fit(X_tr_mixed, y_tr_mixed)
        
        X_va_a = adv_train.iloc[va_idx_a][numeric_feature_names].fillna(0).values
        train_probs[va_idx_a] = clf_fold.predict_proba(X_va_a)[:, 1]
    
    # Calculate density ratio weights: P(Test) / P(Train)
    # Clip to prevent extreme noise weights
    raw_weights = train_probs / (1.0 - train_probs + 1e-6)
    normalized_weights = raw_weights / np.mean(raw_weights)
    clipped_weights = np.clip(normalized_weights, 0.5, 3.0)
    
    train_df.loc[real_mask, 'sample_weight'] = clipped_weights
    print(f"   Adversarial weights computed. Range: [{clipped_weights.min():.4f}, {clipped_weights.max():.4f}]")

# -----------------------------------------------------------------
# 5. MODEL DEFINITIONS & BASE ESTIMATORS
# -----------------------------------------------------------------
# [v56] Renamed models to reflect custom EV-MAE objective
MODEL_NAMES = [
    "XGB-EV-MAE-1 (d7)",
    "CAT-MAE-1 (d5)",
    "CAT-MAE-2 (d5)",
    "CAT-RMSE (d5)",
    "LGB-MAE (d5)",
    "XGB-EV-MAE-2 (d5)",
    "CAT-Quantile-Median",
    "CAT-RankGauss (d5)"
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
print(f"  5-FOLD SPATIAL GROUP CV - SINGLE-SEED v56 PIPELINE")
print("=" * 75)

t_start_global = time.time()

# -----------------------------------------------------------------
# 5.2. LEAVE-ONE-OUT TARGET ENCODING HELPER (Leak-Free)
# -----------------------------------------------------------------
def loo_target_encode(tr_df, va_df, te_df, col, target, global_median, smoothing=10):
    # Only compute statistics on REAL training rows
    real_tr = tr_df[tr_df['is_pseudo'] == 0]
    stats = real_tr.groupby(col)[target].agg(['sum', 'count'])
    
    # Map stats to train rows and cast to float
    tr_sum = tr_df[col].map(stats['sum']).fillna(0).astype(float)
    tr_count = tr_df[col].map(stats['count']).fillna(0).astype(float)
    
    # For training rows: if it is a real row, exclude self
    # If it is a pseudo row, do not exclude (since it wasn't in stats)
    is_real = (tr_df['is_pseudo'] == 0).astype(float)
    
    # LOO calculation
    tr_loo = (tr_sum - is_real * tr_df[target]) / (tr_count - is_real).clip(lower=1)
    
    # Smooth toward global median
    tr_smooth = (tr_count * tr_loo + smoothing * global_median) / (tr_count + smoothing)
    
    # Val/test: use full stats from real training rows
    full_enc = (stats['sum'] + smoothing * global_median) / (stats['count'] + smoothing)
    
    tr_df[f'{col}_loo_enc'] = tr_smooth.fillna(global_median).astype(float)
    va_df[f'{col}_loo_enc'] = va_df[col].map(full_enc).fillna(global_median).astype(float)
    te_df[f'{col}_loo_enc'] = te_df[col].map(full_enc).fillna(global_median).astype(float)

# -----------------------------------------------------------------
# 5.5. CUSTOM EV-MAE OBJECTIVE FUNCTION (XGBoost)
# -----------------------------------------------------------------
def ev_mae_objective(y_true, y_pred, sample_weight=None):
    e = y_pred - y_true
    n = len(e)
    if sample_weight is not None:
        mean_e = np.average(e, weights=sample_weight)
    else:
        mean_e = np.mean(e)
    
    # 1. Smooth MAE (Pseudo-Huber)
    delta = 0.05
    grad_huber = e / np.sqrt(1.0 + (e / delta)**2)
    hess_huber = 1.0 / (1.0 + (e / delta)**2)**1.5
    
    # 2. MSE (gradient stabilizer)
    grad_mse = e
    hess_mse = np.ones_like(e)
    
    # 3. Residual Variance (Explained Variance driver)
    grad_var = (2.0 / n) * (e - mean_e)
    hess_var = np.full_like(e, 2.0 / n)
    
    w_huber, w_mse, w_var = 1.0, 0.05, 0.2
    
    grad = w_huber * grad_huber + w_mse * grad_mse + w_var * grad_var
    hess = w_huber * hess_huber + w_mse * hess_mse + w_var * hess_var
    
    if sample_weight is not None:
        grad *= sample_weight
        hess *= sample_weight
        
    return grad, hess

# -----------------------------------------------------------------
# 6. L2 REGULARIZED CUSTOM METRIC-DRIVEN LEVEL-2 STACKER
# -----------------------------------------------------------------
# Recalibrated simulator weights (17 points fit):
c_mae, c_rmse, c_ev = 0.535196, 1.146326, 0.054898

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
    # Give base predictions positive starting weight, others zero
    init_guess[:len(MODEL_NAMES)] = 1.0 / len(MODEL_NAMES)
    init_guess = np.append(init_guess, 0.0) # Intercept
    
    # base prediction weights (first 8 features) are bounded to be non-negative.
    # dispersion stats and intercept are unconstrained.
    n_base_models = len(MODEL_NAMES)
    n_unconstrained = n_meta_features - n_base_models
    
    bounds = (
        [(0.0, None) for _ in range(n_base_models)] + 
        [(None, None) for _ in range(n_unconstrained)] + 
        [(None, None)]
    )
    
    res = minimize(loss_fn, init_guess, bounds=bounds, method='L-BFGS-B')
    return res.x[:-1], res.x[-1]

# -----------------------------------------------------------------
# 7. TRAINING LOOP
# -----------------------------------------------------------------
oof_analytical = np.zeros(len(train_df))
tst_analytical_fold = np.zeros(len(test_df))

oof_preds = {m: np.zeros(len(train_df)) for m in MODEL_NAMES}
tst_preds = {m: np.zeros(len(test_df))  for m in MODEL_NAMES}

oof_iqr = np.zeros(len(train_df))
oof_skew = np.zeros(len(train_df))
tst_iqr = np.zeros(len(test_df))
tst_skew = np.zeros(len(test_df))

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
            
    # [v56] Compute Hierarchical Spatial Encoding Residuals
    for tgt_df in [tr_rows, va_rows, test_df]:
        tgt_df["grid_delta_050"] = tgt_df["grid_id_050_target_enc"] - tgt_df["grid_id_100_target_enc"]
        tgt_df["grid_delta_025"] = tgt_df["grid_id_025_target_enc"] - tgt_df["grid_id_050_target_enc"]
        tgt_df["grid_delta_012"] = tgt_df["grid_id_012_target_enc"] - tgt_df["grid_id_025_target_enc"]

    # Drop raw fine-grained spatial encodings (enc + q25 + q75 + cnt) from te_features list
    grid_cols_to_drop = [
        "grid_id_050_target_enc", "grid_id_050_target_q25", "grid_id_050_target_q75", "grid_id_050_target_cnt",
        "grid_id_025_target_enc", "grid_id_025_target_q25", "grid_id_025_target_q75", "grid_id_025_target_cnt",
        "grid_id_012_target_enc", "grid_id_012_target_q25", "grid_id_012_target_q75", "grid_id_012_target_cnt"
    ]
    te_features = [f for f in te_features if f not in grid_cols_to_drop]
    # Inject the spatial deltas into the feature matrix instead
    te_features.extend(["grid_delta_050", "grid_delta_025", "grid_delta_012"])
            
    # 2. Leave-One-Out (LOO) target encodings
    loo_target_encode(tr_rows, va_rows, test_df, 'downstream_sig', TARGET, GLOBAL_MEDIAN)
    loo_target_encode(tr_rows, va_rows, test_df, 'district',       TARGET, GLOBAL_MEDIAN)
    loo_target_encode(tr_rows, va_rows, test_df, 'downstream_district_sig', TARGET, GLOBAL_MEDIAN, smoothing=SMOOTHING_COMPOSITE)
    loo_features = ['downstream_sig_loo_enc', 'district_loo_enc', 'downstream_district_sig_loo_enc']

    # Combine all features (deduplicated defensively)
    FEATURES = list(dict.fromkeys(BASE_FEATURES + te_features + loo_features))

    y_tr, y_va = tr_rows[TARGET], va_rows[TARGET]
    X_tr, X_va, X_te = tr_rows[FEATURES].copy(), va_rows[FEATURES].copy(), test_df[FEATURES].copy()
    w_tr = tr_rows['sample_weight'].values

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

    # 1. XGB-EV-MAE-1 (d7) - Custom objective function [v56]
    xgb_m1 = xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=7, min_child_weight=4,
        subsample=0.85, colsample_bytree=0.6, colsample_bylevel=0.6,
        reg_alpha=2.0, reg_lambda=4.0, gamma=0.1, max_delta_step=1,
        objective=ev_mae_objective, eval_metric="mae", tree_method="hist",
        device=XGB_DEVICE,
        enable_categorical=False, early_stopping_rounds=100, random_state=SEED, n_jobs=-1
    )
    xgb_m1.fit(X_tr_xgb, y_tr, sample_weight=w_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
    
    # 2. CAT-MAE-1 (d5)
    cat_m1 = cb.CatBoostRegressor(
        iterations=800, learning_rate=0.05, depth=5, l2_leaf_reg=5.0,
        bagging_temperature=0.7, random_strength=2.0, border_count=254,
        loss_function="MAE", eval_metric="MAE", task_type=CB_TASK_TYPE,
        random_seed=SEED, max_ctr_complexity=2, verbose=False
    )
    cat_m1.fit(X_tr_cat, y_tr, sample_weight=w_tr, cat_features=cat_cols, eval_set=(X_va_cat, y_va), early_stopping_rounds=100, verbose=False)

    # 3. CAT-MAE-2 (d5)
    cat_m2 = cb.CatBoostRegressor(
        iterations=800, learning_rate=0.05, depth=5, l2_leaf_reg=12.0,
        bagging_temperature=0.4, random_strength=5.0, border_count=254,
        loss_function="MAE", eval_metric="MAE", task_type=CB_TASK_TYPE,
        random_seed=SEED + 1, max_ctr_complexity=2, verbose=False
    )
    cat_m2.fit(X_tr_cat, y_tr, sample_weight=w_tr, cat_features=cat_cols, eval_set=(X_va_cat, y_va), early_stopping_rounds=100, verbose=False)

    # 4. CAT-RMSE (d5)
    cat_rmse = cb.CatBoostRegressor(
        iterations=800, learning_rate=0.05, depth=5, l2_leaf_reg=8.0,
        bagging_temperature=0.6, random_strength=3.0, border_count=254,
        loss_function="RMSE", eval_metric="RMSE", task_type=CB_TASK_TYPE,
        random_seed=SEED + 2, max_ctr_complexity=2, verbose=False
    )
    cat_rmse.fit(X_tr_cat, y_tr, sample_weight=w_tr, cat_features=cat_cols, eval_set=(X_va_cat, y_va), early_stopping_rounds=100, verbose=False)

    # 5. LGB-MAE (d5)
    lgb_m1 = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.05, num_leaves=31, max_depth=5,
        min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
        reg_alpha=2.0, reg_lambda=5.0, objective="regression_l1",
        random_state=SEED, n_jobs=-1, verbosity=-1
    )
    lgb_m1.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(150, verbose=False)])

    # 6. XGB-EV-MAE-2 (d5) - Custom objective function [v56]
    xgb_m2 = xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=5, min_child_weight=6,
        subsample=0.75, colsample_bytree=0.5, colsample_bylevel=0.8,
        reg_alpha=5.0, reg_lambda=10.0, gamma=0.2, max_delta_step=1,
        objective=ev_mae_objective, eval_metric="mae", tree_method="hist",
        device=XGB_DEVICE,
        enable_categorical=False, early_stopping_rounds=100, random_state=SEED + 3, n_jobs=-1
    )
    xgb_m2.fit(X_tr_xgb, y_tr, sample_weight=w_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)

    # 7. Quantile Ensemble Base Model
    quantile_preds_va = []
    quantile_preds_te = []
    for alpha in QUANTILE_ALPHAS:
        qmodel = cb.CatBoostRegressor(
            iterations=800, learning_rate=0.05, depth=5, l2_leaf_reg=5.0,
            loss_function=f'Quantile:alpha={alpha}', eval_metric=f'Quantile:alpha={alpha}',
            task_type=CB_TASK_TYPE, thread_count=-1, random_seed=SEED, max_ctr_complexity=2, verbose=False
        )
        qmodel.fit(X_tr_cat, y_tr, sample_weight=w_tr, cat_features=cat_cols, eval_set=(X_va_cat, y_va), early_stopping_rounds=100, verbose=False)
        quantile_preds_va.append(qmodel.predict(X_va_cat))
        quantile_preds_te.append(qmodel.predict(X_te_cat))
        
    va_quantile_median = np.median(quantile_preds_va, axis=0)
    te_quantile_median = np.median(quantile_preds_te, axis=0)

    # 8. CAT-RankGauss (d5)
    # Fit target transformer strictly inside the fold on real training data labels only
    real_tr_y_arr = y_tr[tr_rows['is_pseudo'] == 0].values.reshape(-1, 1)
    qt = QuantileTransformer(output_distribution='normal', random_state=SEED)
    qt.fit(real_tr_y_arr)
    
    y_tr_trans = qt.transform(y_tr.values.reshape(-1, 1)).flatten()
    y_va_trans = qt.transform(y_va.values.reshape(-1, 1)).flatten()
    
    cat_rg = cb.CatBoostRegressor(
        iterations=800, learning_rate=0.05, depth=5, l2_leaf_reg=5.0,
        bagging_temperature=0.7, random_strength=2.0, border_count=254,
        loss_function="RMSE", eval_metric="RMSE", task_type=CB_TASK_TYPE,
        random_seed=SEED + 4, max_ctr_complexity=2, verbose=False
    )
    cat_rg.fit(X_tr_cat, y_tr_trans, sample_weight=w_tr, cat_features=cat_cols, eval_set=(X_va_cat, y_va_trans), early_stopping_rounds=100, verbose=False)
    
    rg_val_pred_trans = cat_rg.predict(X_va_cat)
    rg_te_pred_trans = cat_rg.predict(X_te_cat)
    
    # Inverse-transform predictions back to original risk range
    rg_val_pred = qt.inverse_transform(rg_val_pred_trans.reshape(-1, 1)).flatten()
    rg_te_pred = qt.inverse_transform(rg_te_pred_trans.reshape(-1, 1)).flatten()

    # --- Save Out-Of-Fold Predictions ---
    oof_preds["XGB-EV-MAE-1 (d7)"][va_idx_clean] = xgb_m1.predict(X_va_xgb)
    oof_preds["CAT-MAE-1 (d5)"][va_idx_clean] = cat_m1.predict(X_va_cat)
    oof_preds["CAT-MAE-2 (d5)"][va_idx_clean] = cat_m2.predict(X_va_cat)
    oof_preds["CAT-RMSE (d5)"][va_idx_clean] = cat_rmse.predict(X_va_cat)
    oof_preds["LGB-MAE (d5)"][va_idx_clean] = lgb_m1.predict(X_va)
    oof_preds["XGB-EV-MAE-2 (d5)"][va_idx_clean] = xgb_m2.predict(X_va_xgb)
    oof_preds["CAT-Quantile-Median"][va_idx_clean] = va_quantile_median
    oof_preds["CAT-RankGauss (d5)"][va_idx_clean] = rg_val_pred

    # --- Accumulate Test Predictions ---
    tst_preds["XGB-EV-MAE-1 (d7)"] += xgb_m1.predict(X_te_xgb) / N_FOLDS
    tst_preds["CAT-MAE-1 (d5)"] += cat_m1.predict(X_te_cat) / N_FOLDS
    tst_preds["CAT-MAE-2 (d5)"] += cat_m2.predict(X_te_cat) / N_FOLDS
    tst_preds["CAT-RMSE (d5)"] += cat_rmse.predict(X_te_cat) / N_FOLDS
    tst_preds["LGB-MAE (d5)"]  += lgb_m1.predict(X_te) / N_FOLDS
    tst_preds["XGB-EV-MAE-2 (d5)"] += xgb_m2.predict(X_te_xgb) / N_FOLDS
    tst_preds["CAT-Quantile-Median"] += te_quantile_median / N_FOLDS
    tst_preds["CAT-RankGauss (d5)"] += rg_te_pred / N_FOLDS

    # --- Quantile Dispersion Calculations (Dispersion Meta-Features) ---
    q_arr_va = np.array(quantile_preds_va)  # shape: (5, N_val)
    q_arr_te = np.array(quantile_preds_te)  # shape: (5, N_test)
    
    va_iqr = q_arr_va.max(axis=0) - q_arr_va.min(axis=0)
    te_iqr = q_arr_te.max(axis=0) - q_arr_te.min(axis=0)
    
    va_skew = q_arr_va.mean(axis=0) - va_quantile_median
    te_skew = q_arr_te.mean(axis=0) - te_quantile_median
    
    oof_iqr[va_idx_clean] = va_iqr
    oof_skew[va_idx_clean] = va_skew
    tst_iqr += te_iqr / N_FOLDS
    tst_skew += te_skew / N_FOLDS

    # --- Compute CV-Safe Analytical Group Medians ---
    real_tr_clean = tr_rows[tr_rows['is_pseudo'] == 0]
    sig_stats = real_tr_clean.groupby('downstream_sig')[TARGET].agg(['median', 'count'])
    smoothed_medians = ((sig_stats['count'] * sig_stats['median'] + 10 * GLOBAL_MEDIAN) / (sig_stats['count'] + 10))
    
    oof_analytical[va_idx_clean] = va_rows['downstream_sig'].map(smoothed_medians).fillna(GLOBAL_MEDIAN).values
    tst_analytical_fold += test_df['downstream_sig'].map(smoothed_medians).fillna(GLOBAL_MEDIAN).values / N_FOLDS

    print(f"      Fold {fold+1}/5 | XGB1_it={xgb_m1.best_iteration}  CAT1_it={cat_m1.best_iteration_}  CAT2_it={cat_m2.best_iteration_}  LGB_it={lgb_m1.best_iteration_}  [{time.time()-t0:.0f}s]")

# Construct Stacking Meta-Feature Matrices
oof_meta = np.column_stack([
    *[oof_preds[m][real_mask] for m in MODEL_NAMES],
    oof_iqr[real_mask],
    oof_skew[real_mask],
    oof_analytical[real_mask]
])

tst_meta = np.column_stack([
    *[tst_preds[m] for m in MODEL_NAMES],
    tst_iqr,
    tst_skew,
    tst_analytical_fold
])

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
print(f"      {'Quantile IQR':<18}: {w_final[-3]:.4f}")
print(f"      {'Quantile Skew':<18}: {w_final[-2]:.4f}")
print(f"      {'Analytical Median':<18}: {w_final[-1]:.4f}")
print(f"      {'Intercept':<18}: {b_final:.4f}")

# -----------------------------------------------------------------
# 7. GLOBAL ENSEMBLE RESULTS & METRICS
# -----------------------------------------------------------------
g_mae  = mean_absolute_error(original_y, oof_stacked)
g_rmse = root_mean_squared_error(original_y, oof_stacked)
g_ev   = explained_variance_score(original_y, oof_stacked)
g_lb   = (c_mae * g_mae + c_rmse * g_rmse) * (1.0 + c_ev * (1.0 - g_ev))

print("\n" + "=" * 75)
print("  GLOBAL OOF RESULTS (v56 - Raw Stacking)")
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
fold_report.to_csv("fold_report_v56.csv", index=False)
fold_report.to_csv("submissions/fold_report_v56.csv", index=False)
print(f"\n[DONE] Saved fold reports.")

submission = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": tst_stacked
})
submission.to_csv("submission_v56.csv", index=False)
submission.to_csv("submissions/submission_v56.csv", index=False)
print(f"[DONE] Saved submission_v56.csv ({len(submission)} rows)")

np.save("oof_v56.npy", oof_stacked)
np.save("submissions/oof_v56.npy", oof_stacked)
print(f"[DONE] Saved oof_v56.npy")

# -----------------------------------------------------------------
# 8. INTEGRATED FALLBACK-ENABLED PER-GROUP POWER TRANSFORMATION
# -----------------------------------------------------------------
print("\n" + "=" * 75)
print("  POST-HOC POWER TRANSFORMATION OPTIMIZATION (v56)")
print("=" * 75)

# Fit global calibration parameters first
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

# Initialize calibrated predictions with global parameters
opt_oof = a_glob * np.power(np.clip(oof_stacked, 1e-6, None), b_glob) + c_glob
opt_oof = np.clip(opt_oof, 0.0, 1.0)

opt_test_preds = a_glob * np.power(np.clip(tst_stacked, 1e-6, None), b_glob) + c_glob
opt_test_preds = np.clip(opt_test_preds, 0.0, 1.0)

# Set group labels for train and test
train_groups = train_df.loc[real_mask, 'downstream_sig'].values
test_groups = test_df['downstream_sig'].values

# Apply group-specific calibration override
unique_groups = np.unique(train_groups)
n_calibrated_groups = 0
for grp in unique_groups:
    tr_mask = train_groups == grp
    te_mask = test_groups == grp
    
    if tr_mask.sum() < 10: # Skip sparse groups (use global fallback)
        continue
        
    y_grp = original_y[tr_mask]
    p_grp = oof_stacked[tr_mask]
    
    def group_loss(params):
        a, b, c = params
        pred_cal = np.clip(a * np.power(np.clip(p_grp, 1e-6, None), b) + c, 0, 1)
        return mean_absolute_error(y_grp, pred_cal)
        
    res_grp = minimize(group_loss, x0=[a_glob, b_glob, c_glob],
                       bounds=[(0.5, 1.5), (0.5, 2.0), (-0.20, 0.20)],
                       method='L-BFGS-B')
    
    if res_grp.success:
        a_grp, b_grp, c_grp = res_grp.x
        opt_oof[tr_mask] = np.clip(a_grp * np.power(np.clip(p_grp, 1e-6, None), b_grp) + c_grp, 0, 1)
        if te_mask.any():
            opt_test_preds[te_mask] = np.clip(a_grp * np.power(np.clip(tst_stacked[te_mask], 1e-6, None), b_grp) + c_grp, 0, 1)
        n_calibrated_groups += 1

print(f"Group calibration complete. Overrode {n_calibrated_groups} groups out of {len(unique_groups)} total groups.")

# -----------------------------------------------------------------
# 9. DYNAMIC ANALYTICAL GROUP MEDIAN BLEND OPTIMIZATION
# -----------------------------------------------------------------
if USE_BLEND:
    print("\n" + "=" * 75)
    print("  DYNAMIC ANALYTICAL GROUP MEDIAN BLEND (v56)")
    print("=" * 75)
    
    oof_analytical_clean = oof_analytical[real_mask]
    
    def blend_loss(alpha_val):
        blend_pred = (1.0 - alpha_val[0]) * opt_oof + alpha_val[0] * oof_analytical_clean
        mae = mean_absolute_error(original_y, blend_pred)
        rmse = root_mean_squared_error(original_y, blend_pred)
        ev = explained_variance_score(original_y, blend_pred)
        return (c_mae * mae + c_rmse * rmse) * (1.0 + c_ev * (1.0 - ev))
    
    res_blend = minimize(blend_loss, x0=[0.20], bounds=[(0.0, 1.0)], method='L-BFGS-B')
    best_alpha = res_blend.x[0]
    print(f"Optimal BLEND_ALPHA: {best_alpha:.4f}")
    
    # Compute final blended predictions for both OOF and Test
    final_oof = (1.0 - best_alpha) * opt_oof + best_alpha * oof_analytical_clean
    final_oof = np.clip(final_oof, 0.0, 1.0)
    
    final_test_preds = (1.0 - best_alpha) * opt_test_preds + best_alpha * tst_analytical_fold
    final_test_preds = np.clip(final_test_preds, 0.0, 1.0)
else:
    print("\n[BLEND] Dynamic analytical blending is disabled. Using calibrated predictions directly.")
    final_oof = opt_oof
    final_test_preds = opt_test_preds

opt_mae = mean_absolute_error(original_y, final_oof)
opt_rmse = root_mean_squared_error(original_y, final_oof)
opt_ev = explained_variance_score(original_y, final_oof)
opt_lb = (c_mae * opt_mae + c_rmse * opt_rmse) * (1.0 + c_ev * (1.0 - opt_ev))

print(f"\nOptimized & Blended OOF LB Score: {opt_lb:.5f}")
print(f"  MAE: {opt_mae:.5f}, RMSE: {opt_rmse:.5f}, EV: {opt_ev:.5f}")

np.save("oof_v56_optimized.npy", final_oof)
np.save("submissions/oof_v56_optimized.npy", final_oof)
print(f"[DONE] Saved oof_v56_optimized.npy")

# Save final calibrated & blended test predictions
submission_opt = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": final_test_preds
})
submission_opt.to_csv("submission_v56_optimized.csv", index=False)
submission_opt.to_csv("submissions/submission_v56_optimized.csv", index=False)
print(f"[DONE] Saved submission_v56_optimized.csv ({len(submission_opt)} rows)")
print(f"  Optimized range  : [{final_test_preds.min():.4f}, {final_test_preds.max():.4f}]")
print(f"  Total Time       : {time.time() - t_start_global:.1f}s")
print("=" * 75)
