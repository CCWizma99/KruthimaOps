"""
ML Opsidian: Genesis vmav2 - vmal + Adversarial Drop v2 (8 additional shift features)
==================================================================================
Base: vmal (v703 + Custom Imputation + Adversarial Drop v1, LB 0.38203)

Single Change vs. vmal:
  ADVERSARIAL DROP v2 — 8 additional features identified by Pass #2 adversarial
  validation (AUC=0.65277 on the post-vmal feature set).

  Dropped features (by category):
  A. Opaque pre-computed synthetic indices (pure shift sources, no engineering rationale):
     - seasonal_index         (importance: 262.8)
     - terrain_roughness_index(importance: 232.6)
     - socioeconomic_status_index (importance: 214.6)
  B. Engineered derivatives that inherit shift from their shifted raw parents:
     - drainage_deficit        (importance: 262.4) — rainfall * (1 - drainage_yj)
     - soil_drainage_saturation(importance: 249.2) — soil_sat * (1 - drainage_yj)
     - rain_spike_ratio        (importance: 240.0) — rainfall_7d / monthly_rainfall
     - infra_resilience        (importance: 215.0) — infra_score / pop_density
     - urban_runoff_potential  (importance: 195.8) — rainfall * built_up / drainage

  NOT dropped (despite high adversarial importance):
     - latitude / longitude — irreplaceable spatial signal; dropping collapses EV
     - freq features — v703 peak innovation; computed on combined data
     - slope_proxy — elevation shift is low (rank 60); signal worth keeping

All other vmal settings unchanged:
  - 5 seeds, 5-fold spatial GroupKFold, v703 pseudo-labels
  - Stable stacker coefficients (c_mae=0.539328, c_rmse=1.152263, c_ev=0.048467)
  - True eval coefficients (c_mae=0.544177, c_rmse=1.148953, c_ev=0.049625, c_int=-0.000580)
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
    DATA_DIR = "data"  # Fallback local

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------
USE_PSEUDO = True
SEEDS = [42]  # Single-seed probe — fast turnaround to test CAT-HUBER slot

print("=" * 75)
print("  ML OPSIDIAN vmav2h_s1 - ADVERSARIAL DROP v2 + CAT-HUBER (SINGLE SEED)")
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
# 1.4. VALUE FREQUENCY COUNT FEATURES
# Compute on raw combined data BEFORE imputation to capture true
# synthetic grid density.
# -----------------------------------------------------------------
print("\n[FEAT] Computing value frequency count features (synthetic grid density)...")
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
        print(f"   {col}_freq: {len(freq_maps[col])} unique values, "
              f"max_freq={max(freq_maps[col].values())}, "
              f"mean_freq={sum(freq_maps[col].values())/len(freq_maps[col]):.1f}")

# -----------------------------------------------------------------
# 1.6. SEMI-SUPERVISED PSEUDO-LABELING
# Upgraded to RAW submission_v703.csv
# -----------------------------------------------------------------
print(f"\n[SEMI-SUPERVISED] Pseudo-Labeling configuration: USE_PSEUDO={USE_PSEUDO}")
train_df['is_pseudo'] = 0
test_df['is_pseudo'] = 0

if USE_PSEUDO:
    pseudo_path = "submission_v703.csv"
    if not os.path.exists(pseudo_path):
        pseudo_path = "submissions/submission_v703.csv"
    if not os.path.exists(pseudo_path):
        pseudo_path = "submission_v30.csv"  # fallback
    if not os.path.exists(pseudo_path):
        pseudo_path = "submissions/submission_v30.csv"

    if os.path.exists(pseudo_path):
        sub_blend = pd.read_csv(pseudo_path)
        test_pseudo = test_df.merge(sub_blend, on="record_id", how="left")
        pseudo_rows = test_pseudo.copy()
        pseudo_rows['is_pseudo'] = 1
        print(f"   Added {len(pseudo_rows)} soft pseudo-labeled rows from test set: {pseudo_path}")
        train_df = pd.concat([train_df, pseudo_rows], ignore_index=True)
    else:
        print("   [WARNING] submission pseudo-labels not found. Skipping pseudo-labeling.")

# -----------------------------------------------------------------
# 2. ROBUST CUSTOMIZED IMPUTATION FRAMEWORK
# -----------------------------------------------------------------
print("\n[IMPUTE] Starting customized imputation framework...")
combined = pd.concat([
    train_df.drop(columns=['flood_risk_score'], errors='ignore'), test_df
], ignore_index=True)

# 2.1. Coordinate Geospatial Hot-Deck lookup
coords_lookup = combined.groupby(['place_name', 'district'])[['latitude', 'longitude']].median().to_dict('index')
for df in [train_df, test_df]:
    mask = df['latitude'].isnull() & df['place_name'].notnull() & df['district'].notnull()
    for idx in df[mask].index:
        key = (df.loc[idx, 'place_name'], df.loc[idx, 'district'])
        if key in coords_lookup and not np.isnan(coords_lookup[key]['latitude']):
            df.loc[idx, 'latitude'] = coords_lookup[key]['latitude']
            df.loc[idx, 'longitude'] = coords_lookup[key]['longitude']

# Fallback coordinates to district median
for col in ['latitude', 'longitude']:
    district_median = combined.groupby('district')[col].median().to_dict()
    global_median = train_df[col].median()
    for df in [train_df, test_df]:
        df[col] = df[col].fillna(df['district'].map(district_median))
        df[col] = df[col].fillna(global_median)

# 2.2. KNN Coordinate Imputation for topographic & distance columns
# Used for: elevation_m, distance_to_river_m, nearest_hospital_km, nearest_evac_km
knn_cols = ['elevation_m', 'distance_to_river_m', 'nearest_hospital_km', 'nearest_evac_km']
print("   Imputing geospatial & distance columns via KNN coordinate regression...")
for col in knn_cols:
    donor_pool = combined.dropna(subset=['latitude', 'longitude', col])
    knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
    knn.fit(donor_pool[['latitude', 'longitude']], donor_pool[col])
    for df in [train_df, test_df]:
        mm = df[col].isnull()
        if mm.any():
            df.loc[mm, col] = knn.predict(df.loc[mm, ['latitude', 'longitude']])
        district_median = combined.groupby('district')[col].median().to_dict()
        df[col] = df[col].fillna(df['district'].map(district_median))
        df[col] = df[col].fillna(train_df[col].median())

# 2.3. Soil and Drainage Attributes (drainage_index)
print("   Imputing drainage index via Soil Type + Landcover medians...")
drainage_lookup = combined.groupby(['soil_type', 'landcover'])['drainage_index'].median().to_dict()
soil_drainage_lookup = combined.groupby('soil_type')['drainage_index'].median().to_dict()
district_drainage_lookup = combined.groupby('district')['drainage_index'].median().to_dict()
global_drainage_median = train_df['drainage_index'].median()

for df in [train_df, test_df]:
    mask = df['drainage_index'].isnull()
    for idx in df[mask].index:
        soil = df.loc[idx, 'soil_type']
        lc = df.loc[idx, 'landcover']
        dist = df.loc[idx, 'district']
        
        val = np.nan
        if pd.notnull(soil) and pd.notnull(lc) and (soil, lc) in drainage_lookup:
            val = drainage_lookup[(soil, lc)]
        if np.isnan(val) and pd.notnull(soil) and soil in soil_drainage_lookup:
            val = soil_drainage_lookup[soil]
        if np.isnan(val) and pd.notnull(dist) and dist in district_drainage_lookup:
            val = district_drainage_lookup[dist]
        if np.isnan(val):
            val = global_drainage_median
        
        df.loc[idx, 'drainage_index'] = val

# 2.4. Environmental Indices (ndvi, ndwi)
print("   Imputing NDVI/NDWI via District + Landcover medians...")
for col in ['ndvi', 'ndwi']:
    dist_lc_lookup = combined.groupby(['district', 'landcover'])[col].median().to_dict()
    lc_lookup = combined.groupby('landcover')[col].median().to_dict()
    global_median = train_df[col].median()
    
    for df in [train_df, test_df]:
        mask = df[col].isnull()
        for idx in df[mask].index:
            dist = df.loc[idx, 'district']
            lc = df.loc[idx, 'landcover']
            
            val = np.nan
            if pd.notnull(dist) and pd.notnull(lc) and (dist, lc) in dist_lc_lookup:
                val = dist_lc_lookup[(dist, lc)]
            if np.isnan(val) and pd.notnull(lc) and lc in lc_lookup:
                val = lc_lookup[lc]
            if np.isnan(val):
                val = global_median
            
            df.loc[idx, col] = val

# 2.5. Human/Infrastructure/Development Features
print("   Imputing human development features via District + Urban/Rural medians...")
dev_cols = ['population_density_per_km2', 'built_up_percent', 'infrastructure_score']
for col in dev_cols:
    dist_ur_lookup = combined.groupby(['district', 'urban_rural'])[col].median().to_dict()
    ur_lookup = combined.groupby('urban_rural')[col].median().to_dict()
    dist_lookup = combined.groupby('district')[col].median().to_dict()
    global_median = train_df[col].median()
    
    for df in [train_df, test_df]:
        mask = df[col].isnull()
        for idx in df[mask].index:
            dist = df.loc[idx, 'district']
            ur = df.loc[idx, 'urban_rural']
            
            val = np.nan
            if pd.notnull(dist) and pd.notnull(ur) and (dist, ur) in dist_ur_lookup:
                val = dist_ur_lookup[(dist, ur)]
            if np.isnan(val) and pd.notnull(ur) and ur in ur_lookup:
                val = ur_lookup[ur]
            if np.isnan(val) and pd.notnull(dist) and dist in dist_lookup:
                val = dist_lookup[dist]
            if np.isnan(val):
                val = global_median
            
            df.loc[idx, col] = val

# 2.6. Categorical Imputation via District Mode
print("   Imputing categorical features via District Mode...")
cat_impute_cols = [
    "district", "landcover", "soil_type", "water_supply", 
    "electricity", "road_quality", "urban_rural", "water_presence_flag"
]
for col in cat_impute_cols:
    dist_modes = combined.groupby('district')[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
    global_mode = combined[col].mode().iloc[0]
    for df in [train_df, test_df]:
        if col == "district":
            df[col] = df[col].fillna(global_mode)
        else:
            df[col] = df[col].fillna(df['district'].map(dist_modes))
            df[col] = df[col].fillna(global_mode)

print("   Customized imputation completed successfully.")

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
    df['monsoon_impact'] = df['rainfall_7d_mm'] * df['is_yala'] * (df['zone_code'] == 1).astype(int) + \
                           df['rainfall_7d_mm'] * df['is_maha'] * (df['zone_code'] == 2).astype(int)
    df['urban_runoff_potential'] = df['rainfall_7d_mm'] * df['built_up_percent'] * (1.0 / (df['drainage_index'] + 1e-5))
    df['fluvial_risk_score_feat'] = df['rainfall_7d_mm'] * (1.0 / (df['distance_to_river_m'] + 1.0))
    df['soil_infiltration'] = df['soil_type'].astype(str).map(soil_infilt_map).fillna(0.4)
    df['soil_saturation_limit'] = df['rainfall_7d_mm'] / (df['soil_infiltration'] + 0.1)
    df['pseudo_twi'] = np.log1p((df['distance_to_river_m'] + 1.0) / (df['elevation_m'].clip(lower=0.0) + 1.0))
    df['flatness_index'] = df['district'].astype(str).map(district_elev_std).fillna(df['elevation_m'].std())
    df['in_cyclone_path'] = df['district'].astype(str).map(lambda x: 1 if x in cyclone_districts else 0)
    
    # [CHANGE] Removed extreme_weather_index dependent features (cyclone_vulnerability and flood_pressure)
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
STD_ENC_COLS = [
    "district", "downstream_sig"
]

# [CHANGE v1] extreme_weather_index dropped (AV Pass #1, importance=541)
# [CHANGE v2] 8 additional features dropped (AV Pass #2, AUC=0.65277)
IGNORE_COLS = DROP_COLS + [
    TARGET, "flood_occurrence_yes", "downstream_sig", "downstream_quad_sig", "infra_deficit_sig",
    # --- Adversarial Drop v1 ---
    "extreme_weather_index",
    # --- Adversarial Drop v2: Opaque synthetic indices ---
    "seasonal_index", "terrain_roughness_index", "socioeconomic_status_index",
    # --- Adversarial Drop v2: Engineered derivatives of shifted raw features ---
    "drainage_deficit", "soil_drainage_saturation", "rain_spike_ratio",
    "infra_resilience", "urban_runoff_potential",
]
SPATIAL_HELPERS = ["lat_bin", "lon_bin", "grid_id", "grid_id_100", "grid_id_050", "grid_id_025", "grid_id_012"]
BASE_FEATURES = [c for c in train_df.columns if c not in IGNORE_COLS and c not in SPATIAL_HELPERS]

freq_feature_names = [f'{col}_freq' for col in FREQ_COLS if f'{col}_freq' in train_df.columns]
print(f"\n[FREQ] Frequency features in BASE_FEATURES: {[f for f in freq_feature_names if f in BASE_FEATURES]}")

downstream_cols = [
    "confirmed_severe_risk", "no_flood_confirmed", "inundation_per_capita", "downstream_risk_count",
    "downstream_sig", "downstream_quad_sig", "confirmed_risk", "inundation_ratio", "flood_occurrence_yes",
    "inundation_flood_interaction", "inundation_density_risk"
]
conflict_key_cols = [c for c in BASE_FEATURES if c not in downstream_cols]

print("\n[PREP] Casting dtypes...")
cat_dtype_map = {}
for col in BASE_FEATURES:
    if col in CAT_FEATURES:
        if col == "reason_not_good_to_live":
            # Retain 'missing' category for reason_not_good_to_live
            train_df[col] = train_df[col].fillna("missing").astype(str)
            test_df[col]  = test_df[col].fillna("missing").astype(str)
        
        all_vals = sorted(set(train_df[col].unique()) | set(test_df[col].unique()))
        cdt = pd.CategoricalDtype(categories=all_vals, ordered=False)
        train_df[col] = train_df[col].astype(cdt)
        test_df[col]  = test_df[col].astype(cdt)
        cat_dtype_map[col] = cdt
    elif train_df[col].dtype in ["int64", "float64", "int32", "float32"]:
        # Imputed previously, just fallback safeguard
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col]  = test_df[col].fillna(median_val)

cat_feature_names = [c for c in CAT_FEATURES if c in BASE_FEATURES]
print(f"   Base features: {len(BASE_FEATURES)}")

def to_cat_fmt(df):
    df = df.copy()
    for col in cat_feature_names:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

# -----------------------------------------------------------------
# 5. MODEL DEFINITIONS — CAT-RMSE replaced with CAT-HUBER
# CAT-RMSE received 0.0000 stacker weight across all 5 seeds in vmav2.
# CAT-HUBER (delta=0.05) bridges MAE and RMSE: behaves like MSE for
# errors < 0.05 and like MAE for larger errors, adding a distinct
# gradient profile the stacker can actually use.
# -----------------------------------------------------------------
MODEL_NAMES = [
    "XGB-MAE-1 (d7)",
    "CAT-MAE-1 (d5)",
    "CAT-MAE-2 (d5)",
    "CAT-HUB (d6)",
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
print(f"  5-FOLD SPATIAL GROUP CV - SINGLE-SEED vmav2h_s1 PIPELINE")
print("=" * 75)

t_start_global = time.time()

# [CHANGE] Stable stacker weights optimization coefficients to prevent overfitting
c_mae_opt, c_rmse_opt, c_ev_opt = 0.539328, 1.152263, 0.048467

# [CHANGE] True refitted competition metric coefficients (fitted post-refit on 28 LB points)
c_mae_eval, c_rmse_eval, c_ev_eval, c_int_eval = 0.544177, 1.148953, 0.049625, -0.000580

# -----------------------------------------------------------------
# 6. L2 REGULARIZED CUSTOM METRIC-DRIVEN LEVEL-2 STACKER
# -----------------------------------------------------------------
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
        # Use stable coefficients to prevent EV calibration noise from degrading stacker weights
        score = (c_mae_opt * mae + c_rmse_opt * rmse) * (1.0 + c_ev_opt * (1.0 - ev))
        reg = alpha * np.sum(w**2)
        return score + reg

    init_guess = np.ones(n_models) / n_models
    init_guess = np.append(init_guess, 0.0)
    bounds = [(0.0, None) for _ in range(n_models)] + [(None, None)]
    res = minimize(loss_fn, init_guess, bounds=bounds, method='L-BFGS-B')
    return res.x[:-1], res.x[-1]

# -----------------------------------------------------------------
# MULTI-SEED OUTER LOOP
# -----------------------------------------------------------------
for seed in SEEDS:
    print(f"\n==================== RUNNING SEED {seed} ====================")
    oof_preds = {m: np.zeros(len(train_df)) for m in MODEL_NAMES}
    tst_preds = {m: np.zeros(len(test_df))  for m in MODEL_NAMES}

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, groups)):
        t0 = time.time()

        va_is_pseudo = train_df.iloc[va_idx]['is_pseudo'] == 1
        va_idx_clean = va_idx[~va_is_pseudo] if va_is_pseudo.any() else va_idx

        tr_rows = train_df.iloc[tr_idx].copy()
        va_rows = train_df.iloc[va_idx_clean].copy()

        # Conflict resolution (same as v70 — all tr_rows)
        temp_tr = tr_rows[conflict_key_cols].copy()
        for col in temp_tr.columns:
            if temp_tr[col].dtype in ['object', 'category']:
                temp_tr[col] = temp_tr[col].astype(str).fillna('missing')
            else:
                temp_tr[col] = temp_tr[col].fillna(-999)
        group_medians_tr = tr_rows.groupby([temp_tr[c] for c in temp_tr.columns])[TARGET].transform('median')
        tr_rows[TARGET] = group_medians_tr

        # Target encodings (from real training rows only)
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
            smoothed_mean   = (group_stats['count'] * group_stats['mean']   + s * GLOBAL_MEAN)   / (group_stats['count'] + s)
            smoothed_std    = (group_stats['count'] * group_stats['std']    + s * GLOBAL_STD)    / (group_stats['count'] + s)
            smoothed_q25    = (group_stats['count'] * group_stats['q25']    + s * GLOBAL_Q25)    / (group_stats['count'] + s)
            smoothed_q75    = (group_stats['count'] * group_stats['q75']    + s * GLOBAL_Q75)    / (group_stats['count'] + s)
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

        cat_cols = [c for c in CAT_FEATURES if c in FEATURES]

        def to_xgb_fmt(df):
            df = df.copy()
            for c in df.columns:
                if hasattr(df[c], "cat"):
                    df[c] = df[c].cat.codes.astype("int32")
            return df

        def to_cat_fmt_local(df):
            df = df.copy()
            for c in cat_cols:
                if c in df.columns:
                    df[c] = df[c].astype(str)
            return df

        X_tr_xgb = to_xgb_fmt(X_tr); X_va_xgb = to_xgb_fmt(X_va); X_te_xgb = to_xgb_fmt(X_te)
        X_tr_cat = to_cat_fmt_local(X_tr); X_va_cat = to_cat_fmt_local(X_va); X_te_cat = to_cat_fmt_local(X_te)

        for col in cat_cols:
            cdt = cat_dtype_map[col]
            X_tr[col] = X_tr[col].astype(str).astype(cdt)
            X_va[col] = X_va[col].astype(str).astype(cdt)
            X_te[col] = X_te[col].astype(str).astype(cdt)

        # 1. XGB-MAE-1 (d7)
        xgb_m1 = xgb.XGBRegressor(
            n_estimators=4000, learning_rate=0.03, max_depth=7, min_child_weight=4,
            subsample=0.85, colsample_bytree=0.6, colsample_bylevel=0.6,
            reg_alpha=2.0, reg_lambda=4.0, gamma=0.1, max_delta_step=1,
            objective="reg:absoluteerror", eval_metric="mae", tree_method="hist",
            enable_categorical=False, early_stopping_rounds=100, random_state=seed, n_jobs=-1
        )
        xgb_m1.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)

        # 2. CAT-MAE-1 (d5)
        cat_m1 = cb.CatBoostRegressor(
            iterations=5000, learning_rate=0.03, depth=5, l2_leaf_reg=5.0,
            bagging_temperature=0.7, random_strength=2.0, border_count=254,
            loss_function="MAE", eval_metric="MAE", task_type="CPU",
            random_seed=seed, verbose=False
        )
        cat_m1.fit(X_tr_cat, y_tr, cat_features=cat_cols, eval_set=(X_va_cat, y_va), early_stopping_rounds=150, verbose=False)

        # 3. CAT-MAE-2 (d5)
        cat_m2 = cb.CatBoostRegressor(
            iterations=5000, learning_rate=0.03, depth=5, l2_leaf_reg=12.0,
            bagging_temperature=0.4, random_strength=5.0, border_count=254,
            loss_function="MAE", eval_metric="MAE", task_type="CPU",
            random_seed=seed + 1, verbose=False
        )
        cat_m2.fit(X_tr_cat, y_tr, cat_features=cat_cols, eval_set=(X_va_cat, y_va), early_stopping_rounds=150, verbose=False)

        # 4. CAT-HUB (d6) — Huber loss replaces the zeroed-out CAT-RMSE
        # delta=0.05: quadratic below 5% error, linear above — diverse gradient vs MAE models
        # depth=6: slightly more capacity than the MAE CatBoost pair (d5)
        cat_hub = cb.CatBoostRegressor(
            iterations=4000, learning_rate=0.03, depth=6, l2_leaf_reg=6.0,
            bagging_temperature=0.5, random_strength=2.5, border_count=254,
            loss_function="Huber:delta=0.05", eval_metric="MAE", task_type="CPU",
            random_seed=seed + 2, verbose=False
        )
        cat_hub.fit(X_tr_cat, y_tr, cat_features=cat_cols, eval_set=(X_va_cat, y_va), early_stopping_rounds=150, verbose=False)

        # 5. LGB-MAE (d5)
        lgb_m1 = lgb.LGBMRegressor(
            n_estimators=4000, learning_rate=0.03, num_leaves=31, max_depth=5,
            min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
            reg_alpha=2.0, reg_lambda=5.0, objective="regression_l1",
            random_state=seed, n_jobs=-1, verbosity=-1
        )
        lgb_m1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(150, verbose=False)])

        # 6. XGB-MAE-2 (d5)
        xgb_m2 = xgb.XGBRegressor(
            n_estimators=4000, learning_rate=0.03, max_depth=5, min_child_weight=6,
            subsample=0.75, colsample_bytree=0.5, colsample_bylevel=0.8,
            reg_alpha=5.0, reg_lambda=10.0, gamma=0.2, max_delta_step=1,
            objective="reg:absoluteerror", eval_metric="mae", tree_method="hist",
            enable_categorical=False, early_stopping_rounds=100, random_state=seed + 3, n_jobs=-1
        )
        xgb_m2.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)

        # Predictions
        oof_preds["XGB-MAE-1 (d7)"][va_idx_clean] = xgb_m1.predict(X_va_xgb)
        oof_preds["CAT-MAE-1 (d5)"][va_idx_clean] = cat_m1.predict(X_va_cat)
        oof_preds["CAT-MAE-2 (d5)"][va_idx_clean] = cat_m2.predict(X_va_cat)
        oof_preds["CAT-HUB (d6)"][va_idx_clean]   = cat_hub.predict(X_va_cat)
        oof_preds["LGB-MAE (d5)"][va_idx_clean]   = lgb_m1.predict(X_va)
        oof_preds["XGB-MAE-2 (d5)"][va_idx_clean] = xgb_m2.predict(X_va_xgb)

        tst_preds["XGB-MAE-1 (d7)"] += xgb_m1.predict(X_te_xgb)  / N_FOLDS
        tst_preds["CAT-MAE-1 (d5)"] += cat_m1.predict(X_te_cat)   / N_FOLDS
        tst_preds["CAT-MAE-2 (d5)"] += cat_m2.predict(X_te_cat)   / N_FOLDS
        tst_preds["CAT-HUB (d6)"]   += cat_hub.predict(X_te_cat)  / N_FOLDS
        tst_preds["LGB-MAE (d5)"]   += lgb_m1.predict(X_te)       / N_FOLDS
        tst_preds["XGB-MAE-2 (d5)"] += xgb_m2.predict(X_te_xgb)  / N_FOLDS

        oof_avg_fold = np.mean([oof_preds[m][va_idx_clean] for m in MODEL_NAMES], axis=0)
        y_va_arr = y_va.values
        f_mae  = mean_absolute_error(y_va_arr, oof_avg_fold)
        f_rmse = root_mean_squared_error(y_va_arr, oof_avg_fold)
        f_ev   = explained_variance_score(y_va_arr, oof_avg_fold)
        print(f"      Fold {fold+1}/5 | XGB1={xgb_m1.best_iteration:<4} CAT1={cat_m1.best_iteration_:<4} CAT2={cat_m2.best_iteration_:<4} HUB={cat_hub.best_iteration_:<4} LGB={lgb_m1.best_iteration_:<4} XGB2={xgb_m2.best_iteration:<4} | [ENS MAE={f_mae:.4f}] [{time.time() - t0:.0f}s]")

    oof_meta_seed = np.column_stack([oof_preds[m][real_mask] for m in MODEL_NAMES])
    tst_meta_seed = np.column_stack([tst_preds[m] for m in MODEL_NAMES])

    print("   [STACK] Running nested CV grid search for L2 alpha...")
    best_alpha, best_score = 0.1, np.inf
    alphas_to_test = [0.001, 0.01, 0.1, 1.0, 10.0]

    for alpha in alphas_to_test:
        oof_cv = np.zeros(len(original_y))
        for fold_l2, (tr_idx_l2, va_idx_l2) in enumerate(gkf_l2.split(original_df, original_y, original_groups)):
            w_cv, b_cv = fit_metric_stacker(oof_meta_seed[tr_idx_l2], original_y[tr_idx_l2], alpha=alpha)
            oof_cv[va_idx_l2] = np.clip(np.dot(oof_meta_seed[va_idx_l2], w_cv) + b_cv, 0.0, 1.0)
        cv_mae  = mean_absolute_error(original_y, oof_cv)
        cv_rmse = root_mean_squared_error(original_y, oof_cv)
        cv_ev   = explained_variance_score(original_y, oof_cv)
        # CV alpha search uses stable coefficients
        cv_score = (c_mae_opt * cv_mae + c_rmse_opt * cv_rmse) * (1.0 + c_ev_opt * (1.0 - cv_ev))
        if cv_score < best_score:
            best_score = cv_score
            best_alpha = alpha

    print(f"   [L2 SEARCH SEED {seed}] Best alpha: {best_alpha} (Nested CV Metric Score: {best_score:.5f})")

    oof_stacked_seed = np.zeros(len(original_y))
    for fold, (tr_idx, va_idx) in enumerate(gkf_l2.split(original_df, original_y, original_groups)):
        w_fold, b_fold = fit_metric_stacker(oof_meta_seed[tr_idx], original_y[tr_idx], alpha=best_alpha)
        oof_stacked_seed[va_idx] = np.clip(np.dot(oof_meta_seed[va_idx], w_fold) + b_fold, 0.0, 1.0)

    all_oof_stacked += oof_stacked_seed / len(SEEDS)

    w_final_seed, b_final_seed = fit_metric_stacker(oof_meta_seed, original_y, alpha=best_alpha)
    tst_stacked_seed = np.clip(np.dot(tst_meta_seed, w_final_seed) + b_final_seed, 0.0, 1.0)
    all_tst_stacked.append(tst_stacked_seed)

    print(f"   [FINAL SEED {seed} COEFFICIENTS]")
    for i, name in enumerate(MODEL_NAMES):
        print(f"      {name:<18}: {w_final_seed[i]:.4f}")
    print(f"      {'Intercept':<18}: {b_final_seed:.4f}")

# -----------------------------------------------------------------
# 7. GLOBAL ENSEMBLE RESULTS & METRICS
# -----------------------------------------------------------------
g_mae  = mean_absolute_error(original_y, all_oof_stacked)
g_rmse = root_mean_squared_error(original_y, all_oof_stacked)
g_ev   = explained_variance_score(original_y, all_oof_stacked)
# True simulator score reporting using true fitted coefficients + intercept offset
g_lb   = (c_mae_eval * g_mae + c_rmse_eval * g_rmse) * (1.0 + c_ev_eval * (1.0 - g_ev)) + c_int_eval

print("\n" + "=" * 75)
print("  GLOBAL OOF RESULTS (vmav2h_s1 - Adversarial Drop v2 + CAT-HUBER)")
print("=" * 75)
print(f"    [ALL ROWS]")
print(f"      MAE            : {g_mae:.5f}")
print(f"      RMSE           : {g_rmse:.5f}")
print(f"      Explained Var. : {g_ev:.5f}")
print(f"      Est. LB Score  : {g_lb:.5f}")
print("=" * 75)

fold_results_all = []
for fold, (tr_idx, va_idx) in enumerate(gkf_l2.split(original_df, original_y, original_groups)):
    f_mae  = mean_absolute_error(original_y[va_idx], all_oof_stacked[va_idx])
    f_rmse = root_mean_squared_error(original_y[va_idx], all_oof_stacked[va_idx])
    f_ev   = explained_variance_score(original_y[va_idx], all_oof_stacked[va_idx])
    fold_results_all.append({"fold": fold+1, "MAE": f_mae, "RMSE": f_rmse, "EV": f_ev})

fold_report = pd.DataFrame(fold_results_all)
fold_report.to_csv("fold_report_vmav2h_s1.csv", index=False)
fold_report.to_csv("submissions/fold_report_vmav2h_s1.csv", index=False)
print(f"\n[DONE] Saved fold reports.")

tst_stacked_avg = np.clip(np.mean(all_tst_stacked, axis=0), 0.0, 1.0)
submission = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": tst_stacked_avg
})
submission.to_csv("submission_vmav2h_s1.csv", index=False)
submission.to_csv("submissions/submission_vmav2h_s1.csv", index=False)
print(f"[DONE] Saved submission_vmav2h_s1.csv ({len(submission)} rows)")

np.save("oof_vmav2h_s1.npy", all_oof_stacked)
np.save("submissions/oof_vmav2h_s1.npy", all_oof_stacked)
print(f"[DONE] Saved oof_vmav2h_s1.npy")

# -----------------------------------------------------------------
# 8. POST-HOC POWER TRANSFORMATION OPTIMIZATION
# -----------------------------------------------------------------
print("\n" + "=" * 75)
print("  POST-HOC POWER TRANSFORMATION OPTIMIZATION (vmav2h_s1)")
print("=" * 75)

def transform_loss(params):
    a, b, c = params
    pred = a * np.power(np.clip(all_oof_stacked, 1e-6, None), b) + c
    pred = np.clip(pred, 0.0, 1.0)
    mae  = mean_absolute_error(original_y, pred)
    rmse = root_mean_squared_error(original_y, pred)
    ev   = explained_variance_score(original_y, pred)
    # Post-hoc uses stable optimization coefficients to avoid EV calibration noise
    return (c_mae_opt * mae + c_rmse_opt * rmse) * (1.0 + c_ev_opt * (1.0 - ev))

initial_guess = [1.0, 1.0, 0.0]
bounds = [(0.5, 1.5), (0.5, 2.0), (-0.15, 0.15)]

res_opt = minimize(transform_loss, initial_guess, bounds=bounds, method='L-BFGS-B')
a_opt, b_opt, c_opt = res_opt.x
print(f"Optimal parameters: a={a_opt:.5f}, b={b_opt:.5f}, c={c_opt:.5f}")

opt_oof = a_opt * np.power(np.clip(all_oof_stacked, 1e-6, None), b_opt) + c_opt
opt_oof = np.clip(opt_oof, 0.0, 1.0)

opt_mae  = mean_absolute_error(original_y, opt_oof)
opt_rmse = root_mean_squared_error(original_y, opt_oof)
opt_ev   = explained_variance_score(original_y, opt_oof)
opt_lb   = (c_mae_eval * opt_mae + c_rmse_eval * opt_rmse) * (1.0 + c_ev_eval * (1.0 - opt_ev)) + c_int_eval

print(f"\nOptimized OOF LB Score: {opt_lb:.5f}")
print(f"  MAE: {opt_mae:.5f}, RMSE: {opt_rmse:.5f}, EV: {opt_ev:.5f}")

np.save("oof_vmav2h_s1_optimized.npy", opt_oof)
np.save("submissions/oof_vmav2h_s1_optimized.npy", opt_oof)
print(f"[DONE] Saved oof_vmav2h_s1_optimized.npy")

opt_test_preds = a_opt * np.power(np.clip(tst_stacked_avg, 1e-6, None), b_opt) + c_opt
opt_test_preds = np.clip(opt_test_preds, 0.0, 1.0)

submission_opt = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": opt_test_preds
})
submission_opt.to_csv("submission_vmav2h_s1_optimized.csv", index=False)
submission_opt.to_csv("submissions/submission_vmav2h_s1_optimized.csv", index=False)
print(f"[DONE] Saved submission_vmav2h_s1_optimized.csv ({len(submission_opt)} rows)")
print(f"  Optimized range  : [{opt_test_preds.min():.4f}, {opt_test_preds.max():.4f}]")
print(f"  Total Time       : {time.time() - t_start_global:.1f}s")
print("=" * 75)
