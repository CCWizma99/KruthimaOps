"""
ML Opsidian: Genesis - Calibrated Feature Engineering Pipeline v12
===================================================================
Implements ALL remaining features from data_discussion_notes (Sections 5-9):

  NEW FEATURES (on top of v11):
    - Water Pooling Accumulator    : ndwi * extreme_weather_index
    - Socio-Economic Deficit       : (1/socioeconomic_status_index) * extreme_weather_index
    - Seasonal Rain Anomaly        : rainfall_7d_mm * seasonal_index (Section 6)
    - Inundation Deviational Ratio : inundation_area / landcover_mean_inundation (Section 5)
    - Runoff Engine (simple)       : built_up_percent * rainfall_7d_mm (Section 7)
    - Soil Saturation (Section 7)  : rainfall_7d_mm * (1.0 - drainage_index) [already in v11]

  HYPERPARAMETER CALIBRATION (Section 9):
    - Ground-Truth Sample Weighting: Real rows -> 25.0, Synthetic -> 1.0
    - Shallower Trees              : max_depth 4-5 (down from 7)
    - Micro Learning Rates         : 0.01 (down from 0.05)
    - High Patience                : early_stopping_rounds=150 (up from 100)
    - Enhanced L2 Regularization   : reg_lambda increased
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
print("=" * 70)
print("  ML OPSIDIAN v12 - CALIBRATED FEATURE ENGINEERING PIPELINE")
print("=" * 70)
print("\n[LOAD] Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")
print(f"   Train shape      : {train_df.shape}")
print(f"   Test shape       : {test_df.shape}")
train_df = train_df.drop_duplicates()
print(f"   Train after dedup: {train_df.shape}")

# -----------------------------------------------------------------
# 2. EXTRACT SAMPLE WEIGHTS (Before dropping is_synthetic)
# -----------------------------------------------------------------
print("\n[WGHT] Extracting sample weights (Section 9 - Ground Truth Weighting)...")
# Real rows (is_synthetic is NaN) get weight 25.0, Synthetic get 1.0
sample_weights = train_df['is_synthetic'].apply(
    lambda x: 1.0 if x == True else 25.0
).values
real_count = (sample_weights == 25.0).sum()
synth_count = (sample_weights == 1.0).sum()
print(f"   Real rows  (weight=25.0) : {real_count}")
print(f"   Synth rows (weight= 1.0) : {synth_count}")
print(f"   Effective real fraction   : {real_count * 25.0 / (real_count * 25.0 + synth_count * 1.0):.2%}")

# -----------------------------------------------------------------
# 3. GEOSPATIAL HOT-DECK IMPUTATION
# -----------------------------------------------------------------
print("\n[IMPUTE] Starting Geospatial Hot-Deck Imputation...")

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
# 4. FEATURE ENGINEERING ENGINE
# -----------------------------------------------------------------
print("\n[FEAT] Engineering features...")

# Pre-calculate district-level elevation standard deviation as flatness indicator
combined_imputed = pd.concat([
    train_df.drop(columns=['flood_risk_score'], errors='ignore'),
    test_df
], ignore_index=True)
district_elev_std = combined_imputed.groupby('district')['elevation_m'].std().to_dict()

# Pre-calculate landcover mean inundation for normalization (Section 5 - Inundation Ratio)
landcover_mean_inundation = combined_imputed.groupby('landcover')['inundation_area_sqm'].mean().to_dict()
print(f"   -> Landcover mean inundation map: {len(landcover_mean_inundation)} classes")
for k, v in sorted(landcover_mean_inundation.items(), key=lambda x: -x[1]):
    print(f"      {k:>15}: {v:,.1f} sqm")

# Soil infiltration rates mapping (Section 8)
soil_infilt_map = {
    'Sandy': 0.8,
    'Loamy': 0.6,
    'Silty': 0.4,
    'Clay': 0.2,
    'Peaty': 0.1
}

# Cyclone corridor districts (Section 8)
cyclone_districts = {'Batticaloa', 'Trincomalee', 'Ampara', 'Mullaitivu', 'Jaffna'}

# Wet Zone districts (zone_code = 1) (Section 8)
wet_zone_districts = {'Colombo', 'Gampaha', 'Kalutara', 'Galle', 'Matara', 'Ratnapura', 'Kegalle'}

def engineer_features(df):
    df = df.copy()
    
    # ================ SECTION 5: Proposed Feature Transformations ================
    
    # 1. Downstream signature key
    print("      [S5] Downstream signature key...")
    df['downstream_sig'] = (
        df['flood_occurrence_current_event'].astype(str).str.strip() + "_" +
        df['is_good_to_live'].astype(str).str.strip() + "_" +
        df['reason_not_good_to_live'].astype(str).str.strip()
    )
    
    # 2. Inundation Deviational Ratio (NEW in v12)
    print("      [S5] Inundation deviational ratio (normalize by landcover class)...")
    df['landcover_mean_inundation'] = df['landcover'].astype(str).map(landcover_mean_inundation).fillna(
        combined_imputed['inundation_area_sqm'].mean()
    )
    df['inundation_ratio'] = df['inundation_area_sqm'] / (df['landcover_mean_inundation'] + 1.0)
    
    # 3. Confirmed Risk Indicator
    df['confirmed_risk'] = (
        (df['flood_occurrence_current_event'].astype(str).str.strip().str.lower() == 'yes') &
        (df['is_good_to_live'].astype(str).str.strip().str.lower() == 'no')
    ).astype(int)
    
    # ================ SECTION 6: Seasonal Index Decoupling ================
    
    # 4. Extract calendar month
    print("      [S6] Seasonal rain anomaly coupling...")
    date_series = pd.to_datetime(df['generation_date'])
    df['month'] = date_series.dt.month
    
    # 5. Seasonal Rain Anomaly (NEW in v12)
    df['seasonal_rain_anomaly'] = df['rainfall_7d_mm'] * df['seasonal_index']
    
    # ================ SECTION 7: Advanced Feature Ideas ================
    
    # 6. Slope Proxy
    print("      [S7] Slope proxy, coordinate multipliers, runoff engine...")
    df['slope_proxy'] = df['elevation_m'] / (df['distance_to_river_m'] + 1.0)
    
    # 7. Coordinate Multipliers
    df['lat_lon_multiply'] = df['latitude'] * df['longitude']
    df['lat_lon_divide'] = df['latitude'] / (df['longitude'] + 1e-5)
    
    # 8. Soil Saturation Proxy
    df['soil_saturation'] = df['rainfall_7d_mm'] * (1.0 - df['drainage_index'])
    
    # 9. Runoff Engine (NEW simplified version in v12)
    df['runoff_engine'] = df['built_up_percent'] * df['rainfall_7d_mm']
    
    # 10. Water Pooling Accumulator (NEW in v12)
    print("      [S7] Water pooling accumulator...")
    df['pooling_accumulator'] = df['ndwi'] * df['extreme_weather_index']
    
    # 11. Socio-Economic Deficit (NEW in v12)
    print("      [S7] Socio-economic deficit...")
    df['socioeconomic_deficit'] = (1.0 / (df['socioeconomic_status_index'] + 1e-5)) * df['extreme_weather_index']
    
    # 12. Isolation Multiplier
    df['isolation_multiplier'] = df['nearest_hospital_km'] * df['nearest_evac_km']
    
    # 13. Transformation Divergence
    df['elevation_divergence'] = df['elevation_m'] - df['elevation_m_yeojohnson']
    
    # 14. Infrastructure Deficit String
    df['infra_deficit_sig'] = (
        df['water_supply'].astype(str).str.strip() + "_" +
        df['electricity'].astype(str).str.strip() + "_" +
        df['road_quality'].astype(str).str.strip()
    )
    
    # ================ SECTION 8: Geographical & Climatological ================
    
    # 15. Monsoon Switch Flag & Score
    print("      [S8] Monsoon switch, urban/rural runoff, soil sponge, TWI...")
    df['is_yala'] = df['month'].isin([5, 6, 7, 8, 9]).astype(int)
    df['is_maha'] = df['month'].isin([11, 12, 1]).astype(int)
    df['zone_code'] = df['district'].astype(str).map(lambda x: 1 if x in wet_zone_districts else 2)
    df['monsoon_impact'] = df['rainfall_7d_mm'] * df['is_yala'] * (df['zone_code'] == 1).astype(int) + \
                           df['rainfall_7d_mm'] * df['is_maha'] * (df['zone_code'] == 2).astype(int)
                           
    # 16. Urban Pluvial vs. Rural Fluvial
    df['urban_runoff_potential'] = df['rainfall_7d_mm'] * df['built_up_percent'] * (1.0 / (df['drainage_index'] + 1e-5))
    df['fluvial_risk_score_feat'] = df['rainfall_7d_mm'] * (1.0 / (df['distance_to_river_m'] + 1.0))
    
    # 17. Soil saturation limit (infiltration physics)
    df['soil_infiltration'] = df['soil_type'].astype(str).map(soil_infilt_map).fillna(0.4)
    df['soil_saturation_limit'] = df['rainfall_7d_mm'] / (df['soil_infiltration'] + 0.1)
    
    # 18. Pseudo-TWI & Flatness
    df['pseudo_twi'] = np.log1p((df['distance_to_river_m'] + 1.0) / (df['elevation_m'].clip(lower=0.0) + 1.0))
    df['flatness_index'] = df['district'].astype(str).map(district_elev_std).fillna(df['elevation_m'].std())
    
    # 19. Cyclone vulnerability flag
    df['in_cyclone_path'] = df['district'].astype(str).map(lambda x: 1 if x in cyclone_districts else 0)
    df['cyclone_vulnerability'] = df['in_cyclone_path'] * df['extreme_weather_index']
    
    # ================ BASELINE v10 FEATURES ================
    
    print("      [v10] Baseline interaction features...")
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
    
    # Isolation indices (v11)
    df['isolation_index'] = np.log1p(df['nearest_hospital_km']) + np.log1p(df['nearest_evac_km'])
    df['vulnerability'] = df['isolation_index'] / (df['infrastructure_score'] + 1.0)
    
    # Spatial bins for out-of-fold target encoding
    lat = df["latitude"].fillna(df["latitude"].median())
    lon = df["longitude"].fillna(df["longitude"].median())
    df["lat_bin"] = (lat / 0.5).astype(int)
    df["lon_bin"] = (lon / 0.5).astype(int)
    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
    
    # Remove raw skewed inundation feature (replaced by log and ratio)
    df = df.drop(columns=["inundation_area_sqm", "landcover_mean_inundation"])
    
    return df

print("\n   Engineering TRAIN features...")
train_df = engineer_features(train_df)
print("   Engineering TEST features...")
test_df  = engineer_features(test_df)

# Count new features
new_v12_features = ['inundation_ratio', 'seasonal_rain_anomaly', 'runoff_engine', 
                    'pooling_accumulator', 'socioeconomic_deficit']
print(f"\n   [NEW v12 FEATURES] {len(new_v12_features)} new features added:")
for f in new_v12_features:
    tr_stat = train_df[f].describe()
    print(f"      {f:>30}: mean={tr_stat['mean']:.4f}, std={tr_stat['std']:.4f}, "
          f"min={tr_stat['min']:.4f}, max={tr_stat['max']:.4f}")

# -----------------------------------------------------------------
# 5. COLUMN TAXONOMY & DTYPE CASTING
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

print(f"   Number of Base Features: {len(BASE_FEATURES)}")
print(f"   Categorical features   : {len([c for c in BASE_FEATURES if c in CAT_FEATURES])}")
print(f"   Numeric features       : {len([c for c in BASE_FEATURES if c not in CAT_FEATURES])}")

# -----------------------------------------------------------------
# 6. CROSS VALIDATION SETUP
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
# 7. TRAINING LOOP WITH SAMPLE WEIGHTING & CALIBRATED HYPERPARAMS
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("  5-FOLD STRATIFIED CV -- XGBoost + LightGBM + CatBoost (v12)")
print("  HYPERPARAMETER CALIBRATION: Section 9 Active")
print("    -> Shallow Trees (depth=5)  |  Micro LR (0.01)")
print("    -> Patience=150  |  Sample Weights: Real=25x  Synth=1x")
print("=" * 70)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_bins)):
    t0 = time.time()
    print(f"\n>> Fold {fold+1}/{N_FOLDS}")

    tr_rows = train_df.iloc[tr_idx].copy()
    va_rows = train_df.iloc[va_idx].copy()
    
    # Extract fold-specific sample weights
    w_tr = sample_weights[tr_idx]
    w_va = sample_weights[va_idx]
    print(f"   [WGHT] Train real rows: {(w_tr == 25.0).sum()}, synth: {(w_tr == 1.0).sum()}")

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

    # ========================================================
    # XGBoost Regressor (Section 9 calibrated hyperparameters)
    # ========================================================
    xgb_model = xgb.XGBRegressor(
        n_estimators          = 5000,
        learning_rate         = 0.01,       # Section 9: Micro LR (down from 0.05)
        max_depth             = 5,          # Section 9: Shallower (down from 7)
        min_child_weight      = 5,          # Increased regularization
        subsample             = 0.7,
        colsample_bytree      = 0.6,
        colsample_bylevel     = 0.6,
        reg_alpha             = 0.5,
        reg_lambda            = 5.0,        # Section 9: Enhanced L2
        gamma                 = 0.1,
        tree_method           = "hist",
        enable_categorical    = False,
        early_stopping_rounds = 150,        # Section 9: High patience
        random_state          = 42,
        n_jobs                = -1
    )
    xgb_model.fit(
        X_tr_xgb, y_tr,
        eval_set=[(X_va_xgb, y_va)],
        sample_weight=w_tr,                 # Section 9: Ground truth weighting
        verbose=False
    )
    oof_xgb[va_idx] = xgb_model.predict(X_va_xgb)
    tst_xgb        += xgb_model.predict(X_te_xgb) / N_FOLDS
    print(f"   [XGB] best_iter={xgb_model.best_iteration}")

    # ========================================================
    # LightGBM Regressor (Section 9 calibrated)
    # ========================================================
    lgb_model = lgb.LGBMRegressor(
        n_estimators       = 5000,
        learning_rate      = 0.01,          # Section 9: Micro LR
        num_leaves         = 31,            # Section 9: Shallower (down from 127)
        max_depth          = 5,             # Section 9: Shallower
        min_child_samples  = 30,            # Stronger regularization
        subsample          = 0.7,
        subsample_freq     = 1,
        colsample_bytree   = 0.6,
        reg_alpha          = 0.5,
        reg_lambda         = 5.0,           # Section 9: Enhanced L2
        random_state       = 42,
        n_jobs             = -1,
        verbosity          = -1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set  = [(X_va, y_va)],
        sample_weight = w_tr,               # Section 9: Ground truth weighting
        callbacks = [lgb.early_stopping(150, verbose=False), lgb.log_evaluation(-1)]
    )
    oof_lgb[va_idx] = lgb_model.predict(X_va)
    tst_lgb        += lgb_model.predict(X_te) / N_FOLDS
    print(f"   [LGB] best_iter={lgb_model.best_iteration_}")

    # ========================================================
    # CatBoost Regressor (Section 9 calibrated)
    # ========================================================
    # CatBoost requires Pool for sample weights
    cat_pool_tr = cb.Pool(
        X_tr_cat, y_tr,
        cat_features=cat_feature_names,
        weight=w_tr                         # Section 9: Ground truth weighting
    )
    cat_pool_va = cb.Pool(
        X_va_cat, y_va,
        cat_features=cat_feature_names,
        weight=w_va
    )
    
    cat_model = cb.CatBoostRegressor(
        iterations            = 5000,
        learning_rate         = 0.01,       # Section 9: Micro LR
        depth                 = 5,          # Section 9: Shallower
        l2_leaf_reg           = 10,         # Section 9: Enhanced L2
        bagging_temperature   = 0.5,
        random_strength       = 1.5,
        border_count          = 128,
        loss_function         = "RMSE",
        eval_metric           = "RMSE",
        task_type             = "CPU",
        random_seed           = 42,
        verbose               = False
    )
    cat_model.fit(
        cat_pool_tr,
        eval_set              = cat_pool_va,
        early_stopping_rounds = 150,        # Section 9: High patience
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
# 8. ENSEMBLE COMBINATION & EXPORT
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

print("\n" + "=" * 70)
print("  GLOBAL OOF RESULTS (v12 - Calibrated Ensemble)")
print("=" * 70)
print(f"    MAE            : {g_mae:.5f}")
print(f"    RMSE           : {g_rmse:.5f}")
print(f"    Explained Var. : {g_ev:.5f}")
print(f"    Pred Range     : [{oof_ensemble.min():.4f}, {oof_ensemble.max():.4f}]")
print("=" * 70)

# Feature importance (top 20 from XGBoost)
print("\n[FEAT IMPORTANCE] Top 20 Features (XGBoost, last fold):")
imp = pd.Series(xgb_model.feature_importances_, index=FEATURES)
imp_sorted = imp.sort_values(ascending=False).head(20)
for rank, (feat, score) in enumerate(imp_sorted.items(), 1):
    print(f"   {rank:>2}. {feat:<40} {score:.4f}")

# Export OOF report
fold_report = pd.DataFrame(fold_results)
fold_report.to_csv("submissions/fold_report_v12.csv", index=False)
print(f"\n[DONE] Saved fold_report_v12.csv")
print(fold_report.to_string(index=False))

# v11 comparison
print(f"\n[COMPARE] v11 Baseline: MAE=0.17984, RMSE=0.23539, EV=0.02737")
print(f"[COMPARE] v12 Current : MAE={g_mae:.5f}, RMSE={g_rmse:.5f}, EV={g_ev:.5f}")
ev_delta = g_ev - 0.02737
rmse_delta = g_rmse - 0.23539
print(f"[COMPARE] Delta EV    : {ev_delta:+.5f} ({'IMPROVED' if ev_delta > 0 else 'REGRESSED'})")
print(f"[COMPARE] Delta RMSE  : {rmse_delta:+.5f} ({'IMPROVED' if rmse_delta < 0 else 'REGRESSED'})")

# Final predictions exported with boundary preservation
tst_final = np.clip(tst_ensemble, 0.0, 1.0)
submission = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": tst_final
})
submission.to_csv("submissions/submission_v12.csv", index=False)
print(f"\n[DONE] Saved submission_v12.csv ({len(submission)} rows)")
print(f"       Pred range : [{tst_final.min():.4f}, {tst_final.max():.4f}]")
