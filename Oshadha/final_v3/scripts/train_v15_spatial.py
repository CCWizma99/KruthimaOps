"""
ML Opsidian: Genesis - Spatial Intelligence Pipeline v15
=========================================================
Inspired by re-reading the competition description:

  "flood risk in Sri Lanka is NOT random. It follows repeating
   environmental and geographical patterns that develop over time."

  NEW STRATEGIES:
  1. SPATIAL KNN RISK FEATURE: Predict each row's flood_risk from its
     K nearest geographic neighbors. Directly encodes "repeating
     geographical patterns." Computed via OOF to prevent leakage.

  2. ADVERSARIAL VALIDATION: Train a classifier to distinguish train
     vs test. Diagnose data drift. If drift exists, use P(is_test) as
     sample weights to focus models on test-like training rows.

  3. FINER SPATIAL BINS: 0.1-degree grid (~11km) instead of 0.5-degree
     (~55km) for target encoding. Captures within-district patterns.

  4. MULTI-SEED CATBOOST: 2x CatBoost-B (depth=5, best individual)
     with different seeds + CatBoost-A (depth=7) + XGBoost.
     Ridge stacking on 4 models.

  BASE: All v14 features, Bayesian TE (mean+std+count), Ridge stacking.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb  # Used only for adversarial validation
import catboost as cb
import warnings
import time

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# 1. LOAD & DEDUPLICATE
# -----------------------------------------------------------------
print("=" * 70)
print("  ML OPSIDIAN v15 - SPATIAL INTELLIGENCE PIPELINE")
print("=" * 70)
print("\n[LOAD] Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")
print(f"   Train shape      : {train_df.shape}")
print(f"   Test shape       : {test_df.shape}")
train_df = train_df.drop_duplicates()
print(f"   Train after dedup: {train_df.shape}")

# -----------------------------------------------------------------
# 2. GEOSPATIAL HOT-DECK IMPUTATION
# -----------------------------------------------------------------
print("\n[IMPUTE] Geospatial Hot-Deck Imputation...")

combined = pd.concat([
    train_df.drop(columns=['flood_risk_score'], errors='ignore'),
    test_df
], ignore_index=True)

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
print(f"   Coordinates: {imputed_coords_count} rows imputed.")

for col in ['elevation_m', 'distance_to_river_m']:
    donor_pool = combined.dropna(subset=['latitude', 'longitude', col])
    knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
    knn.fit(donor_pool[['latitude', 'longitude']], donor_pool[col])
    cnt = 0
    for df in [train_df, test_df]:
        mm = df[col].isnull() & df['latitude'].notnull() & df['longitude'].notnull()
        if mm.any():
            df.loc[mm, col] = knn.predict(df.loc[mm, ['latitude', 'longitude']])
            cnt += mm.sum()
    print(f"   {col}: {cnt} rows imputed via KNN.")

for col in ['elevation_m', 'distance_to_river_m', 'latitude', 'longitude']:
    for df in [train_df, test_df]:
        df[col] = df[col].fillna(df.groupby('district')[col].transform('median'))
        df[col] = df[col].fillna(train_df[col].median())

assert train_df[['latitude', 'longitude', 'elevation_m', 'distance_to_river_m']].isnull().sum().sum() == 0
assert test_df[['latitude', 'longitude', 'elevation_m', 'distance_to_river_m']].isnull().sum().sum() == 0

# -----------------------------------------------------------------
# 3. FEATURE ENGINEERING
# -----------------------------------------------------------------
print("\n[FEAT] Engineering features...")

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
        combined_imputed['inundation_area_sqm'].mean()
    )
    df['inundation_ratio'] = df['inundation_area_sqm'] / (df['landcover_mean_inundation_val'] + 1.0)
    
    # v15: FINER spatial bins (0.1 degree = ~11km)
    lat = df["latitude"].fillna(df["latitude"].median())
    lon = df["longitude"].fillna(df["longitude"].median())
    df["lat_bin"] = (lat / 0.5).astype(int)
    df["lon_bin"] = (lon / 0.5).astype(int)
    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
    
    df["lat_bin_fine"] = (lat / 0.1).astype(int)
    df["lon_bin_fine"] = (lon / 0.1).astype(int)
    df["grid_id_fine"] = df["lat_bin_fine"].astype(str) + "_" + df["lon_bin_fine"].astype(str)
    
    df = df.drop(columns=["inundation_area_sqm", "landcover_mean_inundation_val"])
    
    return df

train_df = engineer_features(train_df)
test_df  = engineer_features(test_df)

# -----------------------------------------------------------------
# 4. ADVERSARIAL VALIDATION (Data Drift Diagnosis)
# -----------------------------------------------------------------
print("\n[ADV] Adversarial Validation -- Diagnosing data drift...")

# Use only numeric features for adversarial validation
adv_features = [c for c in train_df.columns if c not in [
    'record_id', 'place_name', 'is_synthetic', 'generation_date',
    'flood_risk_score', 'flood_occurrence_yes',
    'lat_bin', 'lon_bin', 'grid_id', 'lat_bin_fine', 'lon_bin_fine', 'grid_id_fine',
    'downstream_sig', 'infra_deficit_sig'
] and train_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

adv_X_train = train_df[adv_features].fillna(-999)
adv_X_test  = test_df[adv_features].fillna(-999)

adv_X = pd.concat([adv_X_train, adv_X_test], ignore_index=True)
adv_y = np.array([0]*len(train_df) + [1]*len(test_df))

# Quick LightGBM classifier
adv_model = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3,
    num_leaves=8, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=-1
)

# 3-fold CV to get OOF probabilities for training rows
from sklearn.model_selection import StratifiedKFold as SKF_adv
adv_skf = SKF_adv(n_splits=3, shuffle=True, random_state=42)
adv_oof_proba = np.zeros(len(adv_X))

for adv_fold, (adv_tr, adv_va) in enumerate(adv_skf.split(adv_X, adv_y)):
    adv_model.fit(adv_X.iloc[adv_tr], adv_y[adv_tr])
    adv_oof_proba[adv_va] = adv_model.predict_proba(adv_X.iloc[adv_va])[:, 1]

# Extract train-only probabilities
train_test_proba = adv_oof_proba[:len(train_df)]
from sklearn.metrics import roc_auc_score
adv_auc = roc_auc_score(adv_y, adv_oof_proba)

print(f"   Adversarial AUC: {adv_auc:.4f}")
if adv_auc > 0.7:
    print(f"   >> SIGNIFICANT DRIFT DETECTED (AUC > 0.7)")
    print(f"   >> Applying adversarial weights to focus on test-like training rows.")
    # Normalize P(is_test) as weights: higher = more test-like = more important
    adv_weights = train_test_proba / train_test_proba.mean()  # Normalize to mean=1
    adv_weights = np.clip(adv_weights, 0.2, 5.0)  # Prevent extreme weights
    USE_ADV_WEIGHTS = True
elif adv_auc > 0.55:
    print(f"   >> Mild drift detected (AUC 0.55-0.70)")
    print(f"   >> Applying soft adversarial weights.")
    adv_weights = 0.5 + 0.5 * (train_test_proba / train_test_proba.mean())
    adv_weights = np.clip(adv_weights, 0.5, 2.0)
    USE_ADV_WEIGHTS = True
else:
    print(f"   >> No significant drift (AUC < 0.55). Skipping adversarial weights.")
    adv_weights = np.ones(len(train_df))
    USE_ADV_WEIGHTS = False

# Top drift features
adv_model.fit(adv_X, adv_y)
adv_imp = pd.Series(adv_model.feature_importances_, index=adv_features).sort_values(ascending=False)
print(f"\n   Top 10 drift features:")
for i, (feat, score) in enumerate(adv_imp.head(10).items(), 1):
    print(f"      {i:>2}. {feat:<40} {score}")

# -----------------------------------------------------------------
# 5. SPATIAL KNN RISK FEATURE (OOF to prevent leakage)
# -----------------------------------------------------------------
print("\n[SPATIAL] Computing OOF Spatial KNN Risk Feature...")

TARGET = "flood_risk_score"
y = train_df[TARGET]
GLOBAL_MEAN = float(y.mean())
GLOBAL_STD  = float(y.std())
y_bins = pd.cut(y, bins=10, labels=False)
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

KNN_NEIGHBORS = 15

# OOF spatial risk for training set
spatial_risk_oof = np.zeros(len(train_df))
spatial_risk_std_oof = np.zeros(len(train_df))

train_coords = train_df[['latitude', 'longitude']].values
train_target = y.values

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_bins)):
    knn_risk = KNeighborsRegressor(n_neighbors=KNN_NEIGHBORS, weights='distance')
    knn_risk.fit(train_coords[tr_idx], train_target[tr_idx])
    spatial_risk_oof[va_idx] = knn_risk.predict(train_coords[va_idx])

# Spatial risk for test set (fit on ALL training data)
knn_risk_full = KNeighborsRegressor(n_neighbors=KNN_NEIGHBORS, weights='distance')
knn_risk_full.fit(train_coords, train_target)
test_spatial_risk = knn_risk_full.predict(test_df[['latitude', 'longitude']].values)

train_df['spatial_risk_knn'] = spatial_risk_oof
test_df['spatial_risk_knn'] = test_spatial_risk

# Also compute spatial risk RESIDUAL (how much does this location deviate from its neighbors?)
# This captures local anomalies
train_df['spatial_risk_residual'] = train_target - spatial_risk_oof

# For test, residual is unknown but we can estimate as deviation from KNN prediction
# Actually, we can't compute this for test since we don't have the target. Skip for test.
# Instead, set to 0 (neutral)
test_df['spatial_risk_residual'] = 0.0

# Diagnostic
corr_spatial = np.corrcoef(spatial_risk_oof, train_target)[0, 1]
ev_spatial = explained_variance_score(train_target, spatial_risk_oof)
print(f"   Spatial KNN (K={KNN_NEIGHBORS}) OOF diagnostics:")
print(f"      Correlation with target : {corr_spatial:.4f}")
print(f"      Explained Variance      : {ev_spatial:.4f}")
print(f"      Prediction range        : [{spatial_risk_oof.min():.4f}, {spatial_risk_oof.max():.4f}]")

# -----------------------------------------------------------------
# 6. COLUMN TAXONOMY & DTYPE CASTING
# -----------------------------------------------------------------
ID_COL    = "record_id"
DROP_COLS = [ID_COL, "place_name", "is_synthetic", "generation_date"]

CAT_FEATURES = [
    "district", "landcover", "soil_type", "water_supply",
    "electricity", "road_quality", "urban_rural",
    "water_presence_flag", "flood_occurrence_current_event",
    "is_good_to_live", "reason_not_good_to_live",
    "downstream_sig", "infra_deficit_sig"
]

TARGET_ENC_COLS = [
    "district", "grid_id", "grid_id_fine", "downstream_sig", "infra_deficit_sig",
    "landcover", "soil_type", "water_supply", "electricity", "road_quality"
]

IGNORE_COLS = DROP_COLS + [TARGET, "flood_occurrence_yes", "spatial_risk_residual"]
SPATIAL_HELPERS = ["lat_bin", "lon_bin", "grid_id", "lat_bin_fine", "lon_bin_fine", "grid_id_fine"]

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

print(f"   Base Features: {len(BASE_FEATURES)}")
print(f"   (includes spatial_risk_knn: NEW)")

# -----------------------------------------------------------------
# 7. CV SETUP
# -----------------------------------------------------------------
MODEL_NAMES = ["XGB", "CAT-B1 (d5)", "CAT-B2 (d5)", "CAT-A (d7)"]
n_models = len(MODEL_NAMES)

oof_preds = {name: np.zeros(len(train_df)) for name in MODEL_NAMES}
tst_preds = {name: np.zeros(len(test_df))  for name in MODEL_NAMES}

fold_results = []
cat_feature_names = [c for c in CAT_FEATURES if c in BASE_FEATURES]

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
# 8. TRAINING LOOP
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("  5-FOLD CV -- XGB + 2x CatBoost(d5) + CatBoost(d7) (v15)")
print("  NEW: Spatial KNN risk + Adversarial weights + Fine-grid TE")
print("=" * 70)

SMOOTHING = 10

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_bins)):
    t0 = time.time()
    print(f"\n>> Fold {fold+1}/{N_FOLDS}")

    tr_rows = train_df.iloc[tr_idx].copy()
    va_rows = train_df.iloc[va_idx].copy()
    
    w_tr = adv_weights[tr_idx] if USE_ADV_WEIGHTS else None

    # Target encoding: mean + std + count (Bayesian smoothed)
    for col in TARGET_ENC_COLS:
        group_stats = tr_rows.groupby(col)[TARGET].agg(['mean', 'std', 'count'])
        group_stats['std'] = group_stats['std'].fillna(0.0)
        
        smoothed_mean = (group_stats['count'] * group_stats['mean'] + SMOOTHING * GLOBAL_MEAN) / (group_stats['count'] + SMOOTHING)
        smoothed_std = (group_stats['count'] * group_stats['std'] + SMOOTHING * GLOBAL_STD) / (group_stats['count'] + SMOOTHING)
        log_count = np.log1p(group_stats['count'])
        
        enc_mean = f"{col}_target_enc"
        enc_std  = f"{col}_target_std"
        enc_cnt  = f"{col}_target_cnt"
        
        for tgt_df in [tr_rows, va_rows, test_df]:
            tgt_df[enc_mean] = tgt_df[col].astype(str).map(smoothed_mean).fillna(GLOBAL_MEAN).astype(float)
            tgt_df[enc_std]  = tgt_df[col].astype(str).map(smoothed_std).fillna(GLOBAL_STD).astype(float)
            tgt_df[enc_cnt]  = tgt_df[col].astype(str).map(log_count).fillna(0.0).astype(float)

    te_features = []
    for col in TARGET_ENC_COLS:
        te_features.extend([f"{col}_target_enc", f"{col}_target_std", f"{col}_target_cnt"])
    
    FEATURES = BASE_FEATURES + te_features

    y_tr = tr_rows[TARGET]
    y_va = va_rows[TARGET]
    X_tr = tr_rows[FEATURES].copy()
    X_va = va_rows[FEATURES].copy()
    X_te = test_df[FEATURES].copy()
    
    for col in cat_feature_names:
        if col in FEATURES and col in cat_dtype_map:
            cdt = cat_dtype_map[col]
            X_tr[col] = X_tr[col].astype(str).astype(cdt)
            X_va[col] = X_va[col].astype(str).astype(cdt)
            X_te[col] = X_te[col].astype(str).astype(cdt)

    X_tr_xgb = to_xgb_fmt(X_tr); X_va_xgb = to_xgb_fmt(X_va); X_te_xgb = to_xgb_fmt(X_te)
    X_tr_cat = to_cat_fmt(X_tr);  X_va_cat = to_cat_fmt(X_va);  X_te_cat = to_cat_fmt(X_te)

    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=3000, learning_rate=0.05, max_depth=7,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.75,
        colsample_bylevel=0.75, reg_alpha=0.1, reg_lambda=1.0,
        gamma=0.05, tree_method="hist", enable_categorical=False,
        early_stopping_rounds=100, random_state=42, n_jobs=-1
    )
    xgb_fit_kwargs = {"sample_weight": w_tr} if w_tr is not None else {}
    xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False, **xgb_fit_kwargs)
    oof_preds["XGB"][va_idx] = xgb_model.predict(X_va_xgb)
    tst_preds["XGB"] += xgb_model.predict(X_te_xgb) / N_FOLDS
    print(f"   [XGB]    best_iter={xgb_model.best_iteration}")

    # --- CatBoost-B1 (depth=5, seed=42) ---
    cat_b1_pool_tr = cb.Pool(X_tr_cat, y_tr, cat_features=cat_feature_names,
                             weight=w_tr if w_tr is not None else None)
    cat_b1_pool_va = cb.Pool(X_va_cat, y_va, cat_features=cat_feature_names)
    
    cat_b1 = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.03, depth=5,
        l2_leaf_reg=5, bagging_temperature=0.8, random_strength=1.5,
        border_count=128, loss_function="RMSE", eval_metric="RMSE",
        task_type="CPU", random_seed=42, verbose=False
    )
    cat_b1.fit(cat_b1_pool_tr, eval_set=cat_b1_pool_va, early_stopping_rounds=100, verbose=False)
    oof_preds["CAT-B1 (d5)"][va_idx] = cat_b1.predict(X_va_cat)
    tst_preds["CAT-B1 (d5)"] += cat_b1.predict(X_te_cat) / N_FOLDS
    print(f"   [CAT-B1] best_iter={cat_b1.best_iteration_}  (d5, s42)")

    # --- CatBoost-B2 (depth=5, seed=123) ---
    cat_b2 = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.03, depth=5,
        l2_leaf_reg=5, bagging_temperature=0.8, random_strength=1.5,
        border_count=128, loss_function="RMSE", eval_metric="RMSE",
        task_type="CPU", random_seed=123, verbose=False
    )
    cat_b2.fit(cat_b1_pool_tr, eval_set=cat_b1_pool_va, early_stopping_rounds=100, verbose=False)
    oof_preds["CAT-B2 (d5)"][va_idx] = cat_b2.predict(X_va_cat)
    tst_preds["CAT-B2 (d5)"] += cat_b2.predict(X_te_cat) / N_FOLDS
    print(f"   [CAT-B2] best_iter={cat_b2.best_iteration_}  (d5, s123)")

    # --- CatBoost-A (depth=7, seed=456) ---
    cat_a = cb.CatBoostRegressor(
        iterations=3000, learning_rate=0.05, depth=7,
        l2_leaf_reg=3, bagging_temperature=0.5, random_strength=1,
        border_count=128, loss_function="RMSE", eval_metric="RMSE",
        task_type="CPU", random_seed=456, verbose=False
    )
    cat_a.fit(cat_b1_pool_tr, eval_set=cat_b1_pool_va, early_stopping_rounds=100, verbose=False)
    oof_preds["CAT-A (d7)"][va_idx] = cat_a.predict(X_va_cat)
    tst_preds["CAT-A (d7)"] += cat_a.predict(X_te_cat) / N_FOLDS
    print(f"   [CAT-A]  best_iter={cat_a.best_iteration_}  (d7, s456)")

    # Fold metrics
    oof_avg_fold = np.mean([oof_preds[m][va_idx] for m in MODEL_NAMES], axis=0)
    y_va_arr = y_va.values
    f_mae  = mean_absolute_error(y_va_arr, oof_avg_fold)
    f_rmse = root_mean_squared_error(y_va_arr, oof_avg_fold)
    f_ev   = explained_variance_score(y_va_arr, oof_avg_fold)
    fold_results.append({"fold": fold+1, "MAE": f_mae, "RMSE": f_rmse, "EV": f_ev})
    print(f"   [ENS-4]  MAE={f_mae:.4f}  RMSE={f_rmse:.4f}  EV={f_ev:.4f}  [{time.time()-t0:.0f}s]")

# -----------------------------------------------------------------
# 9. LEVEL-2 STACKING
# -----------------------------------------------------------------
print("\n" + "-" * 70)
print("  LEVEL-2 STACKING: Ridge Meta-Learner on 4 Models")
print("-" * 70)

y_arr = y.values
oof_meta = np.column_stack([oof_preds[m] for m in MODEL_NAMES])
tst_meta = np.column_stack([tst_preds[m] for m in MODEL_NAMES])

print(f"\n   [INDIVIDUAL MODEL PERFORMANCE]")
for name in MODEL_NAMES:
    m_rmse = root_mean_squared_error(y_arr, oof_preds[name])
    m_ev   = explained_variance_score(y_arr, oof_preds[name])
    print(f"      {name:<15}: RMSE={m_rmse:.5f}  EV={m_ev:.5f}")

# Ridge stacking with CV
oof_stacked = np.zeros(len(train_df))
tst_stacked_accum = np.zeros(len(test_df))
for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_bins)):
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(oof_meta[tr_idx], y_arr[tr_idx])
    oof_stacked[va_idx] = ridge.predict(oof_meta[va_idx])
    tst_stacked_accum += ridge.predict(tst_meta) / N_FOLDS

ridge_final = Ridge(alpha=1.0, fit_intercept=True)
ridge_final.fit(oof_meta, y_arr)
tst_stacked = ridge_final.predict(tst_meta)

print(f"\n   [RIDGE COEFFICIENTS]")
for i, name in enumerate(MODEL_NAMES):
    print(f"      {name:<15}: {ridge_final.coef_[i]:.4f}")
print(f"      {'Intercept':<15}: {ridge_final.intercept_:.4f}")

# Blend comparison
print(f"\n   [BLEND COMPARISON]")
methods = {}

oof_avg = np.mean([oof_preds[m] for m in MODEL_NAMES], axis=0)
tst_avg = np.mean([tst_preds[m] for m in MODEL_NAMES], axis=0)
methods["Simple Average (4)"] = (oof_avg, tst_avg)

rmses = {m: root_mean_squared_error(y_arr, oof_preds[m]) for m in MODEL_NAMES}
weights = {m: 1.0/rmses[m] for m in MODEL_NAMES}
total_w = sum(weights.values())
oof_invw = sum(weights[m] * oof_preds[m] for m in MODEL_NAMES) / total_w
tst_invw = sum(weights[m] * tst_preds[m] for m in MODEL_NAMES) / total_w
methods["Inverse-RMSE (4)"] = (oof_invw, tst_invw)

methods["Ridge Stacked"] = (oof_stacked, tst_stacked)
methods["Ridge Stacked CV"] = (oof_stacked, tst_stacked_accum)

cat_names = [m for m in MODEL_NAMES if m.startswith("CAT")]
oof_cats = np.mean([oof_preds[m] for m in cat_names], axis=0)
tst_cats = np.mean([tst_preds[m] for m in cat_names], axis=0)
methods["CatBoost-Only Avg (3)"] = (oof_cats, tst_cats)

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
    print(f"      {name:>25}: MAE={m_mae:.5f}  RMSE={m_rmse:.5f}  EV={m_ev:.5f}{winner}")

print(f"\n   [SELECTED] {best_method}")

# -----------------------------------------------------------------
# 10. FINAL RESULTS & EXPORT
# -----------------------------------------------------------------
g_mae  = mean_absolute_error(y_arr, best_oof)
g_rmse = root_mean_squared_error(y_arr, best_oof)
g_ev   = explained_variance_score(y_arr, best_oof)

print("\n" + "=" * 70)
print("  GLOBAL OOF RESULTS (v15 - Spatial Intelligence)")
print("=" * 70)
print(f"    MAE            : {g_mae:.5f}")
print(f"    RMSE           : {g_rmse:.5f}")
print(f"    Explained Var. : {g_ev:.5f}")
print(f"    Pred Range     : [{best_oof.min():.4f}, {best_oof.max():.4f}]")
print("=" * 70)

# Feature importance
print(f"\n[FEAT IMPORTANCE] Top 25 Features (XGBoost, last fold):")
imp_final = pd.Series(xgb_model.feature_importances_, index=FEATURES)
for rank, (feat, score) in enumerate(imp_final.sort_values(ascending=False).head(25).items(), 1):
    marker = "** NEW **" if feat in ['spatial_risk_knn', 'grid_id_fine_target_enc', 'grid_id_fine_target_std', 'grid_id_fine_target_cnt'] else ""
    print(f"   {rank:>2}. {feat:<45} {score:.4f}  {marker}")

# Export
fold_report = pd.DataFrame(fold_results)
fold_report.to_csv("submissions/fold_report_v15.csv", index=False)
print(f"\n[DONE] Saved fold_report_v15.csv")
print(fold_report.to_string(index=False))

print(f"\n[COMPARE] v11 Baseline : MAE=0.17984, RMSE=0.23539, EV=0.02737  (LB: 0.38637)")
print(f"[COMPARE] v13          : MAE=0.17937, RMSE=0.23500, EV=0.03060  (LB: 0.38476)")
print(f"[COMPARE] v14          : MAE=0.17920, RMSE=0.23497, EV=0.03085")
print(f"[COMPARE] v15 Current  : MAE={g_mae:.5f}, RMSE={g_rmse:.5f}, EV={g_ev:.5f}")
ev_delta = g_ev - 0.03085
rmse_delta = g_rmse - 0.23497
print(f"[COMPARE] v15 vs v14 EV   : {ev_delta:+.5f} ({'IMPROVED' if ev_delta > 0 else 'REGRESSED'})")
print(f"[COMPARE] v15 vs v14 RMSE : {rmse_delta:+.5f} ({'IMPROVED' if rmse_delta < 0 else 'REGRESSED'})")

tst_final = np.clip(best_tst, 0.0, 1.0)
submission = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": tst_final
})
submission.to_csv("submissions/submission_v15.csv", index=False)
print(f"\n[DONE] Saved submission_v15.csv ({len(submission)} rows)")
print(f"       Pred range : [{tst_final.min():.4f}, {tst_final.max():.4f}]")
