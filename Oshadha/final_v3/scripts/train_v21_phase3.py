"""
ML Opsidian: Genesis v21 - Phase 3 (Representation Learning & Advanced MLOps)
=============================================================================
Enhancements over v20:
1. 3D Topographical Clustering (Replaces grid_id with K-Means on lat/lon/elev)
2. Denoising Autoencoder (Extracts 16-dim bottleneck from corrupted inputs)
3. RankGauss Target Normalization (Forces target to normal distribution for trees)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import catboost as cb
import warnings
import time
import os

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# 1. LOAD & DEDUPLICATE
# -----------------------------------------------------------------
print("=" * 70)
print("  ML OPSIDIAN v21 - PHASE 3 ADVANCED MLOPS")
print("=" * 70)
print("\n[LOAD] Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")
train_df = train_df.drop_duplicates()
print(f"   Train: {train_df.shape}  Test: {test_df.shape}")

# -----------------------------------------------------------------
# 1.5. SYNTHETIC FINGERPRINT & PSEUDO-LABELING
# -----------------------------------------------------------------
print("\n[FEAT] Extracting precision fingerprint...")
for df in [train_df, test_df]:
    df['lat_decimal_len'] = df['latitude'].apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) and str(x).lower() != 'nan' else 0)
    df['lon_decimal_len'] = df['longitude'].apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) and str(x).lower() != 'nan' else 0)

print("\n[SEMI-SUPERVISED] Pseudo-Labeling from v20...")
if os.path.exists("submissions/submission_v20.csv"):
    sub_v20 = pd.read_csv("submissions/submission_v20.csv")
    test_pseudo = test_df.merge(sub_v20, on="record_id", how="left")
    
    # Filter highly confident predictions (around median)
    mask = (test_pseudo['flood_risk_score'] >= 0.46) & (test_pseudo['flood_risk_score'] <= 0.49)
    pseudo_rows = test_pseudo[mask].copy()
    pseudo_rows['is_pseudo'] = 1
    train_df['is_pseudo'] = 0
    test_df['is_pseudo'] = 0
    
    print(f"   Added {len(pseudo_rows)} pseudo-labeled rows to training.")
    train_df = pd.concat([train_df, pseudo_rows], ignore_index=True)
else:
    print("   [WARNING] submission_v20.csv not found. Skipping pseudo-labeling.")
    train_df['is_pseudo'] = 0
    test_df['is_pseudo'] = 0

# -----------------------------------------------------------------
# 2. GEOSPATIAL HOT-DECK IMPUTATION
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
# 2.5. 3D TOPOGRAPHICAL CLUSTERING (Phase 3)
# -----------------------------------------------------------------
print("\n[FEAT] 3D Topographical Clustering...")
topo_features = ['latitude', 'longitude', 'elevation_m']
scaler_topo = StandardScaler()
X_topo_train = scaler_topo.fit_transform(train_df[topo_features])
X_topo_test = scaler_topo.transform(test_df[topo_features])

kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
train_df['topo_cluster_id'] = kmeans.fit_predict(X_topo_train)
test_df['topo_cluster_id'] = kmeans.predict(X_topo_test)

# -----------------------------------------------------------------
# 3. FEATURE ENGINEERING (v14 proven set + Fingerprint)
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
    "is_good_to_live", "reason_not_good_to_live",
    "downstream_sig", "infra_deficit_sig"
]
TARGET_ENC_COLS = [
    "district", "topo_cluster_id", "downstream_sig", "infra_deficit_sig",
    "landcover", "soil_type", "water_supply", "electricity", "road_quality"
]
IGNORE_COLS = DROP_COLS + [TARGET, "flood_occurrence_yes"]
SPATIAL_HELPERS = ["topo_cluster_id"]
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

# -----------------------------------------------------------------
# 4.5. DENOISING AUTOENCODER (DAE) BTLNECK EXTRACTION (Phase 3)
# -----------------------------------------------------------------
print("\n[FEAT] Denoising Autoencoder (DAE) Bottleneck Extraction...")
dae_features = [c for c in BASE_FEATURES if c not in CAT_FEATURES]
X_dae_all = np.vstack([train_df[dae_features].values, test_df[dae_features].values])

scaler_dae = StandardScaler()
X_dae_scaled = scaler_dae.fit_transform(X_dae_all)

# Corrupt input with 15% random swap noise
np.random.seed(42)
X_dae_corrupted = X_dae_scaled.copy()
mask = np.random.rand(*X_dae_scaled.shape) < 0.15
random_idx = np.random.randint(0, X_dae_scaled.shape[0], size=X_dae_scaled.shape)
for j in range(X_dae_scaled.shape[1]):
    X_dae_corrupted[mask[:, j], j] = X_dae_scaled[random_idx[mask[:, j], j], j]

dae_mlp = MLPRegressor(hidden_layer_sizes=(64, 16, 64), max_iter=50, random_state=42)
dae_mlp.fit(X_dae_corrupted, X_dae_scaled)

def get_bottleneck(X, mlp):
    a1 = np.maximum(0, np.dot(X, mlp.coefs_[0]) + mlp.intercepts_[0])
    a2 = np.maximum(0, np.dot(a1, mlp.coefs_[1]) + mlp.intercepts_[1])
    return a2

X_dae_train_bottleneck = get_bottleneck(scaler_dae.transform(train_df[dae_features].values), dae_mlp)
X_dae_test_bottleneck = get_bottleneck(scaler_dae.transform(test_df[dae_features].values), dae_mlp)

for i in range(16):
    train_df[f"dae_bottleneck_{i}"] = X_dae_train_bottleneck[:, i]
    test_df[f"dae_bottleneck_{i}"] = X_dae_test_bottleneck[:, i]
    BASE_FEATURES.append(f"dae_bottleneck_{i}")

cat_feature_names = [c for c in CAT_FEATURES if c in BASE_FEATURES]
print(f"   Base features (with DAE): {len(BASE_FEATURES)}")

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
# 5. MODEL CONFIGS
# -----------------------------------------------------------------
MODEL_NAMES = [
    "XGB-MAE (d7)",
    "CAT-MAE-1 (d5)",
    "CAT-MAE-2 (d5)",
    "CAT-RMSE (d5)",
]

# -----------------------------------------------------------------
# 6. CV LOOP & RANKGAUSS
# -----------------------------------------------------------------
N_FOLDS     = 5
y           = train_df[TARGET]

# RankGauss Transformation (Phase 3)
print("\n[PREP] Fitting RankGauss Target Normalizer...")
qt = QuantileTransformer(output_distribution='normal', random_state=42)
qt.fit(y.values.reshape(-1, 1))

GLOBAL_MEAN = float(y.mean())
GLOBAL_STD  = float(y.std())
GLOBAL_Q25  = float(y.quantile(0.25))
GLOBAL_Q75  = float(y.quantile(0.75))

# GroupKFold on topo_cluster_id
gkf = GroupKFold(n_splits=N_FOLDS)
groups = train_df['topo_cluster_id'].values
SMOOTHING   = 10

oof_preds = {m: np.zeros(len(train_df)) for m in MODEL_NAMES}
tst_preds = {m: np.zeros(len(test_df))  for m in MODEL_NAMES}
fold_results = []

print("\n" + "=" * 70)
print("  5-FOLD TOPO-GROUP CV WITH RANKGAUSS & DAE")
print("=" * 70)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, groups)):
    t0 = time.time()
    print(f"\n>> Fold {fold+1}/{N_FOLDS}")

    va_is_pseudo = train_df.iloc[va_idx]['is_pseudo'] == 1
    if va_is_pseudo.any():
        va_idx_clean = va_idx[~va_is_pseudo]
    else:
        va_idx_clean = va_idx
        
    tr_rows = train_df.iloc[tr_idx].copy()
    va_rows = train_df.iloc[va_idx_clean].copy()

    # Target encodings (on ORIGINAL target)
    real_tr_rows = tr_rows[tr_rows['is_pseudo'] == 0]
    for col in TARGET_ENC_COLS:
        group_stats = real_tr_rows.groupby(col)[TARGET].agg(
            mean='mean', std='std', count='count', 
            q25=lambda x: x.quantile(0.25), 
            q75=lambda x: x.quantile(0.75)
        )
        group_stats['std'] = group_stats['std'].fillna(0.0)
        
        smoothed_mean = (group_stats['count'] * group_stats['mean'] + SMOOTHING * GLOBAL_MEAN) / (group_stats['count'] + SMOOTHING)
        smoothed_std  = (group_stats['count'] * group_stats['std'] + SMOOTHING * GLOBAL_STD) / (group_stats['count'] + SMOOTHING)
        smoothed_q25  = (group_stats['count'] * group_stats['q25'] + SMOOTHING * GLOBAL_Q25) / (group_stats['count'] + SMOOTHING)
        smoothed_q75  = (group_stats['count'] * group_stats['q75'] + SMOOTHING * GLOBAL_Q75) / (group_stats['count'] + SMOOTHING)
        log_count = np.log1p(group_stats['count'])
        
        for tgt_df in [tr_rows, va_rows, test_df]:
            tgt_df[f"{col}_target_enc"] = tgt_df[col].astype(str).map(smoothed_mean).fillna(GLOBAL_MEAN).astype(float)
            tgt_df[f"{col}_target_std"] = tgt_df[col].astype(str).map(smoothed_std).fillna(GLOBAL_STD).astype(float)
            tgt_df[f"{col}_target_q25"] = tgt_df[col].astype(str).map(smoothed_q25).fillna(GLOBAL_Q25).astype(float)
            tgt_df[f"{col}_target_q75"] = tgt_df[col].astype(str).map(smoothed_q75).fillna(GLOBAL_Q75).astype(float)
            tgt_df[f"{col}_target_cnt"] = tgt_df[col].astype(str).map(log_count).fillna(0.0).astype(float)

    te_features = []
    for col in TARGET_ENC_COLS:
        te_features.extend([f"{col}_target_enc", f"{col}_target_std", f"{col}_target_q25", f"{col}_target_q75", f"{col}_target_cnt"])
    FEATURES = BASE_FEATURES + te_features

    y_tr_orig, y_va_orig = tr_rows[TARGET], va_rows[TARGET]
    
    # Apply RankGauss for training
    y_tr_rg = qt.transform(y_tr_orig.values.reshape(-1, 1)).flatten()
    
    X_tr, X_va, X_te = tr_rows[FEATURES].copy(), va_rows[FEATURES].copy(), test_df[FEATURES].copy()

    for col in cat_feature_names:
        if col in FEATURES and col in cat_dtype_map:
            cdt = cat_dtype_map[col]
            X_tr[col] = X_tr[col].astype(str).astype(cdt)
            X_va[col] = X_va[col].astype(str).astype(cdt)
            X_te[col] = X_te[col].astype(str).astype(cdt)

    X_tr_xgb, X_va_xgb, X_te_xgb = to_xgb_fmt(X_tr), to_xgb_fmt(X_va), to_xgb_fmt(X_te)
    X_tr_cat, X_va_cat, X_te_cat  = to_cat_fmt(X_tr), to_cat_fmt(X_va), to_cat_fmt(X_te)
    cat_pool_tr = cb.Pool(X_tr_cat, y_tr_rg, cat_features=cat_feature_names)
    
    # We do NOT use eval_set for CatBoost because evaluating MAE on RankGauss scale might stop early wrongly.
    # However, keeping it on RankGauss scale for early stopping is fine since it's monotonic.
    y_va_rg = qt.transform(y_va_orig.values.reshape(-1, 1)).flatten()
    cat_pool_va = cb.Pool(X_va_cat, y_va_rg, cat_features=cat_feature_names)

    # === 1. XGBoost ===
    xgb_mae = xgb.XGBRegressor(
        n_estimators=3000, learning_rate=0.05, max_depth=7,
        objective='reg:absoluteerror', 
        min_child_weight=3, subsample=0.8, colsample_bytree=0.75,
        tree_method="hist", early_stopping_rounds=100, random_state=42, n_jobs=-1,
        eval_metric='mae'
    )
    xgb_mae.fit(X_tr_xgb, y_tr_rg, eval_set=[(X_va_xgb, y_va_rg)], verbose=False)
    
    # Predict and inverse transform
    oof_preds["XGB-MAE (d7)"][va_idx_clean] = qt.inverse_transform(xgb_mae.predict(X_va_xgb).reshape(-1, 1)).flatten()
    tst_preds["XGB-MAE (d7)"] += qt.inverse_transform(xgb_mae.predict(X_te_xgb).reshape(-1, 1)).flatten() / N_FOLDS
    print(f"   [XGB-MAE]     iter={xgb_mae.best_iteration}")

    # === 2. CatBoost-MAE-1 ===
    cat_mae1 = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.03, depth=5,
        l2_leaf_reg=5, loss_function="MAE", eval_metric="MAE",
        random_seed=42, verbose=False
    )
    cat_mae1.fit(cat_pool_tr, eval_set=cat_pool_va, early_stopping_rounds=100, verbose=False)
    oof_preds["CAT-MAE-1 (d5)"][va_idx_clean] = qt.inverse_transform(cat_mae1.predict(X_va_cat).reshape(-1, 1)).flatten()
    tst_preds["CAT-MAE-1 (d5)"] += qt.inverse_transform(cat_mae1.predict(X_te_cat).reshape(-1, 1)).flatten() / N_FOLDS
    print(f"   [CAT-MAE-1]   iter={cat_mae1.best_iteration_}")

    # === 3. CatBoost-MAE-2 ===
    cat_mae2 = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.05, depth=5,
        l2_leaf_reg=5, loss_function="MAE", eval_metric="MAE",
        random_seed=789, verbose=False
    )
    cat_mae2.fit(cat_pool_tr, eval_set=cat_pool_va, early_stopping_rounds=100, verbose=False)
    oof_preds["CAT-MAE-2 (d5)"][va_idx_clean] = qt.inverse_transform(cat_mae2.predict(X_va_cat).reshape(-1, 1)).flatten()
    tst_preds["CAT-MAE-2 (d5)"] += qt.inverse_transform(cat_mae2.predict(X_te_cat).reshape(-1, 1)).flatten() / N_FOLDS
    print(f"   [CAT-MAE-2]   iter={cat_mae2.best_iteration_}")

    # === 4. CatBoost-RMSE ===
    cat_rmse = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.03, depth=5,
        l2_leaf_reg=5, loss_function="RMSE", eval_metric="RMSE",
        random_seed=123, verbose=False
    )
    cat_rmse.fit(cat_pool_tr, eval_set=cat_pool_va, early_stopping_rounds=100, verbose=False)
    oof_preds["CAT-RMSE (d5)"][va_idx_clean] = qt.inverse_transform(cat_rmse.predict(X_va_cat).reshape(-1, 1)).flatten()
    tst_preds["CAT-RMSE (d5)"] += qt.inverse_transform(cat_rmse.predict(X_te_cat).reshape(-1, 1)).flatten() / N_FOLDS
    print(f"   [CAT-RMSE]    iter={cat_rmse.best_iteration_}")

    # Fold summary (on ORIGINAL scale)
    oof_avg_fold = np.mean([oof_preds[m][va_idx_clean] for m in MODEL_NAMES], axis=0)
    y_va_arr = y_va_orig.values
    f_mae  = mean_absolute_error(y_va_arr, oof_avg_fold)
    f_rmse = root_mean_squared_error(y_va_arr, oof_avg_fold)
    f_ev   = explained_variance_score(y_va_arr, oof_avg_fold)
    fold_results.append({"fold": fold+1, "MAE": f_mae, "RMSE": f_rmse, "EV": f_ev})
    print(f"   [ENS-4]       MAE={f_mae:.4f}  RMSE={f_rmse:.4f}  EV={f_ev:.4f}  [{time.time()-t0:.0f}s]")

# -----------------------------------------------------------------
# 7. RIDGE STACKING + BLENDING
# -----------------------------------------------------------------
print("\n" + "-" * 70)
print("  LEVEL-2: Ridge Stacking + Blend Comparison")
print("-" * 70)

real_mask = train_df['is_pseudo'] == 0
y_arr = y[real_mask].values
oof_meta = np.column_stack([oof_preds[m][real_mask] for m in MODEL_NAMES])
tst_meta = np.column_stack([tst_preds[m] for m in MODEL_NAMES])

print(f"\n   [INDIVIDUAL MODEL PERFORMANCE]")
for name in MODEL_NAMES:
    m_rmse = root_mean_squared_error(y_arr, oof_preds[name][real_mask])
    m_mae  = mean_absolute_error(y_arr, oof_preds[name][real_mask])
    m_ev   = explained_variance_score(y_arr, oof_preds[name][real_mask])
    pred_lb = -22.87 * m_mae + 6.60 * m_rmse + 3.03 * (1 - m_ev)
    print(f"      {name:<18}: MAE={m_mae:.5f}  RMSE={m_rmse:.5f}  EV={m_ev:.5f}  est_LB={pred_lb:.5f}")

oof_stacked = np.zeros(len(train_df))
tst_stacked_accum = np.zeros(len(test_df))
for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, groups)):
    va_is_pseudo = train_df.iloc[va_idx]['is_pseudo'] == 1
    va_idx_clean = va_idx[~va_is_pseudo]
    tr_is_pseudo = train_df.iloc[tr_idx]['is_pseudo'] == 1
    tr_idx_clean = tr_idx[~tr_is_pseudo]
    
    oof_meta_all = np.column_stack([oof_preds[m] for m in MODEL_NAMES])
    y_all = y.values
    
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(oof_meta_all[tr_idx_clean], y_all[tr_idx_clean])
    oof_stacked[va_idx_clean] = ridge.predict(oof_meta_all[va_idx_clean])
    tst_stacked_accum += ridge.predict(tst_meta) / N_FOLDS

ridge_final = Ridge(alpha=1.0, fit_intercept=True)
ridge_final.fit(oof_meta, y_arr)
tst_stacked = ridge_final.predict(tst_meta)

print(f"\n   [RIDGE COEFFICIENTS]")
for i, name in enumerate(MODEL_NAMES):
    print(f"      {name:<18}: {ridge_final.coef_[i]:.4f}")
print(f"      {'Intercept':<18}: {ridge_final.intercept_:.4f}")

print(f"\n   [BLEND COMPARISON] (with estimated LB from formula)")
methods = {}

oof_avg = np.mean([oof_preds[m][real_mask] for m in MODEL_NAMES], axis=0)
tst_avg = np.mean([tst_preds[m] for m in MODEL_NAMES], axis=0)
methods["Simple Avg (4)"] = (oof_avg, tst_avg)

methods["Ridge Stacked"] = (oof_stacked[real_mask], tst_stacked)
methods["Ridge Stacked CV"] = (oof_stacked[real_mask], tst_stacked_accum)

mae_names = [m for m in MODEL_NAMES if "MAE" in m]
oof_mae = np.mean([oof_preds[m][real_mask] for m in mae_names], axis=0)
tst_mae = np.mean([tst_preds[m] for m in mae_names], axis=0)
methods["MAE-Only Avg (3)"] = (oof_mae, tst_mae)

best_method = None
best_lb_est = 999
for name, (oof_pred, tst_pred) in methods.items():
    m_mae  = mean_absolute_error(y_arr, oof_pred)
    m_rmse = root_mean_squared_error(y_arr, oof_pred)
    m_ev   = explained_variance_score(y_arr, oof_pred)
    est_lb = -22.87 * m_mae + 6.60 * m_rmse + 3.03 * (1 - m_ev)
    winner = ""
    if est_lb < best_lb_est:
        best_lb_est = est_lb
        best_method = name
        best_oof = oof_pred
        best_tst = tst_pred
        winner = " <-- BEST"
    print(f"      {name:>20}: MAE={m_mae:.5f}  RMSE={m_rmse:.5f}  EV={m_ev:.5f}  est_LB={est_lb:.5f}{winner}")

print(f"\n   [SELECTED] {best_method}")

# -----------------------------------------------------------------
# 8. FINAL RESULTS
# -----------------------------------------------------------------
g_mae  = mean_absolute_error(y_arr, best_oof)
g_rmse = root_mean_squared_error(y_arr, best_oof)
g_ev   = explained_variance_score(y_arr, best_oof)
g_lb   = -22.87 * g_mae + 6.60 * g_rmse + 3.03 * (1 - g_ev)

print("\n" + "=" * 70)
print("  GLOBAL OOF RESULTS (v21 - Phase 3)")
print("=" * 70)
print(f"    MAE            : {g_mae:.5f}")
print(f"    RMSE           : {g_rmse:.5f}")
print(f"    Explained Var. : {g_ev:.5f}")
print(f"    Est. LB Score  : {g_lb:.5f}")
print(f"    Pred Range     : [{best_oof.min():.4f}, {best_oof.max():.4f}]")
print("=" * 70)

fold_report = pd.DataFrame(fold_results)
fold_report.to_csv("submissions/fold_report_v21.csv", index=False)
print(f"\n[DONE] Saved fold_report_v21.csv")

tst_final = np.clip(best_tst, 0.0, 1.0)
submission = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": tst_final
})
submission.to_csv("submissions/submission_v21.csv", index=False)
print(f"\n[DONE] Saved submission_v21.csv ({len(submission)} rows)")

np.save("submissions/oof_v21.npy", best_oof)
print(f"[DONE] Saved oof_v21.npy (for evaluate.py)")
