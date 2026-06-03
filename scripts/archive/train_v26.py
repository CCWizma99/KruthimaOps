"""
ML Opsidian: Genesis v26 - Adversarial Metric Alignment
========================================================================
4. CatBoost max_ctr_complexity=2 (Internal Interactions)
5. Multi-Seed Averaging (Variance Reduction, seeds: 42, 123, 456, 789)
6. Removed Pseudo-Labeling entirely (Pure Organic Focus)
7. Positive constraint on Level-2 stacker via LinearRegression
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, LinearRegression
import xgboost as xgb
import catboost as cb
import warnings
import time
import os
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# 1. LOAD & DEDUPLICATE
# -----------------------------------------------------------------
print("=" * 70)
print("  ML OPSIDIAN v26 - ADVERSARIAL METRIC ALIGNMENT")
print("=" * 70)
print("\n[LOAD] Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")
train_df = train_df.drop_duplicates()
print(f"   Train: {train_df.shape}  Test: {test_df.shape}")

# -----------------------------------------------------------------
# 1.5. SYNTHETIC FINGERPRINT & PSEUDO-LABELING
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

# --- Strategy 1: "Snap Features" Reconstruction Matrix ---
# We isolate the real records first
real_nodes = train_df[train_df['is_synthetic'].isna()][['latitude', 'longitude']].dropna().values
spatial_tree = cKDTree(real_nodes)

def engineer_features(df):
    df = df.copy()
    
    # 1. Snap Features
    coords = df[['latitude', 'longitude']].values
    distances, indices = spatial_tree.query(coords, k=1)
    
    df['snapped_lat'] = real_nodes[indices, 0]
    df['snapped_lon'] = real_nodes[indices, 1]
    
    df['lat_perturbation_noise'] = df['latitude'] - df['snapped_lat']
    df['lon_perturbation_noise'] = df['longitude'] - df['snapped_lon']
    df['spatial_perturbation_magnitude'] = distances

    # 2. Strategy 2: Multi-Scale Decimal Digit Extraction
    for col in ['inundation_area_sqm', 'latitude', 'longitude', 'ndvi_qmap', 'ndwi_qmap']:
        if col in df.columns:
            frac = np.abs(df[col] - np.floor(df[col]))
            df[f'{col}_dec_d1'] = np.floor(frac * 10)
            df[f'{col}_dec_d2'] = np.floor(frac * 100) % 10
            df[f'{col}_dec_d3'] = np.floor(frac * 1000) % 10
            df[f'{col}_is_perfect_round'] = ((frac < 0.001) | (frac > 0.999)).astype(int)
            df[f'{col}_mod_quarter'] = frac % 0.25
            df[f'{col}_mod_tenth'] = frac % 0.10

    
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
DROP_COLS = [ID_COL, "place_name", "is_synthetic", "generation_date"]
CAT_FEATURES = [
    "district", "landcover", "soil_type", "water_supply",
    "electricity", "road_quality", "urban_rural",
    "water_presence_flag", "flood_occurrence_current_event",
    "is_good_to_live", "reason_not_good_to_live"
]

TARGET_ENC_COLS = [
    "district", "downstream_sig", "infra_deficit_sig",
    "landcover", "soil_type", "water_supply", "electricity", "road_quality"
]

COMPOSITE_ENC_COLS = []

STD_ENC_COLS = [
    "district", "downstream_sig"
]

IGNORE_COLS = DROP_COLS + [TARGET, "flood_occurrence_yes",
                           "downstream_sig", "infra_deficit_sig"]
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


def joint_mae_rmse_objective(y_true, y_pred):
    """
    Custom objective for XGBoost. 
    Signature for XGBRegressor in Scikit-Learn API is (y_true, y_pred).
    """
    residual = y_pred - y_true

    
    alpha = 13.2460  # MAE Weight
    beta = 4.6735    # RMSE/MSE Weight
    delta = 1e-3     # Pseudo-Huber smoothing
    
    grad_mae = alpha * (residual / np.sqrt(residual**2 + delta))
    grad_rmse = 2 * beta * residual
    gradient = grad_mae + grad_rmse
    
    hess_mae = alpha * (delta / (residual**2 + delta)**(1.5))
    hess_rmse = 2 * beta * np.ones_like(residual)
    hessian = hess_mae + hess_rmse
    
    return gradient, hessian

# -----------------------------------------------------------------
# 5. MODEL CONFIGS & MULTI-SEED SETUP
# -----------------------------------------------------------------
MODEL_NAMES = [
    "XGB-Custom (d7)",
    "CAT-MAE-1 (d5)",
    "CAT-MAE-2 (d5)",
    "CAT-RMSE (d5)",
]

N_FOLDS     = 5
y           = train_df[TARGET]
GLOBAL_MEAN = float(y.mean())
GLOBAL_STD  = float(y.std())
GLOBAL_Q25  = float(y.quantile(0.25))
GLOBAL_Q75  = float(y.quantile(0.75))
GLOBAL_MEDIAN = float(y.median())

# GroupKFold on grid_id for spatial strat
gkf = GroupKFold(n_splits=N_FOLDS)
groups = train_df['grid_id'].values
SMOOTHING   = 10
SMOOTHING_COMPOSITE = 15

# Evaluate only on REAL training data, not pseudo labels
y_arr = y.values

SEEDS = [42, 123, 456, 789]

oof_predictions_accum = np.zeros(len(train_df))
test_predictions_accum = np.zeros(len(test_df))

print("\n" + "=" * 70)
print(f"  MULTI-SEED TOPO-CV TRAINING ({len(SEEDS)} SEEDS * {N_FOLDS} FOLDS)")
print("=" * 70)

total_fits = len(SEEDS) * N_FOLDS * len(MODEL_NAMES)
current_fit = 0
t_start_global = time.time()

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n>>>> SEED {seed_idx+1}/{len(SEEDS)} (Seed value: {seed})")
    
    oof_preds = {m: np.zeros(len(train_df)) for m in MODEL_NAMES}
    tst_preds = {m: np.zeros(len(test_df))  for m in MODEL_NAMES}
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, groups)):
        t0 = time.time()
        
        # Pure organic validation
        va_idx_clean = va_idx
        tr_rows = train_df.iloc[tr_idx].copy()
        va_rows = train_df.iloc[va_idx_clean].copy()

        # Target encodings
        real_tr_rows = tr_rows        # Combine all columns that need median encoding
        all_te_cols = TARGET_ENC_COLS + COMPOSITE_ENC_COLS
        
        for col in all_te_cols:
            # 1. Median-Based Target Encoding
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
                
                # 2. Target Variance Feature (Uncertainty Calibration) for specific columns
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
        cat_pool_tr = cb.Pool(X_tr_cat, y_tr, cat_features=cat_feature_names)
        cat_pool_va = cb.Pool(X_va_cat, y_va, cat_features=cat_feature_names)

        # === 1. XGBoost with Custom Metric-Driven Objective ===
        xgb_mae = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.05, max_depth=7,
            objective=joint_mae_rmse_objective, 
            min_child_weight=30, subsample=0.8, colsample_bytree=0.75,
            tree_method="hist", early_stopping_rounds=100, random_state=seed, n_jobs=-1,
            eval_metric='mae'
        )
        xgb_mae.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
        oof_preds["XGB-Custom (d7)"][va_idx_clean] = xgb_mae.predict(X_va_xgb)
        tst_preds["XGB-Custom (d7)"] += xgb_mae.predict(X_te_xgb) / N_FOLDS

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

        # === 4. CatBoost-RMSE ===
        cat_rmse = cb.CatBoostRegressor(
            iterations=5000, learning_rate=0.03, depth=5,
            l2_leaf_reg=5, loss_function="RMSE", eval_metric="RMSE",
            max_ctr_complexity=2,
            random_seed=seed + 200, verbose=False
        )
        cat_rmse.fit(cat_pool_tr, eval_set=cat_pool_va, early_stopping_rounds=100, verbose=False)
        oof_preds["CAT-RMSE (d5)"][va_idx_clean] = cat_rmse.predict(X_va_cat)
        tst_preds["CAT-RMSE (d5)"] += cat_rmse.predict(X_te_cat) / N_FOLDS

        oof_avg_fold = np.mean([oof_preds[m][va_idx_clean] for m in MODEL_NAMES], axis=0)
        y_va_arr = y_va.values
        f_mae  = mean_absolute_error(y_va_arr, oof_avg_fold)
        f_rmse = root_mean_squared_error(y_va_arr, oof_avg_fold)
        f_ev   = explained_variance_score(y_va_arr, oof_avg_fold)
        print(f"      Fold {fold+1}/{N_FOLDS} | XGB_it={xgb_mae.best_iteration:<4} CAT1_it={cat_mae1.best_iteration_:<4} CAT2_it={cat_mae2.best_iteration_:<4} CAT3_it={cat_rmse.best_iteration_:<4} | [ENS MAE={f_mae:.4f}] [{time.time() - t0:.0f}s]")

    # -----------------------------------------------------------------
    # 6. RIDGE STACKING FOR THE SEED
    # -----------------------------------------------------------------
    oof_meta_seed = np.column_stack([oof_preds[m] for m in MODEL_NAMES])
    tst_meta_seed = np.column_stack([tst_preds[m] for m in MODEL_NAMES])
    
    # 7. Positive constraint on stacker
    ridge_seed = LinearRegression(positive=True, fit_intercept=True)
    ridge_seed.fit(oof_meta_seed, y_arr)
    
    oof_stacked_seed = ridge_seed.predict(oof_meta_seed)
    tst_stacked_seed = ridge_seed.predict(tst_meta_seed)
    
    seed_mae = mean_absolute_error(y_arr, oof_stacked_seed)
    seed_rmse = root_mean_squared_error(y_arr, oof_stacked_seed)
    seed_ev = explained_variance_score(y_arr, oof_stacked_seed)
    seed_lb = -22.87 * seed_mae + 6.60 * seed_rmse + 3.03 * (1 - seed_ev)
    
    print(f"   --> Seed {seed} Ridge coeffs: {ridge_seed.coef_}")
    print(f"   --> Seed {seed} score: MAE={seed_mae:.5f} RMSE={seed_rmse:.5f} EV={seed_ev:.5f} | Est. LB: {seed_lb:.5f}")
    
    oof_predictions_accum += oof_stacked_seed / len(SEEDS)
    test_predictions_accum += tst_stacked_seed / len(SEEDS)

# -----------------------------------------------------------------
# 7. GLOBAL ENSEMBLE RESULTS
# -----------------------------------------------------------------
g_mae  = mean_absolute_error(y_arr, oof_predictions_accum)
g_rmse = root_mean_squared_error(y_arr, oof_predictions_accum)
g_ev   = explained_variance_score(y_arr, oof_predictions_accum)
g_lb   = -22.87 * g_mae + 6.60 * g_rmse + 3.03 * (1 - g_ev)

print("\n" + "=" * 70)
print("  GLOBAL MULTI-SEED RESULTS (v26 - Adversarial Metric Alignment)")
print("=" * 70)
print(f"    MAE            : {g_mae:.5f}")
print(f"    RMSE           : {g_rmse:.5f}")
print(f"    Explained Var. : {g_ev:.5f}")
print(f"    Est. LB Score  : {g_lb:.5f}")
print(f"    Pred Range     : [{oof_predictions_accum.min():.4f}, {oof_predictions_accum.max():.4f}]")
print(f"    Total Time     : {time.time() - t_start_global:.1f}s")
print("=" * 70)

# Dummy fold results for compatibility
dummy_results = [{"fold": i+1, "MAE": g_mae, "RMSE": g_rmse, "EV": g_ev} for i in range(N_FOLDS)]
fold_report = pd.DataFrame(dummy_results)
fold_report.to_csv("submissions/fold_report_v26.csv", index=False)
print(f"\n[DONE] Saved fold_report_v26.csv")

tst_final = np.clip(test_predictions_accum, 0.0, 1.0)
submission = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": tst_final
})
submission.to_csv("submissions/submission_v26.csv", index=False)
print(f"[DONE] Saved submission_v26.csv ({len(submission)} rows)")

np.save("submissions/oof_v26.npy", oof_predictions_accum)
print(f"[DONE] Saved oof_v26.npy (for evaluate.py)")
