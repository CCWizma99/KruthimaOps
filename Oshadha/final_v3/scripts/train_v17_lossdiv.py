"""
ML Opsidian: Genesis - Loss Diversity Maximizer v17
====================================================
v16 proved MAE-loss CatBoost is the best individual model on ALL metrics
(RMSE=0.23472, MAE=0.17889, EV=0.03335) with 75% Ridge weight.

v17 doubles down: 3x MAE-loss CatBoosts (seed/depth diversity) +
Huber loss (intermediate MAE/RMSE) + RMSE anchor + XGBoost.
6 models → Ridge meta-learner.

MODEL LINEUP:
  1. XGB        (d7, RMSE)           — algorithmic diversity
  2. CAT-MAE-1  (d5, lr=0.05, MAE)  — star player (v16 winner)
  3. CAT-MAE-2  (d5, lr=0.03, MAE)  — slower LR, seed diversity
  4. CAT-MAE-3  (d7, lr=0.05, MAE)  — depth diversity
  5. CAT-Huber  (d5, Huber:δ=0.5)   — intermediate loss landscape
  6. CAT-RMSE   (d5, lr=0.03, RMSE) — loss anchor for Ridge balance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import catboost as cb
import warnings
import time

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# 1. LOAD & DEDUPLICATE
# -----------------------------------------------------------------
print("=" * 70)
print("  ML OPSIDIAN v17 - LOSS DIVERSITY MAXIMIZER")
print("=" * 70)
print("\n[LOAD] Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")
train_df = train_df.drop_duplicates()
print(f"   Train: {train_df.shape}  Test: {test_df.shape}")

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
# 3. FEATURE ENGINEERING (v14 proven set)
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
    "is_good_to_live", "reason_not_good_to_live",
    "downstream_sig", "infra_deficit_sig"
]
TARGET_ENC_COLS = [
    "district", "grid_id", "downstream_sig", "infra_deficit_sig",
    "landcover", "soil_type", "water_supply", "electricity", "road_quality"
]
IGNORE_COLS = DROP_COLS + [TARGET, "flood_occurrence_yes"]
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

print(f"   Base features: {len(BASE_FEATURES)}")

# -----------------------------------------------------------------
# 5. CV LOOP WITH 6 MODELS
# -----------------------------------------------------------------
N_FOLDS     = 5
y           = train_df[TARGET]
GLOBAL_MEAN = float(y.mean())
GLOBAL_STD  = float(y.std())
y_bins      = pd.cut(y, bins=10, labels=False)
skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
SMOOTHING   = 10

MODEL_NAMES = [
    "XGB (RMSE)",
    "CAT-MAE-1 (d5)",
    "CAT-MAE-2 (d5)",
    "CAT-MAE-3 (d7)",
    "CAT-Huber (d5)",
    "CAT-RMSE (d5)"
]
oof_preds = {m: np.zeros(len(train_df)) for m in MODEL_NAMES}
tst_preds = {m: np.zeros(len(test_df))  for m in MODEL_NAMES}
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

print("\n" + "=" * 70)
print("  5-FOLD CV -- 6 Models: 3x MAE + Huber + RMSE + XGB (v17)")
print("  Loss Diversity Maximizer")
print("=" * 70)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_bins)):
    t0 = time.time()
    print(f"\n>> Fold {fold+1}/{N_FOLDS}")

    tr_rows = train_df.iloc[tr_idx].copy()
    va_rows = train_df.iloc[va_idx].copy()

    # Target encodings (mean + std + count, Bayesian smoothed)
    for col in TARGET_ENC_COLS:
        group_stats = tr_rows.groupby(col)[TARGET].agg(['mean', 'std', 'count'])
        group_stats['std'] = group_stats['std'].fillna(0.0)
        smoothed_mean = (group_stats['count'] * group_stats['mean'] + SMOOTHING * GLOBAL_MEAN) / (group_stats['count'] + SMOOTHING)
        smoothed_std  = (group_stats['count'] * group_stats['std'] + SMOOTHING * GLOBAL_STD) / (group_stats['count'] + SMOOTHING)
        log_count = np.log1p(group_stats['count'])
        for tgt_df in [tr_rows, va_rows, test_df]:
            tgt_df[f"{col}_target_enc"] = tgt_df[col].astype(str).map(smoothed_mean).fillna(GLOBAL_MEAN).astype(float)
            tgt_df[f"{col}_target_std"] = tgt_df[col].astype(str).map(smoothed_std).fillna(GLOBAL_STD).astype(float)
            tgt_df[f"{col}_target_cnt"] = tgt_df[col].astype(str).map(log_count).fillna(0.0).astype(float)

    te_features = []
    for col in TARGET_ENC_COLS:
        te_features.extend([f"{col}_target_enc", f"{col}_target_std", f"{col}_target_cnt"])
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

    # === 1. XGBoost (RMSE) ===
    xgb_m = xgb.XGBRegressor(
        n_estimators=3000, learning_rate=0.05, max_depth=7,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.75,
        tree_method="hist", early_stopping_rounds=100, random_state=42, n_jobs=-1
    )
    xgb_m.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
    oof_preds["XGB (RMSE)"][va_idx] = xgb_m.predict(X_va_xgb)
    tst_preds["XGB (RMSE)"] += xgb_m.predict(X_te_xgb) / N_FOLDS
    print(f"   [XGB-RMSE]    iter={xgb_m.best_iteration}")

    # === 2. CatBoost-MAE-1 (d5, lr=0.05, seed=789) — star player ===
    cat_mae1 = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.05, depth=5,
        l2_leaf_reg=5, loss_function="MAE", eval_metric="MAE",
        random_seed=789, verbose=False
    )
    cat_mae1.fit(cat_pool_tr, eval_set=cat_pool_va, early_stopping_rounds=100, verbose=False)
    oof_preds["CAT-MAE-1 (d5)"][va_idx] = cat_mae1.predict(X_va_cat)
    tst_preds["CAT-MAE-1 (d5)"] += cat_mae1.predict(X_te_cat) / N_FOLDS
    print(f"   [CAT-MAE-1]   iter={cat_mae1.best_iteration_}  (d5, lr=0.05, s789)")

    # === 3. CatBoost-MAE-2 (d5, lr=0.03, seed=42) — slower LR ===
    cat_mae2 = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.03, depth=5,
        l2_leaf_reg=5, loss_function="MAE", eval_metric="MAE",
        random_seed=42, verbose=False
    )
    cat_mae2.fit(cat_pool_tr, eval_set=cat_pool_va, early_stopping_rounds=100, verbose=False)
    oof_preds["CAT-MAE-2 (d5)"][va_idx] = cat_mae2.predict(X_va_cat)
    tst_preds["CAT-MAE-2 (d5)"] += cat_mae2.predict(X_te_cat) / N_FOLDS
    print(f"   [CAT-MAE-2]   iter={cat_mae2.best_iteration_}  (d5, lr=0.03, s42)")

    # === 4. CatBoost-MAE-3 (d7, lr=0.05, seed=456) — deeper ===
    cat_mae3 = cb.CatBoostRegressor(
        iterations=3000, learning_rate=0.05, depth=7,
        l2_leaf_reg=3, loss_function="MAE", eval_metric="MAE",
        random_seed=456, verbose=False
    )
    cat_mae3.fit(cat_pool_tr, eval_set=cat_pool_va, early_stopping_rounds=100, verbose=False)
    oof_preds["CAT-MAE-3 (d7)"][va_idx] = cat_mae3.predict(X_va_cat)
    tst_preds["CAT-MAE-3 (d7)"] += cat_mae3.predict(X_te_cat) / N_FOLDS
    print(f"   [CAT-MAE-3]   iter={cat_mae3.best_iteration_}  (d7, lr=0.05, s456)")

    # === 5. CatBoost-Huber (d5, Huber:delta=0.5) — intermediate loss ===
    cat_hub = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.05, depth=5,
        l2_leaf_reg=5, loss_function="Huber:delta=0.5", eval_metric="MAE",
        random_seed=123, verbose=False
    )
    cat_hub.fit(cat_pool_tr, eval_set=cat_pool_va, early_stopping_rounds=100, verbose=False)
    oof_preds["CAT-Huber (d5)"][va_idx] = cat_hub.predict(X_va_cat)
    tst_preds["CAT-Huber (d5)"] += cat_hub.predict(X_te_cat) / N_FOLDS
    print(f"   [CAT-Huber]   iter={cat_hub.best_iteration_}  (d5, Huber:d=0.5)")

    # === 6. CatBoost-RMSE (d5, lr=0.03) — loss anchor ===
    cat_rmse = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.03, depth=5,
        l2_leaf_reg=5, loss_function="RMSE", eval_metric="RMSE",
        random_seed=321, verbose=False
    )
    cat_rmse.fit(cat_pool_tr, eval_set=cat_pool_va, early_stopping_rounds=100, verbose=False)
    oof_preds["CAT-RMSE (d5)"][va_idx] = cat_rmse.predict(X_va_cat)
    tst_preds["CAT-RMSE (d5)"] += cat_rmse.predict(X_te_cat) / N_FOLDS
    print(f"   [CAT-RMSE]    iter={cat_rmse.best_iteration_}  (d5, lr=0.03, s321)")

    # Fold summary (simple average of all 6)
    oof_avg_fold = np.mean([oof_preds[m][va_idx] for m in MODEL_NAMES], axis=0)
    y_va_arr = y_va.values
    f_mae  = mean_absolute_error(y_va_arr, oof_avg_fold)
    f_rmse = root_mean_squared_error(y_va_arr, oof_avg_fold)
    f_ev   = explained_variance_score(y_va_arr, oof_avg_fold)
    fold_results.append({"fold": fold+1, "MAE": f_mae, "RMSE": f_rmse, "EV": f_ev})
    print(f"   [ENS-6]       MAE={f_mae:.4f}  RMSE={f_rmse:.4f}  EV={f_ev:.4f}  [{time.time()-t0:.0f}s]")

# -----------------------------------------------------------------
# 6. RIDGE STACKING ON 6 MODELS
# -----------------------------------------------------------------
print("\n" + "-" * 70)
print("  LEVEL-2 STACKING: Ridge Meta-Learner on 6 Models")
print("-" * 70)

y_arr = y.values
oof_meta = np.column_stack([oof_preds[m] for m in MODEL_NAMES])
tst_meta = np.column_stack([tst_preds[m] for m in MODEL_NAMES])

print(f"\n   [INDIVIDUAL MODEL PERFORMANCE]")
for name in MODEL_NAMES:
    m_rmse = root_mean_squared_error(y_arr, oof_preds[name])
    m_mae  = mean_absolute_error(y_arr, oof_preds[name])
    m_ev   = explained_variance_score(y_arr, oof_preds[name])
    print(f"      {name:<20}: RMSE={m_rmse:.5f}  MAE={m_mae:.5f}  EV={m_ev:.5f}")

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
    print(f"      {name:<20}: {ridge_final.coef_[i]:.4f}")
print(f"      {'Intercept':<20}: {ridge_final.intercept_:.4f}")

# Blend comparison
print(f"\n   [BLEND COMPARISON]")
methods = {}

oof_avg = np.mean([oof_preds[m] for m in MODEL_NAMES], axis=0)
tst_avg = np.mean([tst_preds[m] for m in MODEL_NAMES], axis=0)
methods["Simple Average (6)"] = (oof_avg, tst_avg)

rmses = {m: root_mean_squared_error(y_arr, oof_preds[m]) for m in MODEL_NAMES}
weights = {m: 1.0/rmses[m] for m in MODEL_NAMES}
total_w = sum(weights.values())
oof_invw = sum(weights[m] * oof_preds[m] for m in MODEL_NAMES) / total_w
tst_invw = sum(weights[m] * tst_preds[m] for m in MODEL_NAMES) / total_w
methods["Inverse-RMSE (6)"] = (oof_invw, tst_invw)

methods["Ridge Stacked"] = (oof_stacked, tst_stacked)
methods["Ridge Stacked CV"] = (oof_stacked, tst_stacked_accum)

# MAE-only blend (3 MAE models)
mae_names = [m for m in MODEL_NAMES if "MAE" in m]
oof_mae_only = np.mean([oof_preds[m] for m in mae_names], axis=0)
tst_mae_only = np.mean([tst_preds[m] for m in mae_names], axis=0)
methods["MAE-Only Avg (3)"] = (oof_mae_only, tst_mae_only)

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
    print(f"      {name:>22}: MAE={m_mae:.5f}  RMSE={m_rmse:.5f}  EV={m_ev:.5f}{winner}")

print(f"\n   [SELECTED] {best_method}")

# -----------------------------------------------------------------
# 7. FINAL RESULTS & EXPORT
# -----------------------------------------------------------------
g_mae  = mean_absolute_error(y_arr, best_oof)
g_rmse = root_mean_squared_error(y_arr, best_oof)
g_ev   = explained_variance_score(y_arr, best_oof)

print("\n" + "=" * 70)
print("  GLOBAL OOF RESULTS (v17 - Loss Diversity Maximizer)")
print("=" * 70)
print(f"    MAE            : {g_mae:.5f}")
print(f"    RMSE           : {g_rmse:.5f}")
print(f"    Explained Var. : {g_ev:.5f}")
print(f"    Pred Range     : [{best_oof.min():.4f}, {best_oof.max():.4f}]")
print("=" * 70)

# Feature importance (last XGB fold)
print(f"\n[FEAT IMPORTANCE] Top 20 (XGBoost, last fold):")
imp = pd.Series(xgb_m.feature_importances_, index=FEATURES)
for rank, (feat, score) in enumerate(imp.sort_values(ascending=False).head(20).items(), 1):
    print(f"   {rank:>2}. {feat:<45} {score:.4f}")

# Fold report
fold_report = pd.DataFrame(fold_results)
fold_report.to_csv("submissions/fold_report_v17.csv", index=False)
print(f"\n[DONE] Saved fold_report_v17.csv")
print(fold_report.to_string(index=False))

# Comparison
print(f"\n[COMPARE] v11 Baseline : MAE=0.17984, RMSE=0.23539, EV=0.02737  (LB: 0.38637)")
print(f"[COMPARE] v13          : MAE=0.17937, RMSE=0.23500, EV=0.03060  (LB: 0.38476)")
print(f"[COMPARE] v14          : MAE=0.17920, RMSE=0.23497, EV=0.03085")
print(f"[COMPARE] v16          : MAE=0.17904, RMSE=0.23476, EV=0.03258")
print(f"[COMPARE] v17 Current  : MAE={g_mae:.5f}, RMSE={g_rmse:.5f}, EV={g_ev:.5f}")
ev_delta = g_ev - 0.03258
rmse_delta = g_rmse - 0.23476
print(f"[COMPARE] v17 vs v16 EV   : {ev_delta:+.5f} ({'IMPROVED' if ev_delta > 0 else 'REGRESSED'})")
print(f"[COMPARE] v17 vs v16 RMSE : {rmse_delta:+.5f} ({'IMPROVED' if rmse_delta < 0 else 'REGRESSED'})")

# Submission
tst_final = np.clip(best_tst, 0.0, 1.0)
submission = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": tst_final
})
submission.to_csv("submissions/submission_v17.csv", index=False)
print(f"\n[DONE] Saved submission_v17.csv ({len(submission)} rows)")

np.save("submissions/oof_v17.npy", best_oof)
print(f"       Pred range : [{tst_final.min():.4f}, {tst_final.max():.4f}]")
