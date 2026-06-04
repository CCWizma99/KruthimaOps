"""
ML Opsidian: Genesis - Pure MAE Ensemble v18
==============================================
v17 proved definitively:
  - MAE-Only Average (3 models) beat Ridge Stacked (6 models)
  - Ridge assigned NEGATIVE weights to RMSE (-0.28) and Huber (-0.08)
  - RMSE loss is the WRONG objective for this competition metric

v18: Drop everything except MAE-loss CatBoost. Run 5 diverse MAE 
CatBoosts across depths 4/5/6/7, learning rates 0.03/0.04/0.05, 
5 different seeds. Simple average all 5.

MODEL LINEUP (all CatBoost, all MAE loss):
  1. CAT-MAE-A  (d4, lr=0.03, s42)  — shallowest, most regularized
  2. CAT-MAE-B  (d5, lr=0.03, s123) — proven best config from v17
  3. CAT-MAE-C  (d5, lr=0.05, s789) — faster LR, v16 star player
  4. CAT-MAE-D  (d6, lr=0.04, s456) — intermediate depth
  5. CAT-MAE-E  (d7, lr=0.05, s321) — deepest, captures complex interactions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
import catboost as cb
import warnings
import time

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# 1. LOAD & DEDUPLICATE
# -----------------------------------------------------------------
print("=" * 70)
print("  ML OPSIDIAN v18 - PURE MAE ENSEMBLE")
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

cat_feature_names = [c for c in CAT_FEATURES if c in BASE_FEATURES]
print(f"   Base features: {len(BASE_FEATURES)}")

def to_cat_fmt(df):
    df = df.copy()
    for col in cat_feature_names:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

# -----------------------------------------------------------------
# 5. MODEL CONFIGS — 5 MAE CatBoosts
# -----------------------------------------------------------------
MODEL_CONFIGS = [
    {"name": "CAT-MAE-A (d4)", "depth": 4, "lr": 0.03, "l2": 7, "seed": 42,  "iters": 5000},
    {"name": "CAT-MAE-B (d5)", "depth": 5, "lr": 0.03, "l2": 5, "seed": 123, "iters": 5000},
    {"name": "CAT-MAE-C (d5)", "depth": 5, "lr": 0.05, "l2": 5, "seed": 789, "iters": 5000},
    {"name": "CAT-MAE-D (d6)", "depth": 6, "lr": 0.04, "l2": 3, "seed": 456, "iters": 4000},
    {"name": "CAT-MAE-E (d7)", "depth": 7, "lr": 0.05, "l2": 3, "seed": 321, "iters": 3000},
]
MODEL_NAMES = [c["name"] for c in MODEL_CONFIGS]

# -----------------------------------------------------------------
# 6. CV LOOP
# -----------------------------------------------------------------
N_FOLDS     = 5
y           = train_df[TARGET]
GLOBAL_MEAN = float(y.mean())
GLOBAL_STD  = float(y.std())
y_bins      = pd.cut(y, bins=10, labels=False)
skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
SMOOTHING   = 10

oof_preds = {m: np.zeros(len(train_df)) for m in MODEL_NAMES}
tst_preds = {m: np.zeros(len(test_df))  for m in MODEL_NAMES}
fold_results = []

print("\n" + "=" * 70)
print("  5-FOLD CV -- 5x CatBoost MAE (Pure MAE Ensemble) v18")
print("  No RMSE. No Huber. No XGBoost. Pure MAE dominance.")
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
    X_tr = to_cat_fmt(tr_rows[FEATURES].copy())
    X_va = to_cat_fmt(va_rows[FEATURES].copy())
    X_te = to_cat_fmt(test_df[FEATURES].copy())

    cat_pool_tr = cb.Pool(X_tr, y_tr, cat_features=cat_feature_names)
    cat_pool_va = cb.Pool(X_va, y_va, cat_features=cat_feature_names)

    # Train all 5 MAE CatBoost models
    for cfg in MODEL_CONFIGS:
        model = cb.CatBoostRegressor(
            iterations=cfg["iters"],
            learning_rate=cfg["lr"],
            depth=cfg["depth"],
            l2_leaf_reg=cfg["l2"],
            loss_function="MAE",
            eval_metric="MAE",
            random_seed=cfg["seed"],
            verbose=False
        )
        model.fit(cat_pool_tr, eval_set=cat_pool_va, early_stopping_rounds=100, verbose=False)
        
        oof_preds[cfg["name"]][va_idx] = model.predict(X_va)
        tst_preds[cfg["name"]] += model.predict(X_te) / N_FOLDS
        
        print(f"   [{cfg['name']}]  iter={model.best_iteration_}")

    # Fold summary (simple average)
    oof_avg_fold = np.mean([oof_preds[m][va_idx] for m in MODEL_NAMES], axis=0)
    y_va_arr = y_va.values
    f_mae  = mean_absolute_error(y_va_arr, oof_avg_fold)
    f_rmse = root_mean_squared_error(y_va_arr, oof_avg_fold)
    f_ev   = explained_variance_score(y_va_arr, oof_avg_fold)
    fold_results.append({"fold": fold+1, "MAE": f_mae, "RMSE": f_rmse, "EV": f_ev})
    print(f"   [ENS-5 AVG]  MAE={f_mae:.4f}  RMSE={f_rmse:.4f}  EV={f_ev:.4f}  [{time.time()-t0:.0f}s]")

# -----------------------------------------------------------------
# 7. BLENDING COMPARISON
# -----------------------------------------------------------------
print("\n" + "-" * 70)
print("  BLENDING: Simple Average vs Ridge vs Subsets")
print("-" * 70)

y_arr = y.values

# Individual model performance
print(f"\n   [INDIVIDUAL MODEL PERFORMANCE]")
for name in MODEL_NAMES:
    m_rmse = root_mean_squared_error(y_arr, oof_preds[name])
    m_mae  = mean_absolute_error(y_arr, oof_preds[name])
    m_ev   = explained_variance_score(y_arr, oof_preds[name])
    print(f"      {name:<20}: RMSE={m_rmse:.5f}  MAE={m_mae:.5f}  EV={m_ev:.5f}")

# Ridge stacking (still worth checking)
oof_meta = np.column_stack([oof_preds[m] for m in MODEL_NAMES])
tst_meta = np.column_stack([tst_preds[m] for m in MODEL_NAMES])

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

# All blend methods
print(f"\n   [BLEND COMPARISON]")
methods = {}

# Simple average (all 5)
oof_avg = np.mean([oof_preds[m] for m in MODEL_NAMES], axis=0)
tst_avg = np.mean([tst_preds[m] for m in MODEL_NAMES], axis=0)
methods["Simple Avg (5)"] = (oof_avg, tst_avg)

# Ridge stacked
methods["Ridge Stacked"] = (oof_stacked, tst_stacked)
methods["Ridge Stacked CV"] = (oof_stacked, tst_stacked_accum)

# Top-3 average (best 3 individual models by EV)
evs = {m: explained_variance_score(y_arr, oof_preds[m]) for m in MODEL_NAMES}
top3 = sorted(evs, key=evs.get, reverse=True)[:3]
oof_top3 = np.mean([oof_preds[m] for m in top3], axis=0)
tst_top3 = np.mean([tst_preds[m] for m in top3], axis=0)
methods[f"Top-3 Avg"] = (oof_top3, tst_top3)

# Top-4 average
top4 = sorted(evs, key=evs.get, reverse=True)[:4]
oof_top4 = np.mean([oof_preds[m] for m in top4], axis=0)
tst_top4 = np.mean([tst_preds[m] for m in top4], axis=0)
methods[f"Top-4 Avg"] = (oof_top4, tst_top4)

# Inverse-RMSE weighted
rmses = {m: root_mean_squared_error(y_arr, oof_preds[m]) for m in MODEL_NAMES}
weights = {m: 1.0/rmses[m] for m in MODEL_NAMES}
total_w = sum(weights.values())
oof_invw = sum(weights[m] * oof_preds[m] for m in MODEL_NAMES) / total_w
tst_invw = sum(weights[m] * tst_preds[m] for m in MODEL_NAMES) / total_w
methods["Inverse-RMSE (5)"] = (oof_invw, tst_invw)

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
    print(f"      {name:>20}: MAE={m_mae:.5f}  RMSE={m_rmse:.5f}  EV={m_ev:.5f}{winner}")

print(f"\n   [SELECTED] {best_method}")
if "Top-3" in best_method:
    print(f"   Top-3 models: {top3}")
elif "Top-4" in best_method:
    print(f"   Top-4 models: {top4}")

# -----------------------------------------------------------------
# 8. FINAL RESULTS
# -----------------------------------------------------------------
g_mae  = mean_absolute_error(y_arr, best_oof)
g_rmse = root_mean_squared_error(y_arr, best_oof)
g_ev   = explained_variance_score(y_arr, best_oof)

print("\n" + "=" * 70)
print("  GLOBAL OOF RESULTS (v18 - Pure MAE Ensemble)")
print("=" * 70)
print(f"    MAE            : {g_mae:.5f}")
print(f"    RMSE           : {g_rmse:.5f}")
print(f"    Explained Var. : {g_ev:.5f}")
print(f"    Pred Range     : [{best_oof.min():.4f}, {best_oof.max():.4f}]")
print("=" * 70)

# Fold report
fold_report = pd.DataFrame(fold_results)
fold_report.to_csv("submissions/fold_report_v18.csv", index=False)
print(f"\n[DONE] Saved fold_report_v18.csv")
print(fold_report.to_string(index=False))

# Comparison
print(f"\n[COMPARE] v11 Baseline : MAE=0.17984, RMSE=0.23539, EV=0.02737  (LB: 0.38637)")
print(f"[COMPARE] v13          : MAE=0.17937, RMSE=0.23500, EV=0.03060  (LB: 0.38476)")
print(f"[COMPARE] v16          : MAE=0.17904, RMSE=0.23476, EV=0.03258")
print(f"[COMPARE] v17          : MAE=0.17882, RMSE=0.23465, EV=0.03390")
print(f"[COMPARE] v18 Current  : MAE={g_mae:.5f}, RMSE={g_rmse:.5f}, EV={g_ev:.5f}")
ev_delta = g_ev - 0.03390
rmse_delta = g_rmse - 0.23465
print(f"[COMPARE] v18 vs v17 EV   : {ev_delta:+.5f} ({'IMPROVED' if ev_delta > 0 else 'REGRESSED'})")
print(f"[COMPARE] v18 vs v17 RMSE : {rmse_delta:+.5f} ({'IMPROVED' if rmse_delta < 0 else 'REGRESSED'})")

# Submission
tst_final = np.clip(best_tst, 0.0, 1.0)
submission = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": tst_final
})
submission.to_csv("submissions/submission_v18.csv", index=False)
print(f"\n[DONE] Saved submission_v18.csv ({len(submission)} rows)")
print(f"       Pred range : [{tst_final.min():.4f}, {tst_final.max():.4f}]")
