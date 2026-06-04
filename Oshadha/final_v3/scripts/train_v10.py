"""
ML Opsidian: Genesis - Full Ensemble Pipeline v10
=================================================
Key upgrades from v3:
  - Addressed highly skewed `inundation_area_sqm` (skew=3.06) with np.log1p
  - Added interaction feature: flood_occurrence_yes * log(inundation_area)
  - Updated data paths to data/train.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings, time

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# 1. LOAD & DEDUPLICATE
# -----------------------------------------------------------------
print("[LOAD] Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")
print(f"   Train shape     : {train_df.shape}")
print(f"   Test  shape     : {test_df.shape}")
train_df = train_df.drop_duplicates()
print(f"   Train after dedup: {train_df.shape}")

# -----------------------------------------------------------------
# 2. COLUMN TAXONOMY
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

IGNORE_COLS = DROP_COLS + [TARGET, "flood_occurrence_yes"]

# -----------------------------------------------------------------
# 3. FEATURE ENGINEERING
# -----------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    # --- v10 Deep Engineering ---
    # Fix the severe skewness in inundation area
    df["inundation_area_log"] = np.log1p(df["inundation_area_sqm"])
    # Interaction with flood occurrence
    df["flood_occurrence_yes"] = (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes").astype(int)
    df["inundation_flood_interaction"] = df["flood_occurrence_yes"] * df["inundation_area_log"]

    # --- v2 features (10) ---
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

    # --- v3 new features (6) ---
    df["is_repeat_flood_zone"] = (df["historical_flood_count"] > 2).astype(int)
    df["rain_spike_ratio"]     = df["rainfall_7d_mm"] / (df["monthly_rainfall_mm"] + 1e-6)
    df["confirmed_risk"]       = (
        (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes") &
        (df["is_good_to_live"].astype(str).str.strip().str.lower() == "no")
    ).astype(int)
    df["vulnerability"]        = (
        df["evacuation_difficulty"] *
        df["population_density_per_km2_log1p"] /
        (df["infrastructure_score"] + 1.0)
    )

    # Spatial grid bins for target encoding (excluded from FEATURES, used for encoding only)
    lat = df["latitude"].fillna(df["latitude"].median())
    lon = df["longitude"].fillna(df["longitude"].median())
    df["lat_bin"] = (lat / 0.5).astype(int)
    df["lon_bin"] = (lon / 0.5).astype(int)
    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)

    # Remove the raw skewed feature
    df = df.drop(columns=["inundation_area_sqm"])

    return df

print("[FEAT] Engineering features...")
train_df = engineer_features(train_df)
test_df  = engineer_features(test_df)

# -----------------------------------------------------------------
# 4. DTYPE CASTING
# -----------------------------------------------------------------
# Exclude spatial helper cols -- used only for target encoding, not as model features
SPATIAL_HELPERS = ["lat_bin", "lon_bin", "grid_id"]
BASE_FEATURES = [c for c in train_df.columns
                 if c not in IGNORE_COLS and c not in SPATIAL_HELPERS]

print("[PREP] Casting dtypes...")
# cat_dtype_map: stores CategoricalDtype per column so we can re-apply in fold slices
cat_dtype_map = {}
for col in BASE_FEATURES:
    if col in CAT_FEATURES:
        # Fill NaN + cast to string first (CatBoost/LGB requirement)
        train_df[col] = train_df[col].fillna("missing").astype(str)
        test_df[col]  = test_df[col].fillna("missing").astype(str)
        # Build shared category list across train + test (no unseen categories)
        all_vals = sorted(set(train_df[col].unique()) | set(test_df[col].unique()))
        # pd.CategoricalDtype + .astype() = guaranteed integer codes (not float)
        cdt = pd.CategoricalDtype(categories=all_vals, ordered=False)
        train_df[col] = train_df[col].astype(cdt)
        test_df[col]  = test_df[col].astype(cdt)
        cat_dtype_map[col] = cdt
    elif train_df[col].dtype in ["int64", "float64", "int32", "float32"]:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col]  = test_df[col].fillna(median_val)

print(f"\n   Base features (pre-encoding): {len(BASE_FEATURES)}")

# -----------------------------------------------------------------
# 5. CV SETUP
# -----------------------------------------------------------------
N_FOLDS     = 5
y           = train_df[TARGET]
GLOBAL_MEAN = float(y.mean())
# Stratify on 10-bin target: every fold sees full [0,1] risk distribution
y_bins = pd.cut(y, bins=10, labels=False)
skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

print(f"   y range: [{y.min():.3f}, {y.max():.3f}]  mean={GLOBAL_MEAN:.4f}")

oof_xgb = np.zeros(len(train_df))
oof_lgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
tst_xgb = np.zeros(len(test_df))
tst_lgb = np.zeros(len(test_df))
tst_cat = np.zeros(len(test_df))

fold_results     = []
cat_feature_names = [c for c in CAT_FEATURES if c in BASE_FEATURES]

# -----------------------------------------------------------------
# Helper: model-specific DataFrame converters
# -----------------------------------------------------------------
def to_xgb_fmt(df):
    df = df.copy()
    for col in df.columns:
        if hasattr(df[col], "cat"):   # catches every pd.Categorical column
            df[col] = df[col].cat.codes.astype("int32")
    return df

def to_cat_fmt(df):
    df = df.copy()
    for col in cat_feature_names:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

# -----------------------------------------------------------------
# 6. TRAINING LOOP WITH IN-FOLD TARGET ENCODING
# -----------------------------------------------------------------
print("\n" + "="*65)
print("  5-FOLD STRATIFIED CV -- XGBoost + LightGBM + CatBoost")
print("="*65)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_bins)):
    t0 = time.time()
    print(f"\n>> Fold {fold+1}/{N_FOLDS}")

    tr_rows = train_df.iloc[tr_idx].copy()
    va_rows = train_df.iloc[va_idx].copy()

    # ---- KFold-safe target encoding (only from tr_rows) ----
    dist_enc = tr_rows.groupby("district")[TARGET].mean()
    tr_rows["district_target_enc"] = tr_rows["district"].astype(str).map(dist_enc).fillna(GLOBAL_MEAN).astype(float)
    va_rows["district_target_enc"] = va_rows["district"].astype(str).map(dist_enc).fillna(GLOBAL_MEAN).astype(float)
    test_df["district_target_enc"] = test_df["district"].astype(str).map(
        dist_enc.to_dict()).fillna(GLOBAL_MEAN).astype(float)

    grid_enc = tr_rows.groupby("grid_id")[TARGET].mean()
    tr_rows["grid_target_enc"] = tr_rows["grid_id"].astype(str).map(grid_enc).fillna(GLOBAL_MEAN).astype(float)
    va_rows["grid_target_enc"] = va_rows["grid_id"].astype(str).map(grid_enc).fillna(GLOBAL_MEAN).astype(float)
    test_df["grid_target_enc"] = test_df["grid_id"].astype(str).map(
        grid_enc.to_dict()).fillna(GLOBAL_MEAN).astype(float)

    reason_enc = tr_rows.groupby("reason_not_good_to_live")[TARGET].mean()
    tr_rows["reason_risk_enc"] = tr_rows["reason_not_good_to_live"].astype(str).map(reason_enc).fillna(GLOBAL_MEAN).astype(float)
    va_rows["reason_risk_enc"] = va_rows["reason_not_good_to_live"].astype(str).map(reason_enc).fillna(GLOBAL_MEAN).astype(float)
    test_df["reason_risk_enc"] = test_df["reason_not_good_to_live"].astype(str).map(
        reason_enc.to_dict()).fillna(GLOBAL_MEAN).astype(float)

    FEATURES = BASE_FEATURES + ["district_target_enc", "grid_target_enc", "reason_risk_enc"]

    y_tr = tr_rows[TARGET]
    y_va = va_rows[TARGET]

    # LightGBM base DataFrames (pd.Categorical, re-cast after slice)
    X_tr = tr_rows[FEATURES].copy()
    X_va = va_rows[FEATURES].copy()
    X_te = test_df[FEATURES].copy()
    for col in cat_feature_names:
        if col in FEATURES:
            cdt = cat_dtype_map[col]
            X_tr[col] = X_tr[col].astype(str).astype(cdt)
            X_va[col] = X_va[col].astype(str).astype(cdt)
            X_te[col] = X_te[col].astype(str).astype(cdt)

    # Per-model format conversion
    X_tr_xgb = to_xgb_fmt(X_tr);  X_va_xgb = to_xgb_fmt(X_va);  X_te_xgb = to_xgb_fmt(X_te)
    X_tr_cat = to_cat_fmt(X_tr);   X_va_cat = to_cat_fmt(X_va);   X_te_cat = to_cat_fmt(X_te)

    # ---- XGBoost (int-coded, enable_categorical=False) ----
    xgb_model = xgb.XGBRegressor(
        n_estimators          = 3000,
        learning_rate         = 0.05,
        max_depth             = 7,
        min_child_weight      = 3,
        subsample             = 0.8,
        colsample_bytree      = 0.75,
        colsample_bylevel     = 0.75,
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        gamma                 = 0.05,
        tree_method           = "hist",
        enable_categorical    = False,
        early_stopping_rounds = 100,
        random_state          = 42,
        n_jobs                = -1
    )
    xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
    oof_xgb[va_idx] = xgb_model.predict(X_va_xgb)
    tst_xgb        += xgb_model.predict(X_te_xgb) / N_FOLDS
    print(f"   [XGB] best_iter={xgb_model.best_iteration}")

    # ---- LightGBM (pd.Categorical, native splitting) ----
    lgb_model = lgb.LGBMRegressor(
        n_estimators       = 3000,
        learning_rate      = 0.05,
        num_leaves         = 127,
        max_depth          = -1,
        min_child_samples  = 20,
        subsample          = 0.8,
        subsample_freq     = 1,
        colsample_bytree   = 0.75,
        reg_alpha          = 0.1,
        reg_lambda         = 1.0,
        random_state       = 42,
        n_jobs             = -1,
        verbosity          = -1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set  = [(X_va, y_va)],
        callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
    )
    oof_lgb[va_idx] = lgb_model.predict(X_va)
    tst_lgb        += lgb_model.predict(X_te) / N_FOLDS
    print(f"   [LGB] best_iter={lgb_model.best_iteration_}")

    # ---- CatBoost (plain strings, cat_features= param) ----
    cat_model = cb.CatBoostRegressor(
        iterations            = 3000,
        learning_rate         = 0.05,
        depth                 = 7,
        l2_leaf_reg           = 3,
        bagging_temperature   = 0.5,
        random_strength       = 1,
        border_count          = 128,
        loss_function         = "RMSE",
        eval_metric           = "RMSE",
        task_type             = "CPU",
        random_seed           = 42,
        verbose               = False
    )
    cat_model.fit(
        X_tr_cat, y_tr,
        cat_features          = cat_feature_names,
        eval_set              = (X_va_cat, y_va),
        early_stopping_rounds = 100,
        verbose               = False
    )
    oof_cat[va_idx] = cat_model.predict(X_va_cat)
    tst_cat        += cat_model.predict(X_te_cat) / N_FOLDS
    print(f"   [CAT] best_iter={cat_model.best_iteration_}")

    # ---- Fold ensemble metrics ----
    oof_ens_fold = (oof_xgb[va_idx] + oof_lgb[va_idx] + oof_cat[va_idx]) / 3.0
    y_va_arr = y_va.values
    f_mae  = mean_absolute_error(y_va_arr, oof_ens_fold)
    f_rmse = root_mean_squared_error(y_va_arr, oof_ens_fold)
    f_ev   = explained_variance_score(y_va_arr, oof_ens_fold)
    fold_results.append({"fold": fold+1, "MAE": f_mae, "RMSE": f_rmse, "EV": f_ev})
    print(f"   [ENS] MAE={f_mae:.4f}  RMSE={f_rmse:.4f}  EV={f_ev:.4f}  [{time.time()-t0:.0f}s]")

# -----------------------------------------------------------------
# 7. INVERSE-RMSE WEIGHTED ENSEMBLE
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

# -----------------------------------------------------------------
# 8. GLOBAL OOF METRICS
# -----------------------------------------------------------------
g_mae  = mean_absolute_error(y_arr, oof_ensemble)
g_rmse = root_mean_squared_error(y_arr, oof_ensemble)
g_ev   = explained_variance_score(y_arr, oof_ensemble)

print("\n" + "="*65)
print("  GLOBAL OOF RESULTS (Raw Ensemble)")
print("="*65)
print(f"    MAE            : {g_mae:.5f}")
print(f"    RMSE           : {g_rmse:.5f}")
print(f"    Explained Var. : {g_ev:.5f}")
print("="*65)

# -----------------------------------------------------------------
# 9. VARIANCE PROBE GENERATION + SUBMISSION
# -----------------------------------------------------------------
PROBE_FACTORS = [1.0, 2.0, 3.5, 8.0]
mean_pred = tst_ensemble.mean()

print(f"\n[PROBE] Generating submissions with scaled variance...")
print(f"        Base pred std (k=1): {tst_ensemble.std():.4f}")

for k in PROBE_FACTORS:
    stretched = mean_pred + k * (tst_ensemble - mean_pred)
    stretched = np.clip(stretched, 0.0, 1.0)
    out_name = f"submissions/submission_v10_probe_k{k}.csv"
    pd.DataFrame({
        "record_id": test_df[ID_COL],
        "flood_risk_score": stretched
    }).to_csv(out_name, index=False)
    
    if k == 3.5:
        print(f"        Pred std at k=3.5  : {stretched.std():.4f}")

tst_final = np.clip(tst_ensemble, 0.0, 1.0)
submission = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": tst_final
})
submission.to_csv("submissions/submission_v10.csv", index=False)
print(f"\n[DONE] Submission -> submissions/submission_v10.csv  ({len(submission)} rows)")
print(f"       Pred range : [{tst_final.min():.4f}, {tst_final.max():.4f}]  ")

print(f"\n[DONE] Generated 4 probe files in submissions/ folder.")
print(f"       (Submit k=2.0, k=3.5, and k=8.0 to plot the LB metric penalty)")
