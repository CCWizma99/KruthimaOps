"""
ML Opsidian: Genesis - Full Ensemble Pipeline v3
=================================================
Key upgrades from v2:
ML Opsidian: Genesis - High Variance Architecture (v7)
======================================================
Emergency Pivot Strategy:
  - XGBoost: Switched to DART booster with dropout to prevent early collapse.
  - ExtraTrees: Added to ensemble to inject massive variance into splits.
  - LightGBM & CatBoost: Lowered learning rates (0.005/0.01) and increased trees to 10k.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings, time

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# 1. LOAD & DEDUPLICATE
# -----------------------------------------------------------------
print("[LOAD] Loading data...")
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")
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
    "is_good_to_live", "reason_not_good_to_live", "elevation_tier"
]

IGNORE_COLS = DROP_COLS + [TARGET]

# -----------------------------------------------------------------
# 3. FEATURE ENGINEERING
# -----------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    # --- v2 features (10) ---
    df["river_rain_interaction"]  = df["distance_to_river_m_log1p"] * df["rainfall_7d_mm_log1p"]
    df["river_monthly_exposure"]  = df["distance_to_river_m_log1p"] * df["monthly_rainfall_mm_log1p"]
    df["elev_rain_risk"]          = df["elevation_m_yeojohnson"] / (df["rainfall_7d_mm_log1p"] + 1e-6)
    df["water_signal"]            = df["ndwi_qmap"].clip(lower=0)
    df["drainage_deficit"]        = (df["rainfall_7d_mm_log1p"] + 1) * (1.0 - df["drainage_index_yeojohnson"].clip(0, 1))
    df["infra_resilience"]        = df["infrastructure_score"] / (df["population_density_per_km2_log1p"] + 1e-6)
    df["evacuation_difficulty"]   = df["nearest_hospital_km_log1p"] + df["nearest_evac_km_log1p"]
    df["inundation_density_risk"] = np.log1p(df["inundation_area_sqm"]) / (df["population_density_per_km2_log1p"] + 1e-6)
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

    # --- v3.5 new features ---
    df["water_veg_combined"] = (
        df["ndwi_qmap"].clip(-3,3) / 6.0 + 0.5
    ) * (
        1.0 - (df["ndvi_qmap"].clip(-3,3) / 6.0 + 0.5)
    )
    
    df["elevation_tier"] = pd.cut(
        df["elevation_m"],
        bins=[-999, 10, 30, 100, 300, 9999],
        labels=["sea_level", "coastal", "lowland", "midland", "highland"]
    ).astype(str)

    # Spatial grid bins for target encoding (excluded from FEATURES, used for encoding only)
    lat = df["latitude"].fillna(df["latitude"].median())
    lon = df["longitude"].fillna(df["longitude"].median())
    df["lat_bin"] = (lat / 0.5).astype(int)
    df["lon_bin"] = (lon / 0.5).astype(int)
    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)

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
oof_et  = np.zeros(len(train_df))
tst_xgb = np.zeros(len(test_df))
tst_lgb = np.zeros(len(test_df))
tst_cat = np.zeros(len(test_df))
tst_et  = np.zeros(len(test_df))

fold_results     = []
cat_feature_names = [c for c in CAT_FEATURES if c in BASE_FEATURES]

# -----------------------------------------------------------------
# Helper: model-specific DataFrame converters
# -----------------------------------------------------------------
def to_xgb_fmt(df):
    """XGBoost 3.x: convert ALL categorical-dtype columns -> int codes."""
    df = df.copy()
    for col in df.columns:
        if hasattr(df[col], "cat"):   # catches every pd.Categorical column
            df[col] = df[col].cat.codes.astype("int32")
    return df

def to_cat_fmt(df):
    """CatBoost: plain str columns, passed via cat_features= param."""
    df = df.copy()
    for col in cat_feature_names:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

# -----------------------------------------------------------------
# 6. TRAINING LOOP WITH IN-FOLD TARGET ENCODING
# -----------------------------------------------------------------
print("\n" + "="*65)
print("  5-FOLD STRATIFIED CV -- XGB + LGB + CAT + ExtraTrees (DART)")
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

    # ---- Phase 2: Spatial Aggregations ----
    agg_cols = ["rainfall_7d_mm", "elevation_m", "distance_to_river_m", "inundation_area_sqm", "infrastructure_score"]
    added_agg_feats = []
    for col in agg_cols:
        d_mean = tr_rows.groupby("district")[col].mean()
        d_std  = tr_rows.groupby("district")[col].std()
        
        tr_rows[f"district_{col}_mean"] = tr_rows["district"].astype(str).map(d_mean).fillna(0).astype(float)
        va_rows[f"district_{col}_mean"] = va_rows["district"].astype(str).map(d_mean).fillna(0).astype(float)
        test_df[f"district_{col}_mean"] = test_df["district"].astype(str).map(d_mean.to_dict()).fillna(0).astype(float)
        
        tr_rows[f"district_{col}_std"] = tr_rows["district"].astype(str).map(d_std).fillna(0).astype(float)
        va_rows[f"district_{col}_std"] = va_rows["district"].astype(str).map(d_std).fillna(0).astype(float)
        test_df[f"district_{col}_std"] = test_df["district"].astype(str).map(d_std.to_dict()).fillna(0).astype(float)
        added_agg_feats.extend([f"district_{col}_mean", f"district_{col}_std"])

    # ---- Phase 2: Polynomial Interactions ----
    poly_feats = [
        "distance_to_river_m_log1p", "inundation_area_sqm", "distance_to_river_m",
        "rainfall_7d_mm_log1p", "rain_spike_ratio", "monthly_rainfall_mm_log1p",
        "rainfall_7d_mm", "water_signal"
    ]
    poly_feats_with_enc = poly_feats + ["district_target_enc", "grid_target_enc"]
    
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    
    tr_poly = poly.fit_transform(tr_rows[poly_feats_with_enc])
    va_poly = poly.transform(va_rows[poly_feats_with_enc])
    te_poly = poly.transform(test_df[poly_feats_with_enc])
    
    poly_names = poly.get_feature_names_out(poly_feats_with_enc)
    num_orig = len(poly_feats_with_enc)
    new_poly_names = poly_names[num_orig:]
    
    tr_poly_df = pd.DataFrame(tr_poly[:, num_orig:], columns=new_poly_names, index=tr_rows.index)
    va_poly_df = pd.DataFrame(va_poly[:, num_orig:], columns=new_poly_names, index=va_rows.index)
    te_poly_df = pd.DataFrame(te_poly[:, num_orig:], columns=new_poly_names, index=test_df.index)
    
    tr_rows = pd.concat([tr_rows, tr_poly_df], axis=1)
    va_rows = pd.concat([va_rows, va_poly_df], axis=1)
    test_df_fold = pd.concat([test_df, te_poly_df], axis=1)

    FEATURES = BASE_FEATURES + ["district_target_enc", "grid_target_enc"] + added_agg_feats + list(new_poly_names)

    y_tr = tr_rows[TARGET]
    y_va = va_rows[TARGET]

    X_tr = tr_rows[FEATURES].copy()
    X_va = va_rows[FEATURES].copy()
    X_te = test_df_fold[FEATURES].copy()
    for col in cat_feature_names:
        if col in FEATURES:
            cdt = cat_dtype_map[col]
            X_tr[col] = X_tr[col].astype(str).astype(cdt)
            X_va[col] = X_va[col].astype(str).astype(cdt)
            X_te[col] = X_te[col].astype(str).astype(cdt)

    # Per-model format conversion
    X_tr_xgb = to_xgb_fmt(X_tr);  X_va_xgb = to_xgb_fmt(X_va);  X_te_xgb = to_xgb_fmt(X_te)
    X_tr_cat = to_cat_fmt(X_tr);   X_va_cat = to_cat_fmt(X_va);   X_te_cat = to_cat_fmt(X_te)

    # ---- XGBoost (DART) ----
    xgb_model = xgb.XGBRegressor(
        booster               = "dart",
        rate_drop             = 0.1,
        skip_drop             = 0.5,
        n_estimators          = 2000,
        learning_rate         = 0.02,
        max_depth             = 8,
        min_child_weight      = 1,
        subsample             = 0.8,
        colsample_bytree      = 0.75,
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

    # ---- ExtraTrees (uses xgb int format) ----
    et_model = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    et_model.fit(X_tr_xgb.fillna(-999), y_tr)
    oof_et[va_idx] = et_model.predict(X_va_xgb.fillna(-999))
    tst_et        += et_model.predict(X_te_xgb.fillna(-999)) / N_FOLDS
    print(f"   [EXT] finished")

    # ---- LightGBM (Micro-LR) ----
    lgb_model = lgb.LGBMRegressor(
        n_estimators       = 5000,
        learning_rate      = 0.005,
        num_leaves         = 127,
        max_depth          = -1,
        min_child_samples  = 10,
        subsample          = 0.8,
        subsample_freq     = 1,
        colsample_bytree   = 0.75,
        random_state       = 42,
        n_jobs             = -1,
        verbosity          = -1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set  = [(X_va, y_va)],
        callbacks = [lgb.early_stopping(300, verbose=False), lgb.log_evaluation(-1)]
    )
    oof_lgb[va_idx] = lgb_model.predict(X_va)
    tst_lgb        += lgb_model.predict(X_te) / N_FOLDS
    print(f"   [LGB] best_iter={lgb_model.best_iteration_}")

    # ---- CatBoost (Micro-LR) ----
    cat_model = cb.CatBoostRegressor(
        iterations            = 5000,
        learning_rate         = 0.01,
        depth                 = 8,
        l2_leaf_reg           = 3,
        bagging_temperature   = 0.5,
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
        early_stopping_rounds = 300,
        verbose               = False
    )
    oof_cat[va_idx] = cat_model.predict(X_va_cat)
    tst_cat        += cat_model.predict(X_te_cat) / N_FOLDS
    print(f"   [CAT] best_iter={cat_model.best_iteration_}")

    # ---- Fold ensemble metrics ----
    ens_fold = (oof_xgb[va_idx] + oof_lgb[va_idx] + oof_cat[va_idx] + oof_et[va_idx]) / 4.0
    y_va_arr = y_va.values
    f_mae  = mean_absolute_error(y_va_arr, ens_fold)
    f_rmse = root_mean_squared_error(y_va_arr, ens_fold)
    f_ev   = explained_variance_score(y_va_arr, ens_fold)
    fold_results.append({"fold": fold+1, "MAE": f_mae, "RMSE": f_rmse, "EV": f_ev})
    print(f"   [ENS] MAE={f_mae:.4f}  RMSE={f_rmse:.4f}  EV={f_ev:.4f}  [{time.time()-t0:.0f}s]")

# -----------------------------------------------------------------
# 7. INVERSE-RMSE WEIGHTED ENSEMBLE
# -----------------------------------------------------------------
y_arr    = y.values
rmse_xgb = root_mean_squared_error(y_arr, oof_xgb)
rmse_lgb = root_mean_squared_error(y_arr, oof_lgb)
rmse_cat = root_mean_squared_error(y_arr, oof_cat)
rmse_et  = root_mean_squared_error(y_arr, oof_et)

w_xgb = 1.0 / rmse_xgb;  w_lgb = 1.0 / rmse_lgb;  w_cat = 1.0 / rmse_cat; w_et = 1.0 / rmse_et
total_w = w_xgb + w_lgb + w_cat + w_et

print(f"\n[WGHT] Model weights (inverse-RMSE):")
print(f"   XGB : {w_xgb/total_w:.3f}  (OOF RMSE={rmse_xgb:.5f})")
print(f"   LGB : {w_lgb/total_w:.3f}  (OOF RMSE={rmse_lgb:.5f})")
print(f"   CAT : {w_cat/total_w:.3f}  (OOF RMSE={rmse_cat:.5f})")
print(f"   EXT : {w_et/total_w:.3f}  (OOF RMSE={rmse_et:.5f})")

oof_ensemble = (w_xgb*oof_xgb + w_lgb*oof_lgb + w_cat*oof_cat + w_et*oof_et) / total_w
tst_ensemble = (w_xgb*tst_xgb + w_lgb*tst_lgb + w_cat*tst_cat + w_et*tst_et) / total_w

# -----------------------------------------------------------------
# 8. ISOTONIC CALIBRATION (fixes prediction range collapse)
# -----------------------------------------------------------------
cal = IsotonicRegression(out_of_bounds="clip")
cal.fit(oof_ensemble, y_arr)
tst_calibrated = cal.predict(tst_ensemble)

# -----------------------------------------------------------------
# 9. GLOBAL OOF METRICS
# -----------------------------------------------------------------
g_mae  = mean_absolute_error(y_arr, oof_ensemble)
g_rmse = root_mean_squared_error(y_arr, oof_ensemble)
g_ev   = explained_variance_score(y_arr, oof_ensemble)
oof_cal      = cal.predict(oof_ensemble)
g_mae_cal    = mean_absolute_error(y_arr, oof_cal)
g_rmse_cal   = root_mean_squared_error(y_arr, oof_cal)
g_ev_cal     = explained_variance_score(y_arr, oof_cal)

print("\n" + "="*65)
print("  GLOBAL OOF RESULTS")
print("="*65)
print(f"  [Raw Ensemble]")
print(f"    MAE            : {g_mae:.5f}")
print(f"    RMSE           : {g_rmse:.5f}")
print(f"    Explained Var. : {g_ev:.5f}")
print(f"  [After Isotonic Calibration]")
print(f"    MAE            : {g_mae_cal:.5f}")
print(f"    RMSE           : {g_rmse_cal:.5f}")
print(f"    Explained Var. : {g_ev_cal:.5f}")
print("="*65)

# -----------------------------------------------------------------
# 10. FEATURE IMPORTANCE (XGBoost, top 20)
# -----------------------------------------------------------------
print("\n[INFO] Top 20 Feature Importances (XGB gain -- last fold):")
fi = pd.Series(xgb_model.feature_importances_, index=X_tr_xgb.columns)
fi = fi.sort_values(ascending=False).head(20)
for feat, score in fi.items():
    bar = "#" * int(score / fi.max() * 30)
    print(f"   {feat:<45} {bar}  {score:.4f}")

# -----------------------------------------------------------------
# 11. BOUNDARY CLIP + SUBMISSION
# -----------------------------------------------------------------
tst_final = np.clip(tst_calibrated, 0.0, 1.0)
submission = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": tst_final
})
submission.to_csv("submission_v3.csv", index=False)
print(f"\n[DONE] Submission -> submission_v3.csv  ({len(submission)} rows)")
print(f"       Pred range : [{tst_final.min():.4f}, {tst_final.max():.4f}]  "
      f"mean={tst_final.mean():.4f}")

# -----------------------------------------------------------------
# 12. FOLD REPORT
# -----------------------------------------------------------------
fold_df = pd.DataFrame(fold_results)
fold_df.loc[len(fold_df)] = {"fold": "OVERALL",     "MAE": g_mae,     "RMSE": g_rmse,     "EV": g_ev}
fold_df.loc[len(fold_df)] = {"fold": "OVERALL_CAL", "MAE": g_mae_cal, "RMSE": g_rmse_cal, "EV": g_ev_cal}
fold_df.to_csv("fold_report_v3.csv", index=False)
print("       Fold report -> fold_report_v3.csv")
