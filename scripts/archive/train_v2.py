"""
ML Opsidian: Genesis - Full Ensemble Pipeline v2
=================================================
Models: XGBoost + LightGBM + CatBoost (equal-weight average)
CV:     5-Fold KFold (shuffle, seed=42)
Metric: OOF MAE, RMSE, Explained Variance per fold + overall
Rules:  AGENTS.md compliant -- no leakage, no external data
Python: 3.14 (cp1252 terminal -- ASCII-safe prints only)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings, time

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# 1. LOAD & BASIC DEDUPLICATE
# -----------------------------------------------------------------
print("[LOAD] Loading data...")
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

print(f"   Train shape: {train_df.shape}")
print(f"   Test  shape: {test_df.shape}")

train_df = train_df.drop_duplicates()
print(f"   Train after dedup: {train_df.shape}")

# -----------------------------------------------------------------
# 2. COLUMN TAXONOMY
# -----------------------------------------------------------------
TARGET      = "flood_risk_score"
ID_COL      = "record_id"

# Hard-drop: leakage / metadata / high-cardinality string IDs
DROP_COLS   = [ID_COL, "place_name", "is_synthetic", "generation_date"]

# String categoricals that encode nominal classes -- must stay as 'category'
CAT_FEATURES = [
    "district", "landcover", "soil_type", "water_supply",
    "electricity", "road_quality", "urban_rural",
    "water_presence_flag", "flood_occurrence_current_event",
    "is_good_to_live", "reason_not_good_to_live"
]

IGNORE_COLS = DROP_COLS + [TARGET]
FEATURES    = [c for c in train_df.columns if c not in IGNORE_COLS]

# -----------------------------------------------------------------
# 3. FEATURE ENGINEERING (on top of pre-existing transforms)
# -----------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    # Interaction: proximity x rainfall intensity (proxy for flood exposure)
    df["river_rain_interaction"]  = df["distance_to_river_m_log1p"] * df["rainfall_7d_mm_log1p"]
    df["river_monthly_exposure"]  = df["distance_to_river_m_log1p"] * df["monthly_rainfall_mm_log1p"]

    # Elevation-adjusted risk: lower elevation + high rain = higher risk
    df["elev_rain_risk"] = df["elevation_m_yeojohnson"] / (df["rainfall_7d_mm_log1p"] + 1e-6)

    # Water signal strength: ndwi positive = standing water
    df["water_signal"] = df["ndwi_qmap"].clip(lower=0)

    # Drainage deficit: low drainage + high rain = flood trap
    df["drainage_deficit"] = (df["rainfall_7d_mm_log1p"] + 1) * (1.0 - df["drainage_index_yeojohnson"].clip(0, 1))

    # Infrastructure resilience index
    df["infra_resilience"] = df["infrastructure_score"] / (df["population_density_per_km2_log1p"] + 1e-6)

    # Hospital + Evac reachability composite (lower = more isolated = higher risk)
    df["evacuation_difficulty"] = df["nearest_hospital_km_log1p"] + df["nearest_evac_km_log1p"]

    # Inundation per population density (area covered vs people at risk)
    df["inundation_density_risk"] = (
        np.log1p(df["inundation_area_sqm"]) / (df["population_density_per_km2_log1p"] + 1e-6)
    )

    # Terrain x NDVI: rough terrain + low vegetation = erosion risk
    df["terrain_veg_risk"] = df["terrain_roughness_index"] * (1.0 - df["ndvi_qmap"].clip(-1, 1))

    # Composite flood pressure score
    df["flood_pressure"] = (
        df["extreme_weather_index"] * df["seasonal_index"].clip(lower=0)
    )

    return df

print("[FEAT] Engineering features...")
train_df = engineer_features(train_df)
test_df  = engineer_features(test_df)

# Rebuild feature list after engineering
FEATURES = [c for c in train_df.columns if c not in IGNORE_COLS]
print(f"   Total features: {len(FEATURES)}")

# -----------------------------------------------------------------
# 4. DTYPE CASTING
# -----------------------------------------------------------------
print("[PREP] Casting dtypes...")
for col in FEATURES:
    if col in CAT_FEATURES:
        # CatBoost requires no NaN in cat columns -- fill before alignment
        train_df[col] = train_df[col].fillna("missing").astype(str)
        test_df[col]  = test_df[col].fillna("missing").astype(str)
        # Align categories across train+test so no unseen category errors
        combined_cats = pd.Categorical(
            pd.concat([train_df[col], test_df[col]], ignore_index=True)
        ).categories
        train_df[col] = pd.Categorical(train_df[col], categories=combined_cats)
        test_df[col]  = pd.Categorical(test_df[col],  categories=combined_cats)
    elif train_df[col].dtype in ["int64", "float64", "int32", "float32"]:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col]  = test_df[col].fillna(median_val)

# -----------------------------------------------------------------
# 5. BUILD X, y, X_test (DataFrame-first, no .values for X)
# -----------------------------------------------------------------
X      = train_df[FEATURES]
y      = train_df[TARGET]          # keep as Series for cleaner indexing
X_test = test_df[FEATURES]

print(f"\n   X shape      : {X.shape}")
print(f"   y range      : [{y.min():.3f}, {y.max():.3f}]")
print(f"   X_test shape : {X_test.shape}")

# -----------------------------------------------------------------
# 6. CROSS-VALIDATION SETUP
# -----------------------------------------------------------------
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train_df))
oof_lgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))

tst_xgb = np.zeros(len(test_df))
tst_lgb = np.zeros(len(test_df))
tst_cat = np.zeros(len(test_df))

fold_results = []

# CatBoost needs explicit cat feature names present in FEATURES
cat_feature_names = [c for c in CAT_FEATURES if c in FEATURES]

print("\n" + "="*60)
print("  5-FOLD CV -- XGBoost + LightGBM + CatBoost ENSEMBLE")
print("="*60)

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
    t0 = time.time()
    print(f"\n>> Fold {fold+1}/{N_FOLDS}")

    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

    # -- XGBoost --------------------------------------------------
    xgb_model = xgb.XGBRegressor(
        n_estimators         = 3000,
        learning_rate        = 0.02,
        max_depth            = 7,
        min_child_weight     = 3,
        subsample            = 0.8,
        colsample_bytree     = 0.75,
        colsample_bylevel    = 0.75,
        reg_alpha            = 0.1,
        reg_lambda           = 1.0,
        gamma                = 0.05,
        tree_method          = "hist",
        enable_categorical   = True,
        early_stopping_rounds= 50,   # XGBoost 3.x: must be in constructor, not fit()
        random_state         = 42,
        n_jobs               = -1
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set             = [(X_va, y_va)],
        verbose              = False
    )
    oof_xgb[va_idx] = xgb_model.predict(X_va)
    tst_xgb        += xgb_model.predict(X_test) / N_FOLDS
    print(f"   [XGB] best_iteration={xgb_model.best_iteration}")

    # -- LightGBM -------------------------------------------------
    lgb_model = lgb.LGBMRegressor(
        n_estimators         = 3000,
        learning_rate        = 0.02,
        num_leaves           = 63,
        max_depth            = -1,
        min_child_samples    = 20,
        subsample            = 0.8,
        subsample_freq       = 1,
        colsample_bytree     = 0.75,
        reg_alpha            = 0.1,
        reg_lambda           = 1.0,
        random_state         = 42,
        n_jobs               = -1,
        verbosity            = -1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set             = [(X_va, y_va)],
        callbacks            = [lgb.early_stopping(50, verbose=False),
                                lgb.log_evaluation(-1)]
    )
    oof_lgb[va_idx] = lgb_model.predict(X_va)
    tst_lgb        += lgb_model.predict(X_test) / N_FOLDS
    print(f"   [LGB] best_iteration={lgb_model.best_iteration_}")

    # -- CatBoost -------------------------------------------------
    cat_model = cb.CatBoostRegressor(
        iterations           = 3000,
        learning_rate        = 0.02,
        depth                = 7,
        l2_leaf_reg          = 3,
        bagging_temperature  = 0.5,
        random_strength      = 1,
        border_count         = 128,
        loss_function        = "RMSE",
        eval_metric          = "RMSE",
        task_type            = "CPU",
        random_seed          = 42,
        verbose              = False
    )
    cat_model.fit(
        X_tr, y_tr,
        cat_features         = cat_feature_names,
        eval_set             = (X_va, y_va),
        early_stopping_rounds= 50,
        verbose              = False
    )
    oof_cat[va_idx] = cat_model.predict(X_va)
    tst_cat        += cat_model.predict(X_test) / N_FOLDS
    print(f"   [CAT] best_iteration={cat_model.best_iteration_}")

    # -- Fold metrics (ensemble = equal average) ------------------
    oof_ens_fold = (oof_xgb[va_idx] + oof_lgb[va_idx] + oof_cat[va_idx]) / 3.0
    y_va_arr = y_va.values

    f_mae  = mean_absolute_error(y_va_arr, oof_ens_fold)
    f_rmse = root_mean_squared_error(y_va_arr, oof_ens_fold)
    f_ev   = explained_variance_score(y_va_arr, oof_ens_fold)

    fold_results.append({"fold": fold+1, "MAE": f_mae, "RMSE": f_rmse, "EV": f_ev})
    elapsed = time.time() - t0
    print(f"   [ENS] MAE={f_mae:.4f}  RMSE={f_rmse:.4f}  EV={f_ev:.4f}  [{elapsed:.0f}s]")

# -----------------------------------------------------------------
# 7. GLOBAL OOF METRICS
# -----------------------------------------------------------------
oof_ensemble = (oof_xgb + oof_lgb + oof_cat) / 3.0
y_arr = y.values

g_mae  = mean_absolute_error(y_arr, oof_ensemble)
g_rmse = root_mean_squared_error(y_arr, oof_ensemble)
g_ev   = explained_variance_score(y_arr, oof_ensemble)

print("\n" + "="*60)
print("  GLOBAL OOF RESULTS")
print("="*60)
print(f"  MAE             : {g_mae:.5f}")
print(f"  RMSE            : {g_rmse:.5f}")
print(f"  Explained Var.  : {g_ev:.5f}   (target -> 1.0)")
print("="*60)

# -----------------------------------------------------------------
# 8. FEATURE IMPORTANCE (XGBoost gain, top 20)
# -----------------------------------------------------------------
print("\n[INFO] Top 20 Feature Importances (XGB gain -- last fold):")
fi = pd.Series(xgb_model.feature_importances_, index=FEATURES)
fi = fi.sort_values(ascending=False).head(20)
for feat, score in fi.items():
    bar = "#" * int(score / fi.max() * 30)
    print(f"   {feat:<45} {bar}  {score:.4f}")

# -----------------------------------------------------------------
# 9. FINAL ENSEMBLE PREDICTION + BOUNDARY CLIP
# -----------------------------------------------------------------
tst_ensemble = (tst_xgb + tst_lgb + tst_cat) / 3.0
tst_ensemble = np.clip(tst_ensemble, 0.0, 1.0)

submission = pd.DataFrame({
    "record_id"       : test_df[ID_COL],
    "flood_risk_score": tst_ensemble
})
submission.to_csv("submission_v2.csv", index=False)
print(f"\n[DONE] Submission saved -> submission_v2.csv  ({len(submission)} rows)")
print(f"       Pred range: [{tst_ensemble.min():.4f}, {tst_ensemble.max():.4f}]  "
      f"mean={tst_ensemble.mean():.4f}")

# -----------------------------------------------------------------
# 10. SAVE FOLD REPORT
# -----------------------------------------------------------------
fold_df = pd.DataFrame(fold_results)
fold_df.loc[len(fold_df)] = ["OVERALL", g_mae, g_rmse, g_ev]
fold_df.to_csv("fold_report_v2.csv", index=False)
print("       Fold report  -> fold_report_v2.csv")
