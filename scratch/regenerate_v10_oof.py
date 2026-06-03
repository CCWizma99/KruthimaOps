import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings, time, os

warnings.filterwarnings("ignore")

# 1. LOAD DATA
train_df = pd.read_csv("data/train.csv")
train_df = train_df.drop_duplicates()

# 2. COLUMN TAXONOMY
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

# 3. FEATURE ENGINEERING
def engineer_features(df):
    df = df.copy()
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

    lat = df["latitude"].fillna(df["latitude"].median())
    lon = df["longitude"].fillna(df["longitude"].median())
    df["lat_bin"] = (lat / 0.5).astype(int)
    df["lon_bin"] = (lon / 0.5).astype(int)
    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
    df = df.drop(columns=["inundation_area_sqm"])
    return df

train_df = engineer_features(train_df)

SPATIAL_HELPERS = ["lat_bin", "lon_bin", "grid_id"]
BASE_FEATURES = [c for c in train_df.columns
                 if c not in IGNORE_COLS and c not in SPATIAL_HELPERS]

# Categorical mapping
cat_dtype_map = {}
for col in BASE_FEATURES:
    if col in CAT_FEATURES:
        train_df[col] = train_df[col].fillna("missing").astype(str)
        all_vals = sorted(set(train_df[col].unique()))
        cdt = pd.CategoricalDtype(categories=all_vals, ordered=False)
        train_df[col] = train_df[col].astype(cdt)
        cat_dtype_map[col] = cdt
    elif train_df[col].dtype in ["int64", "float64", "int32", "float32"]:
        train_df[col] = train_df[col].fillna(train_df[col].median())

N_FOLDS     = 5
y           = train_df[TARGET]
GLOBAL_MEAN = float(y.mean())
y_bins = pd.cut(y, bins=10, labels=False)
skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train_df))
oof_lgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))

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

print("Starting 5-fold training for v10 OOF regeneration...")
for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_bins)):
    print(f"Fold {fold+1}")
    tr_rows = train_df.iloc[tr_idx].copy()
    va_rows = train_df.iloc[va_idx].copy()

    # Target Encoding
    dist_enc = tr_rows.groupby("district")[TARGET].mean()
    tr_rows["district_target_enc"] = tr_rows["district"].astype(str).map(dist_enc).fillna(GLOBAL_MEAN).astype(float)
    va_rows["district_target_enc"] = va_rows["district"].astype(str).map(dist_enc).fillna(GLOBAL_MEAN).astype(float)

    grid_enc = tr_rows.groupby("grid_id")[TARGET].mean()
    tr_rows["grid_target_enc"] = tr_rows["grid_id"].astype(str).map(grid_enc).fillna(GLOBAL_MEAN).astype(float)
    va_rows["grid_target_enc"] = va_rows["grid_id"].astype(str).map(grid_enc).fillna(GLOBAL_MEAN).astype(float)

    reason_enc = tr_rows.groupby("reason_not_good_to_live")[TARGET].mean()
    tr_rows["reason_risk_enc"] = tr_rows["reason_not_good_to_live"].astype(str).map(reason_enc).fillna(GLOBAL_MEAN).astype(float)
    va_rows["reason_risk_enc"] = va_rows["reason_not_good_to_live"].astype(str).map(reason_enc).fillna(GLOBAL_MEAN).astype(float)

    FEATURES = BASE_FEATURES + ["district_target_enc", "grid_target_enc", "reason_risk_enc"]
    y_tr = tr_rows[TARGET]
    y_va = va_rows[TARGET]

    X_tr = tr_rows[FEATURES].copy()
    X_va = va_rows[FEATURES].copy()

    for col in cat_feature_names:
        if col in FEATURES:
            cdt = cat_dtype_map[col]
            X_tr[col] = X_tr[col].astype(str).astype(cdt)
            X_va[col] = X_va[col].astype(str).astype(cdt)

    X_tr_xgb = to_xgb_fmt(X_tr);  X_va_xgb = to_xgb_fmt(X_va)
    X_tr_cat = to_cat_fmt(X_tr);   X_va_cat = to_cat_fmt(X_va)

    # XGB
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

    # LGB
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
        callbacks = [lgb.early_stopping(100, verbose=False)]
    )
    oof_lgb[va_idx] = lgb_model.predict(X_va)

    # CatBoost
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

y_arr = y.values
rmse_xgb = root_mean_squared_error(y_arr, oof_xgb)
rmse_lgb = root_mean_squared_error(y_arr, oof_lgb)
rmse_cat = root_mean_squared_error(y_arr, oof_cat)
w_xgb = 1.0 / rmse_xgb;  w_lgb = 1.0 / rmse_lgb;  w_cat = 1.0 / rmse_cat
total_w = w_xgb + w_lgb + w_cat

oof_ensemble = (w_xgb*oof_xgb + w_lgb*oof_lgb + w_cat*oof_cat) / total_w

g_mae  = mean_absolute_error(y_arr, oof_ensemble)
g_rmse = root_mean_squared_error(y_arr, oof_ensemble)
g_ev   = explained_variance_score(y_arr, oof_ensemble)

print(f"Generated v10 OOF: MAE={g_mae:.6f}, RMSE={g_rmse:.6f}, EV={g_ev:.6f}")
np.save("submissions/oof_v10.npy", oof_ensemble)

# Now generate v10_probe_k3.5 OOF
mean_oof = oof_ensemble.mean()
oof_probe = mean_oof + 3.5 * (oof_ensemble - mean_oof)
oof_probe = np.clip(oof_probe, 0.0, 1.0)

p_mae = mean_absolute_error(y_arr, oof_probe)
p_rmse = root_mean_squared_error(y_arr, oof_probe)
p_ev = explained_variance_score(y_arr, oof_probe)

print(f"Generated v10_probe_k3.5 OOF: MAE={p_mae:.6f}, RMSE={p_rmse:.6f}, EV={p_ev:.6f}")
np.save("submissions/oof_v10_probe_k3.5.npy", oof_probe)
print("Saved both OOF files.")
