"""
ML Opsidian: Genesis - Deep Non-Linearity & Noise Pruning (v5)
==============================================================
Emergency Pivot Strategy:
  - Models: Neural Network (MLPRegressor) and Ridge (Linear Baseline)
  - Preprocessing: One-Hot Encoding + StandardScaler
  - Noise Pruning: Lasso (L1) Feature Selection to drop dead weight
  - Loss: MLP uses MSE (Adam optimizer), Ridge uses MSE.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor
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
# 2. FEATURE ENGINEERING
# -----------------------------------------------------------------
TARGET = "flood_risk_score"
ID_COL = "record_id"
DROP_COLS = [ID_COL, "place_name", "is_synthetic", "generation_date", TARGET]

def engineer_features(df):
    df = df.copy()
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

    df["is_repeat_flood_zone"] = (df["historical_flood_count"] > 2).astype(int)
    df["rain_spike_ratio"]     = df["rainfall_7d_mm"] / (df["monthly_rainfall_mm"] + 1e-6)
    df["confirmed_risk"]       = (
        (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes") &
        (df["is_good_to_live"].astype(str).str.strip().str.lower() == "no")
    ).astype(int)
    df["vulnerability"]        = (
        df["evacuation_difficulty"] * df["population_density_per_km2_log1p"] / (df["infrastructure_score"] + 1.0)
    )
    df["water_veg_combined"] = (df["ndwi_qmap"].clip(-3,3)/6.0 + 0.5) * (1.0 - (df["ndvi_qmap"].clip(-3,3)/6.0 + 0.5))

    lat = df["latitude"].fillna(df["latitude"].median())
    lon = df["longitude"].fillna(df["longitude"].median())
    df["lat_bin"] = (lat / 0.5).astype(int)
    df["lon_bin"] = (lon / 0.5).astype(int)
    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
    return df

print("[FEAT] Engineering features...")
train_df = engineer_features(train_df)
test_df  = engineer_features(test_df)

CAT_COLS = [
    "district", "landcover", "soil_type", "water_supply",
    "electricity", "road_quality", "urban_rural",
    "water_presence_flag", "flood_occurrence_current_event",
    "is_good_to_live", "reason_not_good_to_live"
]

# -----------------------------------------------------------------
# 3. GLOBAL ONE-HOT ENCODING & IMPUTATION
# -----------------------------------------------------------------
# Combine for consistent dummy columns
n_train = len(train_df)
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
for col in CAT_COLS:
    combined[col] = combined[col].fillna("missing").astype(str)

combined = pd.get_dummies(combined, columns=CAT_COLS, dummy_na=False)

# Re-split
train_df = combined.iloc[:n_train].copy()
test_df  = combined.iloc[n_train:].copy()

NUMERIC_COLS = [c for c in train_df.columns if c not in DROP_COLS + ["grid_id", "lat_bin", "lon_bin"]]

# Fill remaining NaNs with median
print("[PREP] Imputing missing values...")
for col in NUMERIC_COLS:
    med = train_df[col].median()
    train_df[col] = train_df[col].fillna(med)
    test_df[col]  = test_df[col].fillna(med)

# -----------------------------------------------------------------
# 4. CV SETUP
# -----------------------------------------------------------------
N_FOLDS     = 5
y           = train_df[TARGET]
GLOBAL_MEAN = float(y.mean())
y_bins      = pd.cut(y, bins=10, labels=False)
skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_mlp   = np.zeros(len(train_df))
oof_ridge = np.zeros(len(train_df))
tst_mlp   = np.zeros(len(test_df))
tst_ridge = np.zeros(len(test_df))

fold_results = []

print("\n" + "="*65)
print("  5-FOLD STRATIFIED CV -- Neural Network (MLP) & Ridge")
print("="*65)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_bins)):
    t0 = time.time()
    print(f"\n>> Fold {fold+1}/{N_FOLDS}")

    tr_rows = train_df.iloc[tr_idx].copy()
    va_rows = train_df.iloc[va_idx].copy()
    te_rows = test_df.copy()

    # ---- Phase 2: Spatial Aggregations (in-fold only) ----
    agg_cols = ["rainfall_7d_mm", "elevation_m", "distance_to_river_m", "inundation_area_sqm", "infrastructure_score"]
    added_agg_feats = []
    
    # We must group by the original 'district' values, but we one-hot encoded it!
    # Let's extract the district one-hot columns and use them, or just use grid_id target encoding.
    # To keep it simple and matrix-based, we'll only do grid_id target encoding.
    grid_enc = tr_rows.groupby("grid_id")[TARGET].mean()
    tr_rows["grid_target_enc"] = tr_rows["grid_id"].map(grid_enc).fillna(GLOBAL_MEAN)
    va_rows["grid_target_enc"] = va_rows["grid_id"].map(grid_enc).fillna(GLOBAL_MEAN)
    te_rows["grid_target_enc"] = te_rows["grid_id"].map(grid_enc).fillna(GLOBAL_MEAN)

    FOLD_FEATURES = NUMERIC_COLS + ["grid_target_enc"]

    X_tr = tr_rows[FOLD_FEATURES].values
    X_va = va_rows[FOLD_FEATURES].values
    X_te = te_rows[FOLD_FEATURES].values
    y_tr = tr_rows[TARGET].values
    y_va = va_rows[TARGET].values

    # ---- Scaling ----
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    # ---- Lasso Feature Pruning ----
    # Fit a strict Lasso model to drop noisy features
    lasso = Lasso(alpha=0.005, random_state=42)
    lasso.fit(X_tr_sc, y_tr)
    # Get mask of features that survived (coef != 0)
    surviving_mask = (lasso.coef_ != 0)
    num_surviving = surviving_mask.sum()
    
    # Fallback if Lasso kills too much (alpha too high)
    if num_surviving < 10:
        print(f"   [WARN] Lasso pruned too heavily (only {num_surviving} left). Falling back to alpha=0.001")
        lasso = Lasso(alpha=0.001, random_state=42)
        lasso.fit(X_tr_sc, y_tr)
        surviving_mask = (lasso.coef_ != 0)
        num_surviving = surviving_mask.sum()
        
    print(f"   [LASSO] Pruned features from {X_tr_sc.shape[1]} down to {num_surviving}")

    X_tr_pruned = X_tr_sc[:, surviving_mask]
    X_va_pruned = X_va_sc[:, surviving_mask]
    X_te_pruned = X_te_sc[:, surviving_mask]

    # ---- Ridge Baseline ----
    ridge = Ridge(alpha=100.0, random_state=42)
    ridge.fit(X_tr_pruned, y_tr)
    oof_ridge[va_idx] = ridge.predict(X_va_pruned)
    tst_ridge        += ridge.predict(X_te_pruned) / N_FOLDS

    # ---- Neural Network (MLP) ----
    # 2 hidden layers, early stopping to prevent overfit
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.1,             # L2 regularization
        learning_rate_init=0.005,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42
    )
    mlp.fit(X_tr_pruned, y_tr)
    oof_mlp[va_idx] = mlp.predict(X_va_pruned)
    tst_mlp        += mlp.predict(X_te_pruned) / N_FOLDS
    
    print(f"   [MLP] Stopped at epoch {mlp.n_iter_}")

    # Fold Metrics
    ens_fold = (oof_ridge[va_idx] + oof_mlp[va_idx]) / 2.0
    f_mae  = mean_absolute_error(y_va, ens_fold)
    f_rmse = root_mean_squared_error(y_va, ens_fold)
    f_ev   = explained_variance_score(y_va, ens_fold)
    fold_results.append({"fold": fold+1, "MAE": f_mae, "RMSE": f_rmse, "EV": f_ev})
    print(f"   [ENS] MAE={f_mae:.4f}  RMSE={f_rmse:.4f}  EV={f_ev:.4f}  [{time.time()-t0:.0f}s]")

# -----------------------------------------------------------------
# 5. GLOBAL OOF METRICS
# -----------------------------------------------------------------
y_arr = y.values
oof_ensemble = (oof_ridge + oof_mlp) / 2.0
tst_ensemble = (tst_ridge + tst_mlp) / 2.0

cal = IsotonicRegression(out_of_bounds="clip")
cal.fit(oof_ensemble, y_arr)
oof_cal = cal.predict(oof_ensemble)
tst_cal = cal.predict(tst_ensemble)

g_mae  = mean_absolute_error(y_arr, oof_ensemble)
g_rmse = root_mean_squared_error(y_arr, oof_ensemble)
g_ev   = explained_variance_score(y_arr, oof_ensemble)

g_mae_cal  = mean_absolute_error(y_arr, oof_cal)
g_rmse_cal = root_mean_squared_error(y_arr, oof_cal)
g_ev_cal   = explained_variance_score(y_arr, oof_cal)

print("\n" + "="*65)
print("  GLOBAL OOF RESULTS (Neural Net + Ridge)")
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
# 6. SUBMISSION
# -----------------------------------------------------------------
tst_final = np.clip(tst_cal, 0.0, 1.0)
submission = pd.DataFrame({
    "record_id"       : test_df.iloc[:, 0] if 'record_id' in test_df.columns else pd.read_csv("test.csv")["record_id"],
    "flood_risk_score": tst_final
})
submission.to_csv("submission_v5.csv", index=False)
print(f"\n[DONE] Submission -> submission_v5.csv")

fold_df = pd.DataFrame(fold_results)
fold_df.to_csv("fold_report_v5.csv", index=False)
