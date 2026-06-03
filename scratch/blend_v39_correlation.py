"""
ML Opsidian: Genesis — Correlation-Aware OOF Portfolio Blender
==============================================================
Uses existing saved OOF arrays (v30 through v38) to find the
portfolio-optimal blend weights that minimise correlated errors.

Steps:
  1. Load OOF arrays and ground truth from train.csv
  2. Compute pairwise Pearson error correlation matrix
  3. Run scipy L-BFGS-B portfolio optimiser to find metric-optimal weights
  4. Generate final blend submission and report metrics
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
import os

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
DATA_DIR = "data"
SUB_DIR  = "submissions"

# Versions to consider — only include versions that have saved OOF
VERSIONS = ["v30", "v33", "v37", "v38"]

# Metric constants (calibrated against all 10 known LB points)
C_MAE = 0.392696
C_RMSE = 0.875527
C_EV   = 0.406963

def lb_estimate(y_true, pred):
    mae  = mean_absolute_error(y_true, pred)
    rmse = root_mean_squared_error(y_true, pred)
    ev   = explained_variance_score(y_true, pred)
    return (C_MAE * mae + C_RMSE * rmse) * (1.0 + C_EV * (1.0 - ev)), mae, rmse, ev

# ---------------------------------------------------------------
# 1. LOAD GROUND TRUTH (non-pseudo rows only)
# ---------------------------------------------------------------
print("=" * 65)
print("  ML OPSIDIAN — CORRELATION-AWARE OOF PORTFOLIO BLEND")
print("=" * 65)

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).drop_duplicates()
y_true = train_df["flood_risk_score"].values
print(f"[LOAD] Train rows: {len(train_df)}")

# ---------------------------------------------------------------
# 2. LOAD OOF ARRAYS
# ---------------------------------------------------------------
oofs = {}
for v in VERSIONS:
    path_local = f"oof_{v}.npy"
    path_sub   = os.path.join(SUB_DIR, f"oof_{v}.npy")
    for p in [path_local, path_sub]:
        if os.path.exists(p):
            arr = np.load(p)
            if len(arr) == len(y_true):
                oofs[v] = arr
                print(f"[OOF ] Loaded oof_{v}.npy  ({len(arr)} rows)")
            break

loaded_versions = list(oofs.keys())
if len(loaded_versions) < 2:
    raise RuntimeError("Need at least 2 OOF arrays. Run train scripts first.")

# ---------------------------------------------------------------
# 3. INDIVIDUAL MODEL METRICS
# ---------------------------------------------------------------
print("\n--- Individual OOF Metrics ---")
print(f"{'Version':<8}  {'MAE':<8}  {'RMSE':<8}  {'EV':<8}  {'Est.LB':<8}")
print("-" * 50)
for v in loaded_versions:
    score, mae, rmse, ev = lb_estimate(y_true, np.clip(oofs[v], 0, 1))
    print(f"{v:<8}  {mae:.5f}  {rmse:.5f}  {ev:.5f}  {score:.5f}")

# ---------------------------------------------------------------
# 4. PAIRWISE ERROR CORRELATION MATRIX
# ---------------------------------------------------------------
errors = {v: oofs[v] - y_true for v in loaded_versions}
E = np.column_stack([errors[v] for v in loaded_versions])
corr_matrix = np.corrcoef(E.T)

print("\n--- Pairwise OOF Error Correlation Matrix ---")
header = f"{'':8}" + "".join(f"{v:>8}" for v in loaded_versions)
print(header)
for i, vi in enumerate(loaded_versions):
    row = f"{vi:<8}" + "".join(f"{corr_matrix[i, j]:>8.4f}" for j in range(len(loaded_versions)))
    print(row)

# ---------------------------------------------------------------
# 5. PORTFOLIO-OPTIMAL BLEND WEIGHTS
# ---------------------------------------------------------------
OOF_MATRIX = np.column_stack([np.clip(oofs[v], 0, 1) for v in loaded_versions])
n = len(loaded_versions)

def blend_metric(weights):
    pred = np.dot(OOF_MATRIX, weights)
    pred = np.clip(pred, 0.0, 1.0)
    mae  = mean_absolute_error(y_true, pred)
    rmse = root_mean_squared_error(y_true, pred)
    ev   = explained_variance_score(y_true, pred)
    return (C_MAE * mae + C_RMSE * rmse) * (1.0 + C_EV * (1.0 - ev))

# Try multiple starting points to avoid local minima
best_weights = None
best_score   = np.inf
print("\n[OPT] Running portfolio optimiser (multiple starting points)...")

# Equal weights
starts = [np.ones(n) / n]
# Individual best
for i in range(n):
    w = np.zeros(n)
    w[i] = 1.0
    starts.append(w)
# Random restarts
rng = np.random.default_rng(42)
for _ in range(20):
    w = rng.dirichlet(np.ones(n))
    starts.append(w)

for w0 in starts:
    res = minimize(
        blend_metric,
        w0,
        method='L-BFGS-B',
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 500, "ftol": 1e-10}
    )
    if res.fun < best_score:
        best_score   = res.fun
        best_weights = res.x

# Normalise
best_weights = np.clip(best_weights, 0.0, None)
best_weights /= best_weights.sum()

print("\n--- Portfolio Blend Weights ---")
for v, w in zip(loaded_versions, best_weights):
    print(f"   {v:<8}: {w:.4f}")

blend_oof = np.clip(np.dot(OOF_MATRIX, best_weights), 0.0, 1.0)
score, mae, rmse, ev = lb_estimate(y_true, blend_oof)
print(f"\n--- Portfolio Blend OOF Metrics ---")
print(f"   MAE       : {mae:.5f}")
print(f"   RMSE      : {rmse:.5f}")
print(f"   EV        : {ev:.5f}")
print(f"   Est. LB   : {score:.5f}")

# ---------------------------------------------------------------
# 6. POST-HOC POWER CALIBRATION ON BLEND OOF
# ---------------------------------------------------------------
print("\n[CAL] Applying post-hoc power calibration...")

def transform_loss(params):
    a, b, c = params
    pred = a * np.power(np.clip(blend_oof, 1e-6, None), b) + c
    pred = np.clip(pred, 0.0, 1.0)
    mae  = mean_absolute_error(y_true, pred)
    rmse = root_mean_squared_error(y_true, pred)
    ev   = explained_variance_score(y_true, pred)
    return (C_MAE * mae + C_RMSE * rmse) * (1.0 + C_EV * (1.0 - ev))

res_cal = minimize(transform_loss, [1.0, 1.0, 0.0],
                   bounds=[(0.5, 1.5), (0.5, 2.0), (-0.15, 0.15)],
                   method='L-BFGS-B')
a_opt, b_opt, c_opt = res_cal.x
print(f"   a={a_opt:.5f}, b={b_opt:.5f}, c={c_opt:.5f}")

cal_oof = a_opt * np.power(np.clip(blend_oof, 1e-6, None), b_opt) + c_opt
cal_oof = np.clip(cal_oof, 0.0, 1.0)
score2, mae2, rmse2, ev2 = lb_estimate(y_true, cal_oof)
print(f"\n--- Calibrated Blend OOF Metrics ---")
print(f"   MAE       : {mae2:.5f}")
print(f"   RMSE      : {rmse2:.5f}")
print(f"   EV        : {ev2:.5f}")
print(f"   Est. LB   : {score2:.5f}")

# ---------------------------------------------------------------
# 7. BUILD TEST SUBMISSION
# ---------------------------------------------------------------
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Load test predictions for each version from their submission CSVs
print("\n[SUB] Loading test predictions for blend...")
test_preds_list = []
missing = []
for v in loaded_versions:
    # Prefer the optimised submission if available
    candidates = [
        os.path.join(SUB_DIR, f"submission_{v}_optimized.csv"),
        os.path.join(SUB_DIR, f"submission_{v}.csv"),
        f"submission_{v}_optimized.csv",
        f"submission_{v}.csv",
    ]
    found = False
    for p in candidates:
        if os.path.exists(p):
            sub = pd.read_csv(p)
            sub = sub.sort_values("record_id").reset_index(drop=True)
            test_preds_list.append(sub["flood_risk_score"].values)
            print(f"   Loaded {p}")
            found = True
            break
    if not found:
        missing.append(v)

if missing:
    print(f"   [WARNING] Could not find test predictions for: {missing}")
    for v in missing:
        idx = loaded_versions.index(v)
        best_weights[idx] = 0.0
    if best_weights.sum() > 0:
        best_weights /= best_weights.sum()

if len(test_preds_list) == len(loaded_versions):
    TST_MATRIX = np.column_stack(test_preds_list)
    blend_test = np.clip(np.dot(TST_MATRIX, best_weights), 0.0, 1.0)
    # Apply same power calibration to test
    blend_test_cal = a_opt * np.power(np.clip(blend_test, 1e-6, None), b_opt) + c_opt
    blend_test_cal = np.clip(blend_test_cal, 0.0, 1.0)

    # Save submissions
    out_sub = pd.DataFrame({
        "record_id": test_df["record_id"],
        "flood_risk_score": blend_test
    })
    out_sub_cal = pd.DataFrame({
        "record_id": test_df["record_id"],
        "flood_risk_score": blend_test_cal
    })
    out_sub.to_csv("submission_portfolio_blend.csv", index=False)
    out_sub.to_csv(os.path.join(SUB_DIR, "submission_portfolio_blend.csv"), index=False)
    out_sub_cal.to_csv("submission_portfolio_blend_cal.csv", index=False)
    out_sub_cal.to_csv(os.path.join(SUB_DIR, "submission_portfolio_blend_cal.csv"), index=False)
    print(f"\n[DONE] submission_portfolio_blend.csv        (raw blend)")
    print(f"[DONE] submission_portfolio_blend_cal.csv    (calibrated blend)")
    print(f"   Test pred range (raw): [{blend_test.min():.4f}, {blend_test.max():.4f}]")
    print(f"   Test pred range (cal): [{blend_test_cal.min():.4f}, {blend_test_cal.max():.4f}]")
else:
    print("[WARNING] Skipping test submission due to missing test predictions.")

print("\n" + "=" * 65)
print(f"  SUMMARY: Best Est. LB = {min(score, score2):.5f}")
print(f"  {'Blend weights: ' + ', '.join(f'{v}={w:.3f}' for v, w in zip(loaded_versions, best_weights))}")
print("=" * 65)
