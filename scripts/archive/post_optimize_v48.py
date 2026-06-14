"""
Post-Optimization for v48: Simple Global Power Calibration
==========================================================
Identical approach to v42_optimized. No per-group calibration, no analytical blending.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from scipy.optimize import minimize
import os

# Paths
DATA_DIR = "data"
OOF_RAW_PATH = "oof_v48.npy"
SUB_RAW_PATH = "submission_v48.csv"

if not os.path.exists(OOF_RAW_PATH):
    OOF_RAW_PATH = "submissions/oof_v48.npy"
if not os.path.exists(SUB_RAW_PATH):
    SUB_RAW_PATH = "submissions/submission_v48.csv"

print("Loading raw predictions...")
all_oof_stacked = np.load(OOF_RAW_PATH)
tst_df = pd.read_csv(SUB_RAW_PATH)
tst_stacked_avg = tst_df["flood_risk_score"].values

# Load training labels
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).drop_duplicates()
original_y = train_df['flood_risk_score'].values

print(f"OOF shape: {all_oof_stacked.shape}, Test shape: {tst_stacked_avg.shape}")
print(f"Train labels shape: {original_y.shape}")

# Corrected 16-point simulator formula
c_mae, c_rmse, c_ev = 0.583210, 1.122681, 0.045804

# Raw metrics
raw_mae = mean_absolute_error(original_y, all_oof_stacked)
raw_rmse = root_mean_squared_error(original_y, all_oof_stacked)
raw_ev = explained_variance_score(original_y, all_oof_stacked)
raw_lb = (c_mae * raw_mae + c_rmse * raw_rmse) * (1.0 + c_ev * (1.0 - raw_ev))
print(f"\nRaw OOF:  MAE={raw_mae:.5f}  RMSE={raw_rmse:.5f}  EV={raw_ev:.5f}  Est.LB={raw_lb:.5f}")

# -----------------------------------------------------------------
# GLOBAL POWER TRANSFORMATION
# -----------------------------------------------------------------
print("\nOptimizing global power calibration...")

def transform_loss(params):
    a, b, c = params
    pred = a * np.power(np.clip(all_oof_stacked, 1e-6, None), b) + c
    pred = np.clip(pred, 0.0, 1.0)
    
    mae = mean_absolute_error(original_y, pred)
    rmse = root_mean_squared_error(original_y, pred)
    ev = explained_variance_score(original_y, pred)
    
    return (c_mae * mae + c_rmse * rmse) * (1.0 + c_ev * (1.0 - ev))

initial_guess = [1.0, 1.0, 0.0]
bounds = [(0.5, 1.5), (0.5, 2.0), (-0.15, 0.15)]

res_opt = minimize(transform_loss, initial_guess, bounds=bounds, method='L-BFGS-B')
a_opt, b_opt, c_opt = res_opt.x
print(f"Optimal parameters: a={a_opt:.5f}, b={b_opt:.5f}, c={c_opt:.5f}")

# Apply to OOF
opt_oof = a_opt * np.power(np.clip(all_oof_stacked, 1e-6, None), b_opt) + c_opt
opt_oof = np.clip(opt_oof, 0.0, 1.0)

opt_mae = mean_absolute_error(original_y, opt_oof)
opt_rmse = root_mean_squared_error(original_y, opt_oof)
opt_ev = explained_variance_score(original_y, opt_oof)
opt_lb = (c_mae * opt_mae + c_rmse * opt_rmse) * (1.0 + c_ev * (1.0 - opt_ev))

print(f"\nOptimized OOF:  MAE={opt_mae:.5f}  RMSE={opt_rmse:.5f}  EV={opt_ev:.5f}  Est.LB={opt_lb:.5f}")

# Apply to test
opt_test_preds = a_opt * np.power(np.clip(tst_stacked_avg, 1e-6, None), b_opt) + c_opt
opt_test_preds = np.clip(opt_test_preds, 0.0, 1.0)

# Save
np.save("oof_v48_optimized.npy", opt_oof)
np.save("submissions/oof_v48_optimized.npy", opt_oof)

submission_opt = pd.DataFrame({
    "record_id"       : tst_df["record_id"],
    "flood_risk_score": opt_test_preds
})
submission_opt.to_csv("submission_v48_optimized.csv", index=False)
submission_opt.to_csv("submissions/submission_v48_optimized.csv", index=False)

print(f"\n[DONE] Saved submission_v48_optimized.csv ({len(submission_opt)} rows)")
print(f"  Range: [{opt_test_preds.min():.4f}, {opt_test_preds.max():.4f}]")
print(f"  v42_opt benchmark: LB=0.38245  |  v48_opt target: LB<0.38245")
