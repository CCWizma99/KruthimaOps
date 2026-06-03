import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
import os

# Metric constants
c_mae, c_rmse, c_ev = 0.392696, 0.875527, 0.406963

# Load ground truth
train_df = pd.read_csv("data/train.csv").drop_duplicates()
y_true = train_df[train_df["is_synthetic"].isna() | (train_df["is_synthetic"] == 0)]["flood_risk_score"].values
if len(y_true) != 20881:
    # Fallback to len of OOF if there is any mismatch (pseudo labels are removed from OOF anyway)
    # Let's read y from train_df clean
    # Wait, train_df has 20881 rows, and the OOF is saved for the 19411 clean training rows.
    # Let's match it exactly.
    pass

# Let's just load the true clean y from train_df
train_df = train_df.reset_index(drop=True)
# We know the training loop defines:
# real_mask = train_df['is_pseudo'] == 0 (where train_df includes test pseudo rows if USE_PSEUDO is true)
# But wait, original_y is y_arr[real_mask] where y_arr is target of train_df (before pseudo labels are appended or after?)
# Let's check train_v39_kaggle.py logic:
# It loads train_df, appends pseudo_rows from test (1470 rows). So train_df grows to 22351 rows.
# then real_mask = train_df['is_pseudo'] == 0, which corresponds exactly to the original 20881 rows!
# So original_y has exactly 20881 rows (the original training rows).
# Let's load the original train.csv target:
orig_train = pd.read_csv("data/train.csv").drop_duplicates()
y_true = orig_train["flood_risk_score"].values

oof_v39 = np.load("submissions/oof_v39.npy")
print(f"Loaded OOF v39 shape: {oof_v39.shape}")
print(f"Loaded ground truth shape: {y_true.shape}")

# Raw metrics
raw_mae = mean_absolute_error(y_true, oof_v39)
raw_rmse = root_mean_squared_error(y_true, oof_v39)
raw_ev = explained_variance_score(y_true, oof_v39)
raw_lb = (c_mae * raw_mae + c_rmse * raw_rmse) * (1.0 + c_ev * (1.0 - raw_ev))

print("\n--- Raw OOF Metrics (v39) ---")
print(f"  MAE            : {raw_mae:.5f}")
print(f"  RMSE           : {raw_rmse:.5f}")
print(f"  Explained Var. : {raw_ev:.5f}")
print(f"  Est. LB Score  : {raw_lb:.5f}")

# Optimize power calibration
def transform_loss(params):
    a, b, c = params
    pred = a * np.power(np.clip(oof_v39, 1e-6, None), b) + c
    pred = np.clip(pred, 0.0, 1.0)
    
    mae = mean_absolute_error(y_true, pred)
    rmse = root_mean_squared_error(y_true, pred)
    ev = explained_variance_score(y_true, pred)
    
    return (c_mae * mae + c_rmse * rmse) * (1.0 + c_ev * (1.0 - ev))

initial_guess = [1.0, 1.0, 0.0]
bounds = [(0.5, 1.5), (0.5, 2.0), (-0.15, 0.15)]

res_opt = minimize(transform_loss, initial_guess, bounds=bounds, method='L-BFGS-B')
a_opt, b_opt, c_opt = res_opt.x
print(f"\nOptimal parameters: a={a_opt:.5f}, b={b_opt:.5f}, c={c_opt:.5f}")

opt_oof = a_opt * np.power(np.clip(oof_v39, 1e-6, None), b_opt) + c_opt
opt_oof = np.clip(opt_oof, 0.0, 1.0)

opt_mae = mean_absolute_error(y_true, opt_oof)
opt_rmse = root_mean_squared_error(y_true, opt_oof)
opt_ev = explained_variance_score(y_true, opt_oof)
opt_lb = (c_mae * opt_mae + c_rmse * opt_rmse) * (1.0 + c_ev * (1.0 - opt_ev))

print(f"\n--- Optimized OOF Metrics (v39 Calibration) ---")
print(f"  MAE            : {opt_mae:.5f}")
print(f"  RMSE           : {opt_rmse:.5f}")
print(f"  Explained Var. : {opt_ev:.5f}")
print(f"  Est. LB Score  : {opt_lb:.5f}")

# Load raw test predictions from submission_v39.csv
sub_v39 = pd.read_csv("submissions/submission_v39.csv")
tst_preds = sub_v39["flood_risk_score"].values

# Calibrate test predictions
opt_test_preds = a_opt * np.power(np.clip(tst_preds, 1e-6, None), b_opt) + c_opt
opt_test_preds = np.clip(opt_test_preds, 0.0, 1.0)

# Save submission
submission_opt = pd.DataFrame({
    "record_id"       : sub_v39["record_id"],
    "flood_risk_score": opt_test_preds
})
submission_opt.to_csv("submission_v39_optimized.csv", index=False)
submission_opt.to_csv("submissions/submission_v39_optimized.csv", index=False)
print(f"\n[DONE] Overwrote submission_v39_optimized.csv")
print(f"  Optimized range: [{opt_test_preds.min():.4f}, {opt_test_preds.max():.4f}]")
