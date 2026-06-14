import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from scipy.optimize import minimize
import os

# Paths
DATA_DIR = "data"
OOF_RAW_PATH = "oof_v47.npy"
SUB_RAW_PATH = "submission_v47.csv"

# Load raw outputs
if not os.path.exists(OOF_RAW_PATH):
    OOF_RAW_PATH = "submissions/oof_v47.npy"
if not os.path.exists(SUB_RAW_PATH):
    SUB_RAW_PATH = "submissions/submission_v47.csv"

print("Loading raw predictions...")
all_oof_stacked = np.load(OOF_RAW_PATH)
tst_df = pd.read_csv(SUB_RAW_PATH)
tst_stacked_avg = tst_df["flood_risk_score"].values

# Load original datasets
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).drop_duplicates()
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Re-create is_pseudo flags (to filter original y)
train_df['is_pseudo'] = 0
test_df['is_pseudo'] = 0

# Soft pseudo-labeling fallback (matches what actually ran)
pseudo_path = "submissions/submission_v45.csv"
if not os.path.exists(pseudo_path):
    pseudo_path = "submission_v45.csv"
    
if os.path.exists(pseudo_path):
    sub_blend = pd.read_csv(pseudo_path)
    test_pseudo = test_df.merge(sub_blend, on="record_id", how="left")
    pseudo_rows = test_pseudo.copy()
    pseudo_rows['is_pseudo'] = 1
    train_df = pd.concat([train_df, pseudo_rows], ignore_index=True)

# Define real mask
real_mask = train_df['is_pseudo'] == 0
original_y = train_df.loc[real_mask, 'flood_risk_score'].values

# Recreate downstream signatures with string fillna fix
for df in [train_df, test_df]:
    df['downstream_sig'] = (
        df['flood_occurrence_current_event'].fillna('missing').astype(str).str.strip() + "_" +
        df['is_good_to_live'].fillna('missing').astype(str).str.strip() + "_" +
        df['reason_not_good_to_live'].fillna('missing').astype(str).str.strip()
    )

GLOBAL_MEDIAN = float(np.median(original_y))

# Re-create Analytical Group Medians
print("Computing analytical group medians...")
# Only compute stats from real train rows
real_tr = train_df[train_df['is_pseudo'] == 0]
sig_stats = real_tr.groupby('downstream_sig')['flood_risk_score'].agg(['median', 'count'])
smoothed_medians = ((sig_stats['count'] * sig_stats['median'] + 10 * GLOBAL_MEDIAN) / (sig_stats['count'] + 10))

oof_analytical_clean = train_df.loc[real_mask, 'downstream_sig'].map(smoothed_medians).fillna(GLOBAL_MEDIAN).values
tst_analytical_fold = test_df['downstream_sig'].map(smoothed_medians).fillna(GLOBAL_MEDIAN).values

# -----------------------------------------------------------------
# POST-HOC CALIBRATION
# -----------------------------------------------------------------
print("Optimizing global calibration...")
c_mae, c_rmse, c_ev = 0.535196, 1.146326, 0.054898

def global_transform_loss(params):
    a, b, c = params
    pred = a * np.power(np.clip(all_oof_stacked, 1e-6, None), b) + c
    pred = np.clip(pred, 0.0, 1.0)
    
    mae = mean_absolute_error(original_y, pred)
    rmse = root_mean_squared_error(original_y, pred)
    ev = explained_variance_score(original_y, pred)
    
    return (c_mae * mae + c_rmse * rmse) * (1.0 + c_ev * (1.0 - ev))

initial_guess = [1.0, 1.0, 0.0]
bounds = [(0.5, 1.5), (0.5, 2.0), (-0.15, 0.15)]

res_glob = minimize(global_transform_loss, initial_guess, bounds=bounds, method='L-BFGS-B')
a_glob, b_glob, c_glob = res_glob.x
print(f"Global parameters: a={a_glob:.5f}, b={b_glob:.5f}, c={c_glob:.5f}")

# Apply global parameters
opt_oof = a_glob * np.power(np.clip(all_oof_stacked, 1e-6, None), b_glob) + c_glob
opt_oof = np.clip(opt_oof, 0.0, 1.0)

opt_test_preds = a_glob * np.power(np.clip(tst_stacked_avg, 1e-6, None), b_glob) + c_glob
opt_test_preds = np.clip(opt_test_preds, 0.0, 1.0)

# Apply group-specific calibration override
print("Optimizing group-specific calibration...")
train_groups = train_df.loc[real_mask, 'downstream_sig'].values
test_groups = test_df['downstream_sig'].values

unique_groups = np.unique(train_groups)
n_calibrated_groups = 0
for grp in unique_groups:
    tr_mask = train_groups == grp
    te_mask = test_groups == grp
    
    if tr_mask.sum() < 10:
        continue
        
    y_grp = original_y[tr_mask]
    p_grp = all_oof_stacked[tr_mask]
    
    def group_loss(params):
        a, b, c = params
        pred_cal = np.clip(a * np.power(np.clip(p_grp, 1e-6, None), b) + c, 0, 1)
        return mean_absolute_error(y_grp, pred_cal)
        
    res_grp = minimize(group_loss, x0=[a_glob, b_glob, c_glob],
                       bounds=[(0.5, 1.5), (0.5, 2.0), (-0.15, 0.15)],
                       method='L-BFGS-B')
    
    if res_grp.success:
        a_grp, b_grp, c_grp = res_grp.x
        opt_oof[tr_mask] = np.clip(a_grp * np.power(np.clip(p_grp, 1e-6, None), b_grp) + c_grp, 0, 1)
        if te_mask.any():
            opt_test_preds[te_mask] = np.clip(a_grp * np.power(np.clip(tst_stacked_avg[te_mask], 1e-6, None), b_grp) + c_grp, 0, 1)
        n_calibrated_groups += 1

print(f"Group calibration complete. Overrode {n_calibrated_groups} groups out of {len(unique_groups)} total groups.")

# Save the raw uncalibrated OOF predictions into the submissions folder so they are visible to evaluate.py
np.save("submissions/oof_v47.npy", all_oof_stacked)
np.save("oof_v47.npy", all_oof_stacked)

# Directly use the calibrated predictions without any analytical blending
final_oof = opt_oof
final_test_preds = opt_test_preds

opt_mae = mean_absolute_error(original_y, final_oof)
opt_rmse = root_mean_squared_error(original_y, final_oof)
opt_ev = explained_variance_score(original_y, final_oof)
opt_lb = (c_mae * opt_mae + c_rmse * opt_rmse) * (1.0 + c_ev * (1.0 - opt_ev))

print(f"\nOptimized (Calibrated, No-Blend) OOF LB Score: {opt_lb:.5f}")
print(f"  MAE: {opt_mae:.5f}, RMSE: {opt_rmse:.5f}, EV: {opt_ev:.5f}")

# Save optimized (calibrated only) predictions
np.save("oof_v47_optimized.npy", final_oof)
np.save("submissions/oof_v47_optimized.npy", final_oof)

submission_opt = pd.DataFrame({
    "record_id"       : test_df["record_id"],
    "flood_risk_score": final_test_preds
})
submission_opt.to_csv("submission_v47_optimized.csv", index=False)
submission_opt.to_csv("submissions/submission_v47_optimized.csv", index=False)

print("[SUCCESS] Raw and optimized (calibrated, no-blend) outputs successfully saved.")
