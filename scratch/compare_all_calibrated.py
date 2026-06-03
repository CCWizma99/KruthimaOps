import numpy as np
import pandas as pd
import glob
import os
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from scipy.optimize import minimize

# Load target
y_true = pd.read_csv("data/train.csv").drop_duplicates()["flood_risk_score"].values

oof_files = glob.glob("submissions/oof_v*.npy") + glob.glob("oof_v*.npy")
# Remove duplicates by basename
seen = set()
unique_files = []
for f in oof_files:
    basename = os.path.basename(f)
    if basename not in seen:
        seen.add(basename)
        unique_files.append(f)

c_mae, c_rmse, c_ev = 0.392696, 0.875527, 0.406963

results = []
for f in sorted(unique_files):
    oof = np.load(f)
    if len(oof) != len(y_true):
        # Maybe length doesn't match due to pseudo labeling row changes?
        # Wait, the target y_true is 20881. Let's make sure we align the indices.
        # Actually, in all runs, the OOF is saved for original_y which has length 20881 (since is_pseudo == 0 rows).
        # Let's verify length.
        if len(oof) != len(y_true):
            print(f"Skipping {f} due to length mismatch: {len(oof)} vs {len(y_true)}")
            continue
    
    # Raw metrics
    raw_mae = mean_absolute_error(y_true, oof)
    raw_rmse = root_mean_squared_error(y_true, oof)
    raw_ev = explained_variance_score(y_true, oof)
    raw_lb = (c_mae * raw_mae + c_rmse * raw_rmse) * (1.0 + c_ev * (1.0 - raw_ev))
    
    # Optimize
    def loss(p):
        a, b, c = p
        pred = np.clip(a * np.power(np.clip(oof, 1e-6, None), b) + c, 0.0, 1.0)
        mae = mean_absolute_error(y_true, pred)
        rmse = root_mean_squared_error(y_true, pred)
        ev = explained_variance_score(y_true, pred)
        return (c_mae * mae + c_rmse * rmse) * (1.0 + c_ev * (1.0 - ev))
    
    res = minimize(loss, [1.0, 1.0, 0.0], bounds=[(0.5, 1.5), (0.5, 2.0), (-0.15, 0.15)], method='L-BFGS-B')
    a, b, c = res.x
    pred_opt = np.clip(a * np.power(np.clip(oof, 1e-6, None), b) + c, 0.0, 1.0)
    
    opt_mae = mean_absolute_error(y_true, pred_opt)
    opt_rmse = root_mean_squared_error(y_true, pred_opt)
    opt_ev = explained_variance_score(y_true, pred_opt)
    opt_lb = loss(res.x)
    
    results.append({
        "file": os.path.basename(f),
        "raw_mae": raw_mae, "raw_rmse": raw_rmse, "raw_ev": raw_ev, "raw_lb": raw_lb,
        "opt_mae": opt_mae, "opt_rmse": opt_rmse, "opt_ev": opt_ev, "opt_lb": opt_lb,
        "opt_a": a, "opt_b": b, "opt_c": c
    })

df_res = pd.DataFrame(results)
# Sort by optimized LB (lower is better)
df_res = df_res.sort_values("opt_lb")
print(df_res.to_string(index=False))
