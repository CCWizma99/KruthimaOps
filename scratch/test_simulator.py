import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score

# Load target
y = pd.read_csv("data/train.csv").drop_duplicates()["flood_risk_score"].values

actual_lbs = {
    "v13": 0.38476,
    "v17": 0.38506,
    "v19": 0.38401,
    "v20": 0.38331,
    "v23": 0.38411,
    "v28_kaggle": 0.38499,
    "v30": 0.38293,
    "v33": 0.38294,
    "v37": 0.38335,
    "v37_optimized": 0.38328,
    "v38_optimized": 0.38298,
    "v42_optimized": 0.38245
}

w_eval = [0.392696, 0.875527, 0.406963]
w_b = [0.64579372, 0.53529254, 0.61379803]

print("\n--- COMPARE ERRORS ---")
print(f"{'Ver':<15} | {'Eval Err':<12} | {'Form B Err':<12}")
print("-" * 45)

errs_eval = []
errs_b = []

for ver, lb in actual_lbs.items():
    if ver.endswith("_optimized"):
        fpath = f"submissions/oof_{ver.replace('_optimized', '')}_optimized.npy"
    else:
        fpath = f"submissions/oof_{ver}.npy"
    
    if not os.path.exists(fpath):
        fpath = f"oof_{ver}.npy"
        if not os.path.exists(fpath):
            print(f"Skipping {ver} (not found)")
            continue
            
    oof = np.load(fpath)
    mae = mean_absolute_error(y, oof)
    rmse = root_mean_squared_error(y, oof)
    ev = explained_variance_score(y, oof)
    
    p_eval = (w_eval[0] * mae + w_eval[1] * rmse) * (1.0 + w_eval[2] * (1.0 - ev))
    p_b = (w_b[0] * mae + w_b[1] * rmse) * (1.0 + w_b[2] * (1.0 - ev))
    
    err_eval = p_eval - lb
    err_b = p_b - lb
    
    print(f"{ver:<15} | {err_eval:+.6f} | {err_b:+.6f}")
    errs_eval.append(abs(err_eval))
    errs_b.append(abs(err_b))

print("-" * 45)
print(f"Max Err Eval   : {max(errs_eval):.6f}")
print(f"Max Err Form B : {max(errs_b):.6f}")
print(f"Mean Err Eval  : {np.mean(errs_eval):.6f}")
print(f"Mean Err Form B: {np.mean(errs_b):.6f}")
