import numpy as np
import pandas as pd
import os
import glob
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score

# 1. Load ground truth targets
train_df = pd.read_csv("data/train.csv").drop_duplicates()
y_true = train_df['flood_risk_score'].values

# 2. Known LB mapping
known_lb = {
    "v703_hub_oof_te_optimized": 0.38232,
    "v703_7m_optimized": 0.38236,
    "v703_hub_optimized": 0.38235,
    "v702_optimized": 0.38246,
    "combined": 0.51454,
    "v778": 0.38488,
    "v1200_optimized": 0.38276,
    "final_submission_v3": 0.38537,
    "vb58": 0.38239,
    "v1000_optimized": 0.38242,
    "lb": 0.38208,
    "900x": 0.38244,
    "v708_optimized": 0.38244,
    "v703_optimized": 0.38203,
    "v77_optimized": 0.38253,
    "v80_optimized": 0.38240,
    "v70_optimized": 0.38216,
    "v67_optimized": 0.38216,
    "v64_optimized": 0.38256,
    "v63_optimized": 0.38309,
    "v60_optimized": 0.38295,
    "v54_optimized": 0.38337,
    "v45_optimized": 0.38272,
    "v44_optimized": 0.38278,
    "v42_optimized": 0.38245,
    "v38_optimized": 0.38298,
    "v37": 0.38335,
    "v37_optimized": 0.38328,
    "v33": 0.38294,
    "v30": 0.38293,
    "v28_kaggle": 0.38499,
    "v23": 0.38411,
    "v20": 0.38331,
    "v19": 0.38401,
    "v17": 0.38506,
    "v13": 0.38476,
    "v11": 0.38637,
    "v10": 0.38598,
    "v10_probe_k3.5": 0.41264,
    "v3": 0.38559
}

# 3. Gather local metrics
data = []
oof_files = glob.glob("submissions/oof_*.npy")
for path in oof_files:
    basename = os.path.basename(path)
    if basename.startswith("oof_"):
        ver = basename[4:-4]
    else:
        ver = basename.replace(".npy", "")
    
    if ver in known_lb:
        y_pred = np.load(path)
        if len(y_pred) != len(y_true):
            continue
            
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        ev   = explained_variance_score(y_true, y_pred)
        lb   = known_lb[ver]
        
        data.append({
            "version": ver,
            "mae": mae,
            "rmse": rmse,
            "ev": ev,
            "actual_lb": lb
        })

df = pd.DataFrame(data)
print(f"Loaded {len(df)} matching submissions with OOF files.")
print(df.to_string(index=False))

if len(df) < 4:
    print("Not enough data points to fit parameters!")
    exit(1)

# 4. Fit the formula parameters: LB = (c_mae * MAE + c_rmse * RMSE) * (1.0 + c_ev * (1.0 - EV)) + c_int
def loss_function(params):
    c_mae, c_rmse, c_ev, c_int = params
    pred_lbs = (c_mae * df['mae'] + c_rmse * df['rmse']) * (1.0 + c_ev * (1.0 - df['ev'])) + c_int
    residuals = pred_lbs - df['actual_lb']
    return np.sum(residuals ** 2)

# Initial guess (from evaluate.py)
init_guess = [0.563014, 1.168097, 0.141008, -0.041736]

res = minimize(loss_function, init_guess, method='L-BFGS-B', bounds=[(0.0, None), (0.0, None), (0.0, None), (None, None)])
c_mae_opt, c_rmse_opt, c_ev_opt, c_int_opt = res.x

print("\n" + "=" * 50)
print("  OPTIMIZED COEFFICIENTS")
print("=" * 50)
print(f"  c_mae  : {c_mae_opt:.6f}")
print(f"  c_rmse : {c_rmse_opt:.6f}")
print(f"  c_ev   : {c_ev_opt:.6f}")
print(f"  c_int  : {c_int_opt:.6f}")
print("=" * 50)

# Print residual errors
df['pred_lb'] = (c_mae_opt * df['mae'] + c_rmse_opt * df['rmse']) * (1.0 + c_ev_opt * (1.0 - df['ev'])) + c_int_opt
df['error'] = df['pred_lb'] - df['actual_lb']
df['abs_error'] = np.abs(df['error'])

print("\nDetailed Errors post-refit:")
print(df[['version', 'actual_lb', 'pred_lb', 'error']].to_string(index=False))
print(f"\nMean Absolute Error of fit: {df['abs_error'].mean():.6f}")
print(f"Max Absolute Error of fit : {df['abs_error'].max():.6f}")

# Save the results to scripts/refit_results.txt
with open("scripts/refit_results.txt", "w") as f:
    f.write(f"c_mae={c_mae_opt:.6f}\n")
    f.write(f"c_rmse={c_rmse_opt:.6f}\n")
    f.write(f"c_ev={c_ev_opt:.6f}\n")
    f.write(f"c_int={c_int_opt:.6f}\n")
    f.write(f"mean_abs_error={df['abs_error'].mean():.6f}\n")
    f.write(f"max_abs_error={df['abs_error'].max():.6f}\n")

