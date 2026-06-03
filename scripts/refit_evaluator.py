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
    "v3": 0.38559,
    "v10": 0.38598,
    "v10_probe_k3.5": 0.41264,
    "v11": 0.38637,
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
    "v42_optimized": 0.38245,
    "v44_optimized": 0.38278,
    "v45_optimized": 0.38272,
    "v54_optimized": 0.38337,
    "v60_optimized": 0.38295,
    "v63_optimized": 0.38309,
    "v64_optimized": 0.38256,
    "v67_optimized": 0.38216,
    "v70_optimized": 0.38216
}

# 3. Gather local metrics
data = []
oof_files = glob.glob("submissions/oof_*.npy")
for path in oof_files:
    basename = os.path.basename(path)
    ver = basename.replace("oof_", "").replace(".npy", "")
    
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

if len(df) < 3:
    print("Not enough data points to fit 3 parameters!")
    exit(1)

# 4. Fit the formula parameters: LB = (c_mae * MAE + c_rmse * RMSE) * (1.0 + c_ev * (1.0 - EV))
def loss_function(params):
    c_mae, c_rmse, c_ev = params
    pred_lbs = (c_mae * df['mae'] + c_rmse * df['rmse']) * (1.0 + c_ev * (1.0 - df['ev']))
    residuals = pred_lbs - df['actual_lb']
    return np.sum(residuals ** 2)

# Initial guess (from evaluate.py)
init_guess = [0.535196, 1.146326, 0.054898]

res = minimize(loss_function, init_guess, method='L-BFGS-B', bounds=[(0.0, None), (0.0, None), (0.0, None)])
c_mae_opt, c_rmse_opt, c_ev_opt = res.x

print("\n" + "=" * 50)
print("  OPTIMIZED COEFFICIENTS")
print("=" * 50)
print(f"  c_mae  : {c_mae_opt:.6f}")
print(f"  c_rmse : {c_rmse_opt:.6f}")
print(f"  c_ev   : {c_ev_opt:.6f}")
print("=" * 50)

# Print residual errors
df['pred_lb'] = (c_mae_opt * df['mae'] + c_rmse_opt * df['rmse']) * (1.0 + c_ev_opt * (1.0 - df['ev']))
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
    f.write(f"mean_abs_error={df['abs_error'].mean():.6f}\n")
    f.write(f"max_abs_error={df['abs_error'].max():.6f}\n")
