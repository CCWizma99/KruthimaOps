import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from scipy.optimize import minimize
import os

# 1. Load ground truth
train_df = pd.read_csv("data/train.csv")
train_df = train_df.drop_duplicates()
y = train_df['flood_risk_score'].values

# Compile all 16 submissions data
data = {
    "v3": (0.179622, 0.235205, 0.028892, 0.38559),
    "v11": (0.179842, 0.235389, 0.027370, 0.38637),
}

actual_lbs = {
    "v10": 0.38598,
    "v10_probe_k3.5": 0.41264,
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
    "v44_optimized": 0.38278
}

for ver, lb in actual_lbs.items():
    fpath = f"submissions/oof_{ver}.npy"
    if not os.path.exists(fpath):
        fpath = f"oof_{ver}.npy"
        if not os.path.exists(fpath):
            continue
            
    oof = np.load(fpath)
    mae = mean_absolute_error(y, oof)
    rmse = root_mean_squared_error(y, oof)
    ev = explained_variance_score(y, oof)
    data[ver] = (mae, rmse, ev, lb)

print(f"Total versions compiled: {len(data)}")

def fit_formula(name, formula_func, init_guess, bounds=None):
    def loss_func(w):
        err = 0
        for ver, (mae, rmse, ev, lb) in data.items():
            pred = formula_func(w, mae, rmse, ev)
            err += (pred - lb) ** 2
        return err / len(data)
    
    res = minimize(loss_func, init_guess, bounds=bounds, method='L-BFGS-B')
    w_opt = res.x
    
    max_err = 0
    mean_err = 0
    print(f"\nFitting {name}:")
    for ver, (mae, rmse, ev, lb) in sorted(data.items(), key=lambda x: x[1][3]):
        pred = formula_func(w_opt, mae, rmse, ev)
        err = pred - lb
        max_err = max(max_err, abs(err))
        mean_err += abs(err)
        print(f"  {ver:<15} -> Pred: {pred:.5f} | Act: {lb:.5f} | Err: {err:+.5f}")
    mean_err /= len(data)
    print(f"  Optimal weights: {list(w_opt)}")
    print(f"  Max Absolute Error: {max_err:.6f}")
    print(f"  Mean Absolute Error: {mean_err:.6f}")
    return w_opt

# --- Form A: Linear (with positive constraint on MAE, RMSE, 1-EV) ---
def form_linear(w, mae, rmse, ev):
    return w[0]*mae + w[1]*rmse + w[2]*(1.0 - ev) + w[3]

# --- Form B: Multiplicative (Error * (1 + Penalty)) ---
def form_mult(w, mae, rmse, ev):
    return (w[0]*mae + w[1]*rmse) * (1.0 + w[2]*(1.0 - ev))

# --- Form C: Single Error Multiplicative (MAE-dominant) ---
def form_mae_mult(w, mae, rmse, ev):
    return mae * (w[0] + w[1]*(1.0 - ev))

# --- Form D: Single Error Multiplicative (RMSE-dominant) ---
def form_rmse_mult(w, mae, rmse, ev):
    return rmse * (w[0] + w[1]*(1.0 - ev))

fit_formula("Form A: Linear (unconstrained)", form_linear, [1.0, 1.0, 1.0, 0.0])
fit_formula("Form A: Linear (constrained positive)", form_linear, [1.0, 1.0, 1.0, 0.0], 
            bounds=[(0, None), (0, None), (0, None), (None, None)])
fit_formula("Form B: Multiplicative", form_mult, [1.0, 1.0, 1.0],
            bounds=[(0, None), (0, None), (0, None)])
fit_formula("Form C: MAE Multiplicative", form_mae_mult, [1.0, 1.0],
            bounds=[(0, None), (0, None)])
fit_formula("Form D: RMSE Multiplicative", form_rmse_mult, [1.0, 1.0],
            bounds=[(0, None), (0, None)])
