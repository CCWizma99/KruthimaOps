import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from scipy.optimize import minimize

# Load target
train_df = pd.read_csv("data/train.csv").drop_duplicates()
y = train_df['flood_risk_score'].values

# Hand-calculated metrics for v3 and v11 (since OOF .npy files are missing but reports are present)
data = {
    "v3": (0.179622, 0.235205, 0.028892, 0.38559),
    "v11": (0.179842, 0.235389, 0.027370, 0.38637)
}

# Rest of the versions (load from OOF files)
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

for ver, lb in actual_lbs.items():
    if ver.endswith("_optimized"):
        fpath = f"submissions/oof_{ver.replace('_optimized', '')}_optimized.npy"
    else:
        fpath = f"submissions/oof_{ver}.npy"
    
    oof = np.load(fpath)
    mae = mean_absolute_error(y, oof)
    rmse = root_mean_squared_error(y, oof)
    ev = explained_variance_score(y, oof)
    data[ver] = (mae, rmse, ev, lb)

# Let's perform fitting on all 14 data points
print("\n" + "=" * 60)
print(f"FITTING MULTIPLICATIVE MODEL ON {len(data)} LB SUBMISSIONS")
print("=" * 60)

def fit_formula(formula_func, init_guess, bounds=None):
    def loss_func(w):
        err = 0
        for ver, (mae, rmse, ev, lb) in data.items():
            pred = formula_func(w, mae, rmse, ev)
            err += (pred - lb) ** 2
        return err / len(data)
    
    res = minimize(loss_func, init_guess, bounds=bounds, method='L-BFGS-B')
    w_opt = res.x
    
    max_err = 0
    print(f"\nResults:")
    for ver, (mae, rmse, ev, lb) in sorted(data.items(), key=lambda x: x[1][3]):
        pred = formula_func(w_opt, mae, rmse, ev)
        err = pred - lb
        max_err = max(max_err, abs(err))
        print(f"  {ver:<15} -> Pred: {pred:.5f} | Act: {lb:.5f} | Err: {err:+.5f}")
    print(f"  Optimal weights: {w_opt}")
    print(f"  Max Absolute Error: {max_err:.5f}")
    print(f"  Mean Absolute Error: {np.mean([abs(formula_func(w_opt, mae, rmse, ev) - lb) for ver, (mae, rmse, ev, lb) in data.items()]):.6f}")
    return w_opt

# --- Form B: Multiplicative (Error * (1 + Penalty)) ---
def form_mult(w, mae, rmse, ev):
    return (w[0]*mae + w[1]*rmse) * (1.0 + w[2]*(1.0 - ev))

fit_formula(form_mult, [1.0, 1.0, 1.0], bounds=[(0, None), (0, None), (0, None)])
print("=" * 60)
