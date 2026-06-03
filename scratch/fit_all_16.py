import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from scipy.optimize import minimize
import os

# 1. Load ground truth
train_df = pd.read_csv("data/train.csv")
train_df = train_df.drop_duplicates()
y = train_df['flood_risk_score'].values

# Hardcoded versions (without OOF npy files)
data = {
    "v3": (0.179622, 0.235205, 0.028892, 0.38559),
    "v11": (0.179842, 0.235389, 0.027370, 0.38637),
}

# Versions to load from files
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
    "v42_optimized": 0.38245
}

for ver, lb in actual_lbs.items():
    fpath = f"submissions/oof_{ver}.npy"
    if not os.path.exists(fpath):
        fpath = f"oof_{ver}.npy"
        if not os.path.exists(fpath):
            print(f"Warning: {ver} not found!")
            continue
            
    oof = np.load(fpath)
    mae = mean_absolute_error(y, oof)
    rmse = root_mean_squared_error(y, oof)
    ev = explained_variance_score(y, oof)
    data[ver] = (mae, rmse, ev, lb)

print(f"Total versions compiled: {len(data)}")

# Form B: Multiplicative (Error * (1 + Penalty))
# LB = (w0 * MAE + w1 * RMSE) * (1.0 + w2 * (1.0 - EV))
def form_mult(w, mae, rmse, ev):
    return (w[0]*mae + w[1]*rmse) * (1.0 + w[2]*(1.0 - ev))

# Form A: Linear
# LB = w0 * MAE + w1 * RMSE + w2 * (1.0 - EV) + w3
def form_linear(w, mae, rmse, ev):
    return w[0]*mae + w[1]*rmse + w[2]*(1.0 - ev) + w[3]

def fit_formula(name, formula_func, init_guess, bounds=None):
    def loss_func(w):
        err = 0
        for ver, (mae, rmse, ev, lb) in data.items():
            pred = formula_func(w, mae, rmse, ev)
            err += (pred - lb) ** 2
        return err / len(data)
    
    res = minimize(loss_func, init_guess, bounds=bounds, method='L-BFGS-B')
    w_opt = res.x
    
    print(f"\nFitting {name}:")
    errors = []
    for ver, (mae, rmse, ev, lb) in sorted(data.items(), key=lambda x: x[1][3]):
        pred = formula_func(w_opt, mae, rmse, ev)
        err = pred - lb
        errors.append(abs(err))
        print(f"  {ver:<15} -> Pred: {pred:.5f} | Act: {lb:.5f} | Err: {err:+.5f} | MAE: {mae:.5f} | RMSE: {rmse:.5f} | EV: {ev:.5f}")
    print(f"  Optimal weights: {list(w_opt)}")
    print(f"  Max Absolute Error: {max(errors):.6f}")
    print(f"  Mean Absolute Error: {np.mean(errors):.6f}")
    return w_opt

fit_formula("Form B Multiplicative (16 points)", form_mult, [0.64579, 0.53529, 0.61380], bounds=[(0, None), (0, None), (0, None)])
fit_formula("Form A Linear (16 points)", form_linear, [1.0, 1.0, 1.0, 0.0])

