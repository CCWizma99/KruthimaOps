import numpy as np
from scipy.optimize import minimize

data = [
    ("v23", 0.17880, 0.23449, 0.03475, 0.38411),
    ("v20", 0.17864, 0.23438, 0.03564, 0.38331),
    ("v19", 0.17891, 0.23461, 0.03379, 0.38401),
    ("v17", 0.17882, 0.23465, 0.03390, 0.38506),
    ("v13", 0.17937, 0.23500, 0.03060, 0.38476),
    ("v11", 0.17984, 0.23539, 0.02737, 0.38637),
    ("v3",  0.17962, 0.23520, 0.02889, 0.38559)
]

def objective(w):
    errs = []
    for name, mae, rmse, ev, lb in data:
        pred = w[0]*mae + w[1]*rmse + w[2]*(1-ev) + w[3]
        errs.append((pred - lb)**2)
    return np.mean(errs)

# Initial guess from current formula
w0 = [-13.246019, 4.673492, 1.715215, 0.0]

res = minimize(objective, w0)
w = res.x

print("Optimal Weights:")
print(f"MAE  : {w[0]:.6f}")
print(f"RMSE : {w[1]:.6f}")
print(f"EV   : {w[2]:.6f}")
print(f"Const: {w[3]:.6f}")

print("\nValidating:")
max_err = 0
for name, mae, rmse, ev, lb in data:
    pred = w[0]*mae + w[1]*rmse + w[2]*(1-ev) + w[3]
    err = pred - lb
    max_err = max(max_err, abs(err))
    print(f"{name:4} -> Pred: {pred:.5f} | Act: {lb:.5f} | Err: {err:+.5f}")
print(f"Max Error: {max_err:.5f}")
