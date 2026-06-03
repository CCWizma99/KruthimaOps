import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from scipy.optimize import minimize

# Load target
y_true = pd.read_csv("data/train.csv").drop_duplicates()["flood_risk_score"].values
oof = np.load("submissions/oof_v40.npy")

print("Raw OOF Metrics:")
print(f"  MAE: {mean_absolute_error(y_true, oof):.5f}")
print(f"  RMSE: {root_mean_squared_error(y_true, oof):.5f}")
print(f"  EV: {explained_variance_score(y_true, oof):.5f}")

c_mae, c_rmse, c_ev = 0.392696, 0.875527, 0.406963
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

print("\nOptimized OOF Metrics:")
print(f"  Optimal a, b, c: {a:.5f}, {b:.5f}, {c:.5f}")
print(f"  MAE: {mean_absolute_error(y_true, pred_opt):.5f}")
print(f"  RMSE: {root_mean_squared_error(y_true, pred_opt):.5f}")
print(f"  EV: {explained_variance_score(y_true, pred_opt):.5f}")
print(f"  Est LB: {loss(res.x):.5f}")
