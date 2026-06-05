import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load the original training targets
train_df = pd.read_csv("data/train.csv")
train_df = train_df.drop_duplicates()
y_true = train_df['flood_risk_score'].values

# 2. Load the OOF predictions for v703
oof_preds = np.load("submissions/oof_v703_optimized.npy")
mean_pred = np.mean(oof_preds)

# 3. Our Metric Evaluator
c_mae, c_rmse, c_ev = 0.539328, 1.152263, 0.048467

def calc_score(y, pred):
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    var_y = np.var(y)
    if var_y == 0:
        ev = 0.0
    else:
        ev = 1.0 - np.var(y - pred) / var_y
    return (c_mae * mae + c_rmse * rmse) * (1.0 + c_ev * (1.0 - ev))

# Base Score
base_score = calc_score(y_true, oof_preds)
print(f"Base Optimized OOF Score: {base_score:.6f}\n")

# 4. Generate the exact same 10 probes but on the OOF data
probes = {
    "probe_01_mul_up_1": oof_preds * 1.001,
    "probe_02_mul_dn_1": oof_preds * 0.999,
    "probe_03_add_up_1": oof_preds + 0.001,
    "probe_04_add_dn_1": oof_preds - 0.001,
    "probe_05_mul_up_2": oof_preds * 1.002,
    "probe_06_mul_dn_2": oof_preds * 0.998,
    "probe_07_add_up_2": oof_preds + 0.002,
    "probe_08_add_dn_2": oof_preds - 0.002,
    "probe_09_var_shrink": (oof_preds - mean_pred) * 0.995 + mean_pred,
    "probe_10_var_expand": (oof_preds - mean_pred) * 1.005 + mean_pred,
}

for name, preds in probes.items():
    preds_clipped = np.clip(preds, 0.0, 1.0)
    score = calc_score(y_true, preds_clipped)
    diff = score - base_score
    print(f"{name:<20} | Local Score: {score:.6f} | Diff: {diff:+.6f}")
