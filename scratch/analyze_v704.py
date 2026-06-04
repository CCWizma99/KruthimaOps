import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score

train_df = pd.read_csv('data/train.csv').drop_duplicates()
y_true   = train_df['flood_risk_score'].values

oof_raw  = np.load('oof_v704.npy')
oof_opt  = np.load('oof_v704_optimized.npy')
oof_v703     = np.load('oof_v703.npy')
oof_v703_opt = np.load('oof_v703_optimized.npy')
oof_v70      = np.load('oof_v70.npy')
oof_v70_opt  = np.load('oof_v70_optimized.npy')

def metrics(y, p, label):
    mae  = mean_absolute_error(y, p)
    rmse = root_mean_squared_error(y, p)
    ev   = explained_variance_score(y, p)
    lb   = (0.563014*mae + 1.168097*rmse)*(1.0 + 0.141008*(1.0 - ev)) - 0.041736
    print(f"  {label:<30} MAE={mae:.5f}  RMSE={rmse:.5f}  EV={ev:.5f}  estLB={lb:.5f}")
    return mae, rmse, ev, lb

print("=== PERFORMANCE COMPARISON ===")
metrics(y_true, oof_v70,     "v70 (raw)")
metrics(y_true, oof_v70_opt, "v70_optimized [LB 0.382160]")
print()
metrics(y_true, oof_v703,     "v703 (raw)")
metrics(y_true, oof_v703_opt, "v703_optimized [LB 0.382030]")
print()
metrics(y_true, oof_raw, "v704 (raw)")
metrics(y_true, oof_opt, "v704_optimized")

# Deltas
mae703, rmse703, ev703 = mean_absolute_error(y_true, oof_v703_opt), root_mean_squared_error(y_true, oof_v703_opt), explained_variance_score(y_true, oof_v703_opt)
mae704, rmse704, ev704 = mean_absolute_error(y_true, oof_opt), root_mean_squared_error(y_true, oof_opt), explained_variance_score(y_true, oof_opt)

print()
print("=== DELTA v704_opt vs v703_opt ===")
dm = mae704 - mae703
dr = rmse704 - rmse703
de = ev704 - ev703
print(f"  dMAE  : {dm:+.6f}  ({'BETTER' if dm < 0 else 'WORSE'})")
print(f"  dRMSE : {dr:+.6f}  ({'BETTER' if dr < 0 else 'WORSE'})")
print(f"  dEV   : {de:+.6f}  ({'BETTER' if de > 0 else 'WORSE'})")

print()
print("=== FOLD REPORT v704 ===")
fr = pd.read_csv('fold_report_v704.csv')
print(fr.to_string(index=False))
print(f"  Mean  MAE={fr['MAE'].mean():.5f}  RMSE={fr['RMSE'].mean():.5f}  EV={fr['EV'].mean():.5f}")
print(f"  Std   MAE={fr['MAE'].std():.5f}  RMSE={fr['RMSE'].std():.5f}  EV={fr['EV'].std():.5f}")

print()
print("=== FOLD REPORT v703 (for comparison) ===")
fr703 = pd.read_csv('fold_report_v703.csv')
print(fr703.to_string(index=False))
print(f"  Mean  MAE={fr703['MAE'].mean():.5f}  RMSE={fr703['RMSE'].mean():.5f}  EV={fr703['EV'].mean():.5f}")

print()
print("=== PREDICTION STATS ===")
for label, arr in [("v703_opt", oof_v703_opt), ("v704_raw", oof_raw), ("v704_opt", oof_opt)]:
    print(f"  {label}: mean={arr.mean():.5f}  std={arr.std():.5f}  min={arr.min():.4f}  max={arr.max():.4f}")

# Check post-hoc transform effect
print()
print("=== POST-HOC TRANSFORM EFFECT ===")
lb_raw = (0.563014*mean_absolute_error(y_true, oof_raw) + 1.168097*root_mean_squared_error(y_true, oof_raw))*(1.0 + 0.141008*(1.0 - explained_variance_score(y_true, oof_raw))) - 0.041736
lb_opt = (0.563014*mae704 + 1.168097*rmse704)*(1.0 + 0.141008*(1.0 - ev704)) - 0.041736
print(f"  raw   estLB={lb_raw:.5f}")
print(f"  opt   estLB={lb_opt:.5f}")
print(f"  gain from post-hoc: {lb_raw - lb_opt:+.5f}")
