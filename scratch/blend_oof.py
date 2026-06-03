import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from scipy.optimize import minimize
import os

# 1. Load ground truth
train_df = pd.read_csv("data/train.csv")
train_df = train_df.drop_duplicates()
y = train_df['flood_risk_score'].values

# 2. Correct competition score function
def competition_score(mae, rmse, ev):
    return (0.645811 * mae + 0.535795 * rmse) * (1.0 + 0.612783 * (1.0 - ev))

# 3. Load OOF predictions
oofs = {}
submissions = {}

versions = ["v13", "v17", "v19", "v20", "v23", "v24", "v28_kaggle"]

for ver in versions:
    oof_path = f"submissions/oof_{ver}.npy"
    sub_path = f"submissions/submission_{ver}.csv"
    if os.path.exists(oof_path) and os.path.exists(sub_path):
        oofs[ver] = np.load(oof_path)
        submissions[ver] = pd.read_csv(sub_path)
        print(f"Loaded: {ver}")

available_vers = list(oofs.keys())
print(f"\nAvailable versions for blending: {available_vers}")

# 4. Fit optimal blend weights
X = np.column_stack([oofs[ver] for ver in available_vers])

def objective(w):
    # Enforce sum to 1 constraint via normalization or penalty
    w_normalized = w / np.sum(w)
    blend = np.dot(X, w_normalized)
    
    mae = mean_absolute_error(y, blend)
    rmse = root_mean_squared_error(y, blend)
    ev = explained_variance_score(y, blend)
    
    return competition_score(mae, rmse, ev)

# Initial guess: equal weights
w0 = np.ones(len(available_vers)) / len(available_vers)
# Constraint: positive weights
bounds = [(0, 1) for _ in range(len(available_vers))]

res = minimize(objective, w0, bounds=bounds, method='L-BFGS-B')
w_opt = res.x / np.sum(res.x)

print("\n" + "="*50)
print("  OPTIMAL BLEND WEIGHTS")
print("="*50)
for i, ver in enumerate(available_vers):
    print(f"  {ver:<12}: {w_opt[i]:.4f}")
print("="*50)

# Evaluate individual vs blend
for ver in available_vers:
    pred = oofs[ver]
    mae = mean_absolute_error(y, pred)
    rmse = root_mean_squared_error(y, pred)
    ev = explained_variance_score(y, pred)
    score = competition_score(mae, rmse, ev)
    print(f"  {ver:<12} -> MAE: {mae:.5f} | RMSE: {rmse:.5f} | EV: {ev:.5f} | Est. LB: {score:.5f}")

# Blended metrics
blend_pred = np.dot(X, w_opt)
b_mae = mean_absolute_error(y, blend_pred)
b_rmse = root_mean_squared_error(y, blend_pred)
b_ev = explained_variance_score(y, blend_pred)
b_score = competition_score(b_mae, b_rmse, b_ev)

print("-" * 50)
print(f"  SUPER BLEND  -> MAE: {b_mae:.5f} | RMSE: {b_rmse:.5f} | EV: {b_ev:.5f} | Est. LB: {b_score:.5f}")
print("="*50)

# 5. Generate blended submission
test_preds = np.column_stack([submissions[ver]['flood_risk_score'].values for ver in available_vers])
blended_test_preds = np.dot(test_preds, w_opt)
blended_test_preds = np.clip(blended_test_preds, 0.0, 1.0)

sub_out = pd.DataFrame({
    "record_id": submissions[available_vers[0]]['record_id'],
    "flood_risk_score": blended_test_preds
})

out_path = "submission_super_blend.csv"
sub_out.to_csv(out_path, index=False)
print(f"\n[DONE] Saved blended submission to {out_path}")
print(f"       Blend range: [{blended_test_preds.min():.4f}, {blended_test_preds.max():.4f}]")
