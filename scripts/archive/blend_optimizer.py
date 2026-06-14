"""
ML Opsidian: OOF Blending Optimizer
===========================================================================
Loads top OOF predictions from the submissions folder and uses L-BFGS-B
to find the optimal linear combination that minimizes the competition metric.
If successful, applies the same weights to the submission files.
"""

import numpy as np
import pandas as pd
import os
import glob
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from scipy.optimize import minimize

# -----------------------------------------------------------------
# 1. CONFIGURATION & METRIC
# -----------------------------------------------------------------
DATA_DIR = "data"
SUB_DIR = "submissions"

_C_MAE  = 0.563014
_C_RMSE = 1.168097
_C_EV   = 0.141008
_C_INT  = -0.041736

def competition_score(mae, rmse, ev):
    return (_C_MAE * mae + _C_RMSE * rmse) * (1.0 + _C_EV * (1.0 - ev)) + _C_INT

def evaluate_predictions(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    ev   = explained_variance_score(y_true, y_pred)
    return competition_score(mae, rmse, ev), mae, rmse, ev

# -----------------------------------------------------------------
# 2. LOAD DATA
# -----------------------------------------------------------------
print("Loading ground truth labels...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train_df = train_df.drop_duplicates()
y_true = train_df['flood_risk_score'].values

test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
record_ids = test_df['record_id'].values

# -----------------------------------------------------------------
# 3. DISCOVER OOF FILES
# -----------------------------------------------------------------
print("\nScanning for OOF arrays...")
oof_files = glob.glob(os.path.join(SUB_DIR, "oof_*.npy"))

models = []
for f in oof_files:
    preds = np.load(f)
    if len(preds) != len(y_true):
        continue
    
    score, mae, rmse, ev = evaluate_predictions(y_true, preds)
    name = os.path.basename(f).replace("oof_", "").replace(".npy", "")
    
    sub_file = os.path.join(SUB_DIR, f"submission_{name}.csv")
    if not os.path.exists(sub_file):
        # Fallback check root folder
        sub_file = f"submission_{name}.csv"
        if not os.path.exists(sub_file):
            continue # Can't use it if we don't have the test predictions
            
    models.append({
        "name": name,
        "oof_path": f,
        "sub_path": sub_file,
        "preds": preds,
        "score": score,
        "mae": mae,
        "rmse": rmse,
        "ev": ev
    })

models.sort(key=lambda x: x['score'])

print(f"\nTop 15 Available Models:")
for m in models[:15]:
    print(f"  {m['name']:<20} | Score: {m['score']:.5f} | MAE: {m['mae']:.5f}")

if not models:
    print("No valid OOF/Submission pairs found.")
    exit()

# -----------------------------------------------------------------
# 4. SELECT TOP MODELS FOR BLENDING
# -----------------------------------------------------------------
TOP_N = min(15, len(models))
selected_models = models[:TOP_N]
best_single_score = selected_models[0]['score']

print(f"\nSelecting top {TOP_N} models for blending optimization.")
X_meta = np.column_stack([m['preds'] for m in selected_models])

# -----------------------------------------------------------------
# 5. L-BFGS-B OPTIMIZATION
# -----------------------------------------------------------------
print("\nOptimizing blend weights...")

def loss_fn(weights):
    # Normalize weights to sum to 1
    w = weights / np.sum(weights)
    pred = np.dot(X_meta, w)
    pred = np.clip(pred, 0.0, 1.0)
    score, _, _, _ = evaluate_predictions(y_true, pred)
    return score

n_models = X_meta.shape[1]
init_guess = np.ones(n_models) / n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]

res = minimize(loss_fn, init_guess, bounds=bounds, method='L-BFGS-B')
final_weights = res.x / np.sum(res.x)

blended_oof = np.clip(np.dot(X_meta, final_weights), 0.0, 1.0)
blend_score, b_mae, b_rmse, b_ev = evaluate_predictions(y_true, blended_oof)

print("\n" + "=" * 60)
print(f"  BLENDING RESULTS")
print("=" * 60)
print(f"  Best Single Model: {selected_models[0]['name']} ({best_single_score:.5f})")
print(f"  Blended OOF Score: {blend_score:.5f}")
print(f"  Delta            : {blend_score - best_single_score:+.5f}")
print("=" * 60)
print(f"  Blended MAE      : {b_mae:.5f}")
print(f"  Blended RMSE     : {b_rmse:.5f}")
print(f"  Blended EV       : {b_ev:.5f}")
print("\n  [WEIGHTS]")
for i, m in enumerate(selected_models):
    if final_weights[i] > 0.001:
        print(f"    {m['name']:<20} : {final_weights[i]:.4f}")

# -----------------------------------------------------------------
# 6. GENERATE FINAL SUBMISSION
# -----------------------------------------------------------------
if blend_score < best_single_score:
    print("\n[SUCCESS] Blend improved the score! Generating final submission...")
    
    test_preds = np.zeros(len(record_ids))
    for i, m in enumerate(selected_models):
        if final_weights[i] > 0.001:
            sub_df = pd.read_csv(m['sub_path'])
            test_preds += sub_df['flood_risk_score'].values * final_weights[i]
            
    test_preds = np.clip(test_preds, 0.0, 1.0)
    
    sub_final = pd.DataFrame({
        "record_id": record_ids,
        "flood_risk_score": test_preds
    })
    
    out_path = os.path.join(SUB_DIR, "submission_blend_final.csv")
    sub_final.to_csv(out_path, index=False)
    np.save(os.path.join(SUB_DIR, "oof_blend_final.npy"), blended_oof)
    
    print(f"  Saved to: {out_path}")
    print(f"  Test Predictions Range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
else:
    print("\n[INFO] Blend did not improve upon the best single model. No submission generated.")
