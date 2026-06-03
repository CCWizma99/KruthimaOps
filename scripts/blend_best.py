import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score

def evaluate_blend(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)
    lb_score = -13.246019 * mae + 4.673492 * rmse + 1.715215 * (1.0 - ev)
    return lb_score, mae, rmse, ev

def main():
    print("======================================================================")
    print("  ML OPSIDIAN - OPTIMAL OOF BLENDING (GRID SEARCH)")
    print("======================================================================\n")

    # Load targets
    print("[LOAD] Loading targets from train.csv...")
    train = pd.read_csv("data/train.csv")
    train = train[train['is_synthetic'].isna()].reset_index(drop=True)
    y_true = train['flood_risk_score'].values

    # Load OOFs
    print("[LOAD] Loading OOF predictions...")
    try:
        oof_v13 = np.load("submissions/oof_v13.npy")
        oof_v20 = np.load("submissions/oof_v20.npy")
        oof_v22 = np.load("submissions/oof_v22.npy")
    except FileNotFoundError as e:
        print(f"Error loading OOFs: {e}")
        return

    # Individual scores
    for name, oof in zip(['v13', 'v20', 'v22'], [oof_v13, oof_v20, oof_v22]):
        lb, mae, rmse, ev = evaluate_blend(y_true, oof)
        print(f"{name} Baseline: LB={lb:.5f} | MAE={mae:.5f} | RMSE={rmse:.5f} | EV={ev:.5f}")

    print("\n[SEARCH] Performing grid search over weights...")
    
    best_score = float('inf')
    best_weights = None
    best_metrics = None
    
    # Grid search step 0.05
    steps = np.arange(0, 1.05, 0.05)
    for w1 in steps:
        for w2 in steps:
            if w1 + w2 > 1.0:
                continue
            w3 = 1.0 - (w1 + w2)
            
            w3 = round(w3, 2)
            w2 = round(w2, 2)
            w1 = round(w1, 2)
            if abs(w1 + w2 + w3 - 1.0) > 1e-5:
                continue

            blend = (w1 * oof_v13) + (w2 * oof_v20) + (w3 * oof_v22)
            lb, mae, rmse, ev = evaluate_blend(y_true, blend)
            
            if lb < best_score:
                best_score = lb
                best_weights = (w1, w2, w3)
                best_metrics = (mae, rmse, ev)

    print(f"\n[RESULT] BEST BLEND FOUND!")
    print(f"Weights: v13 = {best_weights[0]:.2f}, v20 = {best_weights[1]:.2f}, v22 = {best_weights[2]:.2f}")
    print(f"Estimated LB Score: {best_score:.5f}")
    print(f"MAE: {best_metrics[0]:.5f}, RMSE: {best_metrics[1]:.5f}, EV: {best_metrics[2]:.5f}")

    print("\n[CREATE] Generating final blended submission...")
    sub_v13 = pd.read_csv("submissions/submission_v13.csv")
    sub_v20 = pd.read_csv("submissions/submission_v20.csv")
    sub_v22 = pd.read_csv("submissions/submission_v22.csv")

    blend_sub = sub_v13.copy()
    blend_sub['flood_risk_score'] = (
        best_weights[0] * sub_v13['flood_risk_score'] +
        best_weights[1] * sub_v20['flood_risk_score'] +
        best_weights[2] * sub_v22['flood_risk_score']
    )
    
    # Apply hard bounds
    blend_sub['flood_risk_score'] = blend_sub['flood_risk_score'].clip(0.0, 1.0)
    
    out_file = "submissions/submission_optimal_blend.csv"
    blend_sub.to_csv(out_file, index=False)
    print(f"Saved optimized blend to {out_file}")

if __name__ == "__main__":
    main()
