import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
import glob
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score

def competition_score(mae, rmse, ev):
    return (0.645811 * mae + 0.535795 * rmse) * (1.0 + 0.612783 * (1.0 - ev))

def main():
    print("======================================================================")
    print("  ML OPSIDIAN - ADVANCED OOF BLEND OPTIMIZATION")
    print("======================================================================\n")

    # Load targets
    print("[LOAD] Loading targets from train.csv...")
    train = pd.read_csv("data/train.csv")
    train = train.drop_duplicates()
    y_true = train['flood_risk_score'].values
    print(f"   Targets shape: {y_true.shape}")

    # Discover OOF files
    oof_files = glob.glob("submissions/oof_*.npy")
    oof_data = {}
    
    for f in oof_files:
        basename = os.path.basename(f)
        ver = basename.replace("oof_", "").replace(".npy", "")
        pred = np.load(f)
        if len(pred) == len(y_true):
            oof_data[ver] = pred
        else:
            print(f"   [SKIP] {ver} due to size mismatch: {len(pred)} vs {len(y_true)}")

    # Add v29 from root if not already in submissions
    if os.path.exists("oof_v29.npy") and "v29" not in oof_data:
        pred = np.load("oof_v29.npy")
        if len(pred) == len(y_true):
            oof_data["v29"] = pred

    versions = list(oof_data.keys())
    print(f"\n[LOAD] Loaded {len(versions)} valid OOF versions: {versions}")

    # Evaluate individual versions
    individual_scores = []
    for ver in versions:
        pred = oof_data[ver]
        mae = mean_absolute_error(y_true, pred)
        rmse = root_mean_squared_error(y_true, pred)
        ev = explained_variance_score(y_true, pred)
        lb = competition_score(mae, rmse, ev)
        individual_scores.append({"ver": ver, "MAE": mae, "RMSE": rmse, "EV": ev, "est_LB": lb})
    
    df_ind = pd.DataFrame(individual_scores).sort_values("est_LB")
    print("\n   [INDIVIDUAL OOF SCORES]")
    print(df_ind.to_string(index=False))

    # We select the top N best versions to blend to prevent overfitting to the OOF training set
    # Let's select versions with est_LB < 0.3845
    best_versions = df_ind[df_ind["est_LB"] < 0.3845]["ver"].tolist()
    print(f"\n[SELECT] Selected best versions for blending (est_LB < 0.3845): {best_versions}")
    
    if len(best_versions) < 2:
        print("[ERROR] Not enough high-performing models to blend.")
        return

    # Prepare matrix
    X_blend = np.column_stack([oof_data[v] for v in best_versions])

    # Objective function to minimize
    def objective(weights):
        # Normalize weights to sum to 1
        w = weights / np.sum(weights)
        blend_pred = np.dot(X_blend, w)
        mae = mean_absolute_error(y_true, blend_pred)
        rmse = root_mean_squared_error(y_true, blend_pred)
        ev = explained_variance_score(y_true, blend_pred)
        return competition_score(mae, rmse, ev)

    # Initial guess: equal weights
    init_weights = np.ones(len(best_versions)) / len(best_versions)
    # Constraints: weights must be non-negative
    bounds = [(0, 1) for _ in range(len(best_versions))]
    # Sum of weights = 1 constraint
    constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})

    res = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    opt_w = res.x
    opt_w = opt_w / np.sum(opt_w) # normalize just in case

    print("\n" + "=" * 70)
    print("  OPTIMIZATION RESULT")
    print("=" * 70)
    
    blend_pred = np.dot(X_blend, opt_w)
    g_mae = mean_absolute_error(y_true, blend_pred)
    g_rmse = root_mean_squared_error(y_true, blend_pred)
    g_ev = explained_variance_score(y_true, blend_pred)
    opt_lb = competition_score(g_mae, g_rmse, g_ev)

    print(f"  Optimized Blend OOF Metrics:")
    print(f"    MAE            : {g_mae:.5f}")
    print(f"    RMSE           : {g_rmse:.5f}")
    print(f"    Explained Var. : {g_ev:.5f}")
    print(f"    Est. LB Score  : {opt_lb:.5f}  (Improvement: {df_ind['est_LB'].min() - opt_lb:.5f} vs single best)")
    
    print(f"\n  Optimal Weights:")
    for v, w in zip(best_versions, opt_w):
        if w > 0.001:
            print(f"    {v:<15}: {w:.4f}")

    # Generate the blended submission file
    print("\n[BLEND] Generating blended submission file...")
    sub_files = {}
    for v in best_versions:
        # Check standard path
        sub_path = f"submissions/submission_{v}.csv"
        if not os.path.exists(sub_path) and v == "v29":
            sub_path = "submission_v29.csv"
        
        if os.path.exists(sub_path):
            sub_files[v] = pd.read_csv(sub_path)
        else:
            print(f"   [WARNING] Submission file not found for {v}: {sub_path}. Trying fallback path...")
            # check root
            sub_path = f"submission_{v}.csv"
            if os.path.exists(sub_path):
                sub_files[v] = pd.read_csv(sub_path)
            else:
                print(f"   [ERROR] Could not find submission file for {v}.")
                return

    # Check alignment
    ref_sub = list(sub_files.values())[0]
    for v, df in sub_files.items():
        if len(df) != len(ref_sub):
            print(f"   [ERROR] Submission size mismatch for {v}: {len(df)} vs {len(ref_sub)}")
            return
        if not (df['record_id'] == ref_sub['record_id']).all():
            print(f"   [ERROR] Record ID mismatch for {v}")
            return

    # Blend
    blend_target = np.zeros(len(ref_sub))
    for v, w in zip(best_versions, opt_w):
        blend_target += sub_files[v]['flood_risk_score'].values * w
    
    # Clip to [0, 1]
    blend_target = np.clip(blend_target, 0.0, 1.0)
    
    out_sub = pd.DataFrame({
        "record_id": ref_sub['record_id'],
        "flood_risk_score": blend_target
    })
    
    out_path = "submissions/submission_optimized_super_blend.csv"
    out_sub.to_csv(out_path, index=False)
    out_path_root = "submission_optimized_super_blend.csv"
    out_sub.to_csv(out_path_root, index=False)
    
    print(f"\n[DONE] Saved blended submission to:")
    print(f"  - {out_path}")
    print(f"  - {out_path_root}")

if __name__ == "__main__":
    main()
