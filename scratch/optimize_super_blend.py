import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score

def competition_score(mae, rmse, ev):
    return (0.645811 * mae + 0.535795 * rmse) * (1.0 + 0.612783 * (1.0 - ev))

def main():
    print("======================================================================")
    print("  ML OPSIDIAN - SUPER BLEND WEIGHT OPTIMIZATION (MULTI-START)")
    print("======================================================================\n")

    # Load targets
    train_df = pd.read_csv("data/train.csv")
    train_df = train_df.drop_duplicates()
    y_true = train_df['flood_risk_score'].values
    print(f"Targets shape: {y_true.shape}")

    # Load OOF files
    oofs = {}
    submissions = {}
    
    versions = ["v17", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27_kaggle", "v29", "v30"]
    
    for ver in versions:
        oof_path = f"submissions/oof_{ver}.npy"
        if not os.path.exists(oof_path) and ver == "v29":
            oof_path = "oof_v29.npy"
            
        sub_path = f"submissions/submission_{ver}.csv"
        if not os.path.exists(sub_path) and ver == "v29":
            sub_path = "submission_v29.csv"
            
        if os.path.exists(oof_path) and os.path.exists(sub_path):
            oofs[ver] = np.load(oof_path)
            submissions[ver] = pd.read_csv(sub_path)
            print(f"Loaded: {ver}")
        else:
            print(f"Failed to load: {ver} (oof_exists: {os.path.exists(oof_path)}, sub_exists: {os.path.exists(sub_path)})")

    available_vers = list(oofs.keys())
    print(f"\nAvailable versions for blending: {available_vers}")
    
    X = np.column_stack([oofs[ver] for ver in available_vers])
    
    # Objective function
    def objective(w):
        # Normalize weights to sum to 1
        w_norm = w / np.sum(w)
        blend = np.dot(X, w_norm)
        
        mae = mean_absolute_error(y_true, blend)
        rmse = root_mean_squared_error(y_true, blend)
        ev = explained_variance_score(y_true, blend)
        
        return competition_score(mae, rmse, ev)

    bounds = [(0, 1) for _ in range(len(available_vers))]
    constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})

    best_score = float('inf')
    best_weights = None

    # Multi-start optimization with different initializations
    np.random.seed(42)
    starts = [
        np.ones(len(available_vers)) / len(available_vers), # Equal weights
    ]
    # Add random starts
    for _ in range(20):
        w_rand = np.random.uniform(0.1, 1.0, len(available_vers))
        starts.append(w_rand / np.sum(w_rand))

    for idx, w0 in enumerate(starts):
        # Try both SLSQP and L-BFGS-B
        res_slsqp = minimize(objective, w0, bounds=bounds, method='SLSQP', constraints=constraints)
        if res_slsqp.success and res_slsqp.fun < best_score:
            best_score = res_slsqp.fun
            best_weights = res_slsqp.x / np.sum(res_slsqp.x)

        res_lbfgs = minimize(objective, w0, bounds=bounds, method='L-BFGS-B')
        if res_lbfgs.success:
            w_norm = res_lbfgs.x / np.sum(res_lbfgs.x)
            score_lbfgs = objective(w_norm)
            if score_lbfgs < best_score:
                best_score = score_lbfgs
                best_weights = w_norm

    print("\n" + "="*50)
    print("  OPTIMAL SUPER BLEND WEIGHTS")
    print("="*50)
    for i, ver in enumerate(available_vers):
        if best_weights[i] > 0.0001:
            print(f"  {ver:<12}: {best_weights[i]:.4f}")
    print("="*50)

    # Evaluate individual vs blend
    print("\nIndividual model scores:")
    for ver in available_vers:
        pred = oofs[ver]
        mae = mean_absolute_error(y_true, pred)
        rmse = root_mean_squared_error(y_true, pred)
        ev = explained_variance_score(y_true, pred)
        score = competition_score(mae, rmse, ev)
        print(f"  {ver:<12} -> MAE: {mae:.5f} | RMSE: {rmse:.5f} | EV: {ev:.5f} | Est. LB: {score:.5f}")

    # Blended metrics
    blend_pred = np.dot(X, best_weights)
    b_mae = mean_absolute_error(y_true, blend_pred)
    b_rmse = root_mean_squared_error(y_true, blend_pred)
    b_ev = explained_variance_score(y_true, blend_pred)
    b_score = competition_score(b_mae, b_rmse, b_ev)

    print("\n" + "="*50)
    print(f"  SUPER BLEND  -> MAE: {b_mae:.5f} | RMSE: {b_rmse:.5f} | EV: {b_ev:.5f} | Est. LB: {b_score:.5f}")
    print("="*50)

    # Generate blended submission
    test_preds = np.column_stack([submissions[ver]['flood_risk_score'].values for ver in available_vers])
    blended_test_preds = np.dot(test_preds, best_weights)
    blended_test_preds = np.clip(blended_test_preds, 0.0, 1.0)

    sub_out = pd.DataFrame({
        "record_id": submissions[available_vers[0]]['record_id'],
        "flood_risk_score": blended_test_preds
    })

    out_path = "submissions/submission_optimized_super_blend.csv"
    sub_out.to_csv(out_path, index=False)
    out_path_root = "submission_optimized_super_blend.csv"
    sub_out.to_csv(out_path_root, index=False)
    print(f"\n[DONE] Saved blended submission to:")
    print(f"  - {out_path}")
    print(f"  - {out_path_root}")
    print(f"       Blend range: [{blended_test_preds.min():.4f}, {blended_test_preds.max():.4f}]")

if __name__ == "__main__":
    main()
