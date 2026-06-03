import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score

# Calibrated competition metric formula
def competition_score(mae, rmse, ev):
    # Updated to the 17-point recalibrated simulator formula
    return (0.535196 * mae + 1.146326 * rmse) * (1.0 + 0.054898 * (1.0 - ev))

def main():
    print("=" * 70)
    print("  ML OPSIDIAN - SUPER BLEND & POWER CALIBRATION OPTIMIZER (v38)")
    print("=" * 70)

    # 1. Load targets
    train_df = pd.read_csv("data/train.csv")
    train_df = train_df.drop_duplicates()
    y_true = train_df['flood_risk_score'].values
    print(f"Targets shape: {y_true.shape}")

    # 2. Load top-tier OOF files and submissions
    versions = ["v30", "v33", "v37", "v38"]
    oofs = {}
    submissions = {}
    
    for ver in versions:
        oof_path = f"submissions/oof_{ver}.npy"
        sub_path = f"submissions/submission_{ver}.csv"
        # Fallbacks to optimized test predictions for test blending
        if ver in ["v37", "v38"]:
            sub_path = f"submissions/submission_{ver}_optimized.csv"
            
        if os.path.exists(oof_path) and os.path.exists(sub_path):
            oofs[ver] = np.load(oof_path)
            submissions[ver] = pd.read_csv(sub_path)
            print(f"Loaded: {ver}")
        else:
            print(f"[ERROR] Missing files for {ver} (OOF: {os.path.exists(oof_path)}, Sub: {os.path.exists(sub_path)})")
            return

    # 3. Create matrix of OOF predictions
    X_meta = np.column_stack([oofs[ver] for ver in versions])
    n_models = len(versions)

    # 4. Joint Objective: Optimize weights and power transformation simultaneously
    # Params: [w_0, w_1, w_2, w_3, a, b, c]
    def joint_objective(params):
        w = params[:n_models]
        # Normalize weights to sum to 1
        w_norm = w / (np.sum(w) + 1e-9)
        
        a, b, c = params[n_models], params[n_models+1], params[n_models+2]
        
        # Raw blend
        blend = np.dot(X_meta, w_norm)
        
        # Power transformation
        safe_blend = np.clip(blend, 1e-6, None)
        calibrated = a * np.power(safe_blend, b) + c
        calibrated = np.clip(calibrated, 0.0, 1.0)
        
        mae = mean_absolute_error(y_true, calibrated)
        rmse = root_mean_squared_error(y_true, calibrated)
        ev = explained_variance_score(y_true, calibrated)
        
        return competition_score(mae, rmse, ev)

    # Set bounds
    # w: [0, 1]
    # a: [0.5, 1.5]
    # b: [0.5, 2.0]
    # c: [-0.15, 0.15]
    bounds = [(0.0, 1.0) for _ in range(n_models)] + [(0.5, 1.5), (0.5, 2.0), (-0.15, 0.15)]
    
    # Run multi-start optimization to find global minimum
    np.random.seed(42)
    starts = []
    
    # Equal weights initialization
    w_init = np.ones(n_models) / n_models
    starts.append(np.concatenate([w_init, [1.0, 1.0, 0.0]]))
    
    # Random initializations
    for _ in range(30):
        w_rand = np.random.uniform(0.1, 1.0, n_models)
        w_rand = w_rand / np.sum(w_rand)
        a_rand = np.random.uniform(0.8, 1.2)
        b_rand = np.random.uniform(0.8, 1.6)
        c_rand = np.random.uniform(-0.05, 0.05)
        starts.append(np.concatenate([w_rand, [a_rand, b_rand, c_rand]]))

    best_score = float('inf')
    best_params = None

    for idx, p0 in enumerate(starts):
        res = minimize(joint_objective, p0, bounds=bounds, method='L-BFGS-B')
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_params = res.x

    # Extract best parameters
    opt_w = best_params[:n_models]
    opt_w_norm = opt_w / np.sum(opt_w)
    opt_a, opt_b, opt_c = best_params[n_models], best_params[n_models+1], best_params[n_models+2]

    print("\n" + "=" * 50)
    print("  OPTIMIZED SUPER BLEND CONFIGURATION")
    print("=" * 50)
    print("Weights:")
    for i, ver in enumerate(versions):
        print(f"  {ver:<15} : {opt_w_norm[i]:.4f}")
    print(f"\nCalibration Parameters:")
    print(f"  Multiplier (a)  : {opt_a:.5f}")
    print(f"  Exponent (b)    : {opt_b:.5f}")
    print(f"  Offset (c)      : {opt_c:.5f}")
    print("=" * 50)

    # 5. Evaluate individual models vs Optimized Super Blend
    print("\nIndividual model scores:")
    for ver in versions:
        pred = oofs[ver]
        mae = mean_absolute_error(y_true, pred)
        rmse = root_mean_squared_error(y_true, pred)
        ev = explained_variance_score(y_true, pred)
        score = competition_score(mae, rmse, ev)
        print(f"  {ver:<15} -> MAE: {mae:.5f} | RMSE: {rmse:.5f} | EV: {ev:.5f} | Est. LB: {score:.5f}")

    # Blended OOF metrics
    blend_pred = np.dot(X_meta, opt_w_norm)
    calibrated_oof = opt_a * np.power(np.clip(blend_pred, 1e-6, None), opt_b) + opt_c
    calibrated_oof = np.clip(calibrated_oof, 0.0, 1.0)
    
    b_mae = mean_absolute_error(y_true, calibrated_oof)
    b_rmse = root_mean_squared_error(y_true, calibrated_oof)
    b_ev = explained_variance_score(y_true, calibrated_oof)
    b_score = competition_score(b_mae, b_rmse, b_ev)

    print("\n" + "=" * 50)
    print(f"  SUPER BLEND OOF -> MAE: {b_mae:.5f} | RMSE: {b_rmse:.5f} | EV: {b_ev:.5f} | Est. LB: {b_score:.5f}")
    print("=" * 50)

    # 6. Generate Blended Submission File
    test_matrix = np.column_stack([submissions[ver]['flood_risk_score'].values for ver in versions])
    blend_test = np.dot(test_matrix, opt_w_norm)
    calibrated_test = opt_a * np.power(np.clip(blend_test, 1e-6, None), opt_b) + opt_c
    calibrated_test = np.clip(calibrated_test, 0.0, 1.0)

    sub_out = pd.DataFrame({
        "record_id": submissions[versions[0]]['record_id'],
        "flood_risk_score": calibrated_test
    })

    out_path_sub = "submissions/submission_super_blend_v38.csv"
    sub_out.to_csv(out_path_sub, index=False)
    out_path_root = "submission_super_blend_v38.csv"
    sub_out.to_csv(out_path_root, index=False)
    
    print(f"\n[DONE] Saved blended submission to:")
    print(f"  - {out_path_sub}")
    print(f"  - {out_path_root}")
    print(f"       Blend range: [{calibrated_test.min():.4f}, {calibrated_test.max():.4f}]")

if __name__ == "__main__":
    main()
