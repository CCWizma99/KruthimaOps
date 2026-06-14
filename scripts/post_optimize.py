import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
import argparse
from evaluate import competition_score

def load_training_labels():
    train_df = pd.read_csv("data/train.csv")
    train_df = train_df.drop_duplicates()
    return train_df['flood_risk_score'].values

def objective_function(params, oof_preds, y_true):
    a, b, c = params
    
    # Ensure no negative bases for fractional powers
    safe_oof = np.clip(oof_preds, 1e-6, None)
    
    transformed_preds = a * np.power(safe_oof, b) + c
    transformed_preds = np.clip(transformed_preds, 0.0, 1.0)
    
    mae = mean_absolute_error(y_true, transformed_preds)
    rmse = root_mean_squared_error(y_true, transformed_preds)
    ev = explained_variance_score(y_true, transformed_preds)
    
    # We want to minimize the estimated LB score
    return competition_score(mae, rmse, ev)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nelder-Mead Post-Hoc Metric Optimization")
    parser.add_argument("--version", type=str, default="v20", help="Version to optimize (e.g. v20)")
    args = parser.parse_args()
    
    oof_path = f"submissions/oof_{args.version}.npy"
    sub_path = f"submissions/submission_{args.version}.csv"
    
    print(f"Loading {oof_path}...")
    try:
        oof_preds = np.load(oof_path)
    except FileNotFoundError:
        print(f"[ERROR] {oof_path} not found. Run training script for {args.version} first.")
        exit(1)
        
    y_true = load_training_labels()
    
    if len(oof_preds) != len(y_true):
        print(f"Warning: OOF predictions length ({len(oof_preds)}) does not match true labels length ({len(y_true)}).")
    
    initial_mae = mean_absolute_error(y_true, oof_preds)
    initial_rmse = root_mean_squared_error(y_true, oof_preds)
    initial_ev = explained_variance_score(y_true, oof_preds)
    initial_score = competition_score(initial_mae, initial_rmse, initial_ev)
    
    print(f"Initial LB Score: {initial_score:.5f}")
    print(f"  MAE: {initial_mae:.5f}, RMSE: {initial_rmse:.5f}, EV: {initial_ev:.5f}")
    
    # L-BFGS-B optimization with strict bounds to stay within the linear approximation region of the metric
    initial_guess = [1.0, 1.0, 0.0]  # a=1, b=1, c=0 -> no transformation
    
    # a: multiplier [0.5, 1.5]
    # b: exponent [0.5, 2.0]
    # c: offset [-0.25, 0.25]
    bounds = [(0.5, 1.5), (0.5, 2.0), (-0.25, 0.25)]
    
    print("\nRunning L-BFGS-B optimization with bounds...")
    res = minimize(
        objective_function, 
        initial_guess, 
        args=(oof_preds, y_true),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'disp': True}
    )
    
    a_opt, b_opt, c_opt = res.x
    print(f"\nOptimal parameters: a={a_opt:.5f}, b={b_opt:.5f}, c={c_opt:.5f}")
    
    opt_preds = a_opt * np.power(np.clip(oof_preds, 1e-6, None), b_opt) + c_opt
    opt_preds = np.clip(opt_preds, 0.0, 1.0)
    
    opt_mae = mean_absolute_error(y_true, opt_preds)
    opt_rmse = root_mean_squared_error(y_true, opt_preds)
    opt_ev = explained_variance_score(y_true, opt_preds)
    opt_score = competition_score(opt_mae, opt_rmse, opt_ev)
    
    print(f"Optimized LB Score: {opt_score:.5f}")
    print(f"  MAE: {opt_mae:.5f}, RMSE: {opt_rmse:.5f}, EV: {opt_ev:.5f}")
    
    improvement = initial_score - opt_score
    print(f"Improvement: {improvement:+.5f}")
    
    if improvement <= 0:
        print("\nOptimization did not improve the score. The original predictions were already optimal.")
    else:
        print(f"\nLoading {sub_path}...")
        try:
            sub_df = pd.read_csv(sub_path)
            test_preds = sub_df['flood_risk_score'].values
            
            opt_test_preds = a_opt * np.power(np.clip(test_preds, 1e-6, None), b_opt) + c_opt
            opt_test_preds = np.clip(opt_test_preds, 0.0, 1.0)
            
            sub_df['flood_risk_score'] = opt_test_preds
            out_path = f"submissions/submission_{args.version}_optimized.csv"
            sub_df.to_csv(out_path, index=False)
            print(f"Saved optimized submission to {out_path}")
        except FileNotFoundError:
            print(f"[ERROR] {sub_path} not found. Could not optimize test predictions.")
