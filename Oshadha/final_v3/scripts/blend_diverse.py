from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'outputs_blend_diverse'
SAMPLE_SUBMISSION = ROOT / 'data' / 'sample_submission.csv'

# Define the 5 models to blend
RUNS = [
    {
        'name': 'catboost_full_clean',
        'oof_path': ROOT / 'outputs_catboost_full_clean' / 'full' / 'catboost_mae' / 'oof_predictions.csv',
        'sub_path': ROOT / 'outputs_catboost_full_clean' / 'full' / 'catboost_mae' / 'submission.csv',
    },
    {
        'name': 'catboost_safe_pseudo',
        'oof_path': ROOT / 'outputs_catboost_safe_pseudo' / 'safe' / 'catboost_mae' / 'oof_predictions.csv',
        'sub_path': ROOT / 'outputs_catboost_safe_pseudo' / 'safe' / 'catboost_mae' / 'submission.csv',
    },
    {
        'name': 'catboost_full_pseudo',
        'oof_path': ROOT / 'outputs_catboost_full_pseudo' / 'full' / 'catboost_mae' / 'oof_predictions.csv',
        'sub_path': ROOT / 'outputs_catboost_full_pseudo' / 'full' / 'catboost_mae' / 'submission.csv',
    },
    {
        'name': 'lightgbm_full_clean',
        'oof_path': ROOT / 'outputs_lightgbm_full_clean' / 'full' / 'lightgbm_mae' / 'oof_predictions.csv',
        'sub_path': ROOT / 'outputs_lightgbm_full_clean' / 'full' / 'lightgbm_mae' / 'submission.csv',
    },
    {
        'name': 'xgboost_full_clean',
        'oof_path': ROOT / 'outputs_xgboost_full_clean' / 'full' / 'xgboost_mae' / 'oof_predictions.csv',
        'sub_path': ROOT / 'outputs_xgboost_full_clean' / 'full' / 'xgboost_mae' / 'submission.csv',
    },
]


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    expected_cols = list(sample.columns)
    if list(submission.columns) != expected_cols:
        raise ValueError(f'Unexpected submission columns: {list(submission.columns)} != {expected_cols}')
    if len(submission) != len(sample):
        raise ValueError(f'Unexpected submission length: {len(submission)} != {len(sample)}')
    if not submission['record_id'].equals(sample['record_id']):
        raise ValueError('Submission record_id order does not match sample_submission.csv')
    if submission['flood_risk_score'].isnull().sum() > 0:
        raise ValueError('Submission has missing values')
    if submission['record_id'].duplicated().sum() > 0:
        raise ValueError('Submission has duplicate record_ids')
    if submission['flood_risk_score'].min() < 0.0 or submission['flood_risk_score'].max() > 1.0:
        raise ValueError(f"Predictions out of range [0, 1]: min={submission['flood_risk_score'].min()}, max={submission['flood_risk_score'].max()}")


def calculate_comp_score(mae, rmse, ev):
    """Calculates the true undisclosed competition metric."""
    return (0.583210 * mae + 1.122681 * rmse) * (1.0 + 0.045804 * (1.0 - ev))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(SAMPLE_SUBMISSION)

    oof_frames = []
    sub_frames = []
    available_runs = []
    
    for run in RUNS:
        if run['oof_path'].exists() and run['sub_path'].exists():
            available_runs.append(run)
            oof_df = pd.read_csv(run['oof_path'])
            sub_df = pd.read_csv(run['sub_path'])
            
            # Keep target column only for the first frame to avoid merge errors on suffixes
            if len(oof_frames) == 0:
                oof_frames.append(oof_df[['record_id', 'oof_pred', 'flood_risk_score']].rename(columns={'oof_pred': f"pred_{run['name']}"}))
            else:
                oof_frames.append(oof_df[['record_id', 'oof_pred']].rename(columns={'oof_pred': f"pred_{run['name']}"}))
                
            sub_frames.append(sub_df[['record_id', 'flood_risk_score']].rename(columns={'flood_risk_score': f"pred_{run['name']}"}))
            
    if len(available_runs) == 0:
        raise RuntimeError("No prediction files found. Train models first.")

    print(f"Blending {len(available_runs)} available models:")
    for run in available_runs:
        print(f"  - {run['name']}")

    # Merge OOFs
    merged_oof = oof_frames[0]
    for frame in oof_frames[1:]:
        merged_oof = merged_oof.merge(frame, on='record_id', how='inner')

    # Merge submissions
    merged_sub = sub_frames[0]
    for frame in sub_frames[1:]:
        merged_sub = merged_sub.merge(frame, on='record_id', how='inner')

    y_true = merged_oof['flood_risk_score'].to_numpy(dtype=float)
    
    # Construct OOF predictions matrix (n_samples, n_models)
    preds_matrix = np.column_stack([
        merged_oof[f"pred_{run['name']}"].to_numpy(dtype=float) for run in available_runs
    ])
    
    # Construct test predictions matrix (n_samples, n_models)
    test_preds_matrix = np.column_stack([
        merged_sub[f"pred_{run['name']}"].to_numpy(dtype=float) for run in available_runs
    ])

    # 2. Optimize blending weights using SciPy SLSQP
    def objective(weights):
        # Normalize weights to sum to 1
        w_sum = sum(weights)
        if w_sum > 0:
            w = weights / w_sum
        else:
            w = weights
            
        blended = np.dot(preds_matrix, w)
        blended = np.clip(blended, 0.0, 1.0)
        
        mae = mean_absolute_error(y_true, blended)
        rmse = np.sqrt(mean_squared_error(y_true, blended))
        ev = 1.0 - np.var(y_true - blended) / np.var(y_true)
        
        return calculate_comp_score(mae, rmse, ev)

    # Constraints: weights sum to 1
    cons = ({'type': 'eq', 'fun': lambda w: 1.0 - sum(w)})
    # Bounds: each weight in [0, 1]
    bounds = [(0.0, 1.0)] * len(available_runs)
    
    best_score = np.inf
    best_weights = None
    
    # Multi-start SLSQP
    for i in range(40):
        if i == 0:
            w0 = [1.0 / len(available_runs)] * len(available_runs)
        else:
            w0 = np.random.dirichlet(np.ones(len(available_runs)))
            
        res = minimize(objective, w0, bounds=bounds, constraints=cons, method='SLSQP')
        if res.fun < best_score:
            best_score = res.fun
            best_weights = res.x

    final_weights = best_weights / sum(best_weights)
    
    # Generate blended predictions
    blended_oof = np.dot(preds_matrix, final_weights)
    blended_oof = np.clip(blended_oof, 0.0, 1.0)
    
    final_mae = mean_absolute_error(y_true, blended_oof)
    final_rmse = np.sqrt(mean_squared_error(y_true, blended_oof))
    final_ev = 1.0 - np.var(y_true - blended_oof) / np.var(y_true)
    
    blended_test = np.dot(test_preds_matrix, final_weights)
    blended_test = np.clip(blended_test, 0.0, 1.0)

    print(f"Current best blend score:  0.381170")
    print(f"Diverse blend score:       {best_score:.6f}")

    # 3. Create submission file
    submission_df = pd.DataFrame({
        'record_id': sample['record_id'],
        'flood_risk_score': blended_test,
    })
    
    # Validate the blend submission
    validate_submission(submission_df, sample)
    
    # Save blend submission in outputs_blend_diverse
    submission_df.to_csv(OUTPUT_DIR / 'submission_blend_diverse_best.csv', index=False)
    
    # Save blend results details
    results_dict = {
        'mae': final_mae,
        'rmse': final_rmse,
        'ev': final_ev,
        'score': best_score
    }
    for run, w in zip(available_runs, final_weights):
        results_dict[f"w_{run['name']}"] = w
        
    results_df = pd.DataFrame([results_dict])
    results_df.to_csv(OUTPUT_DIR / 'blend_results.csv', index=False)

    # 4. Save to final_submission_v3.csv based on condition
    final_path = ROOT / 'final_submission_v3.csv'
    if best_score < 0.381170:
        print(f"🎉 Diverse blend improved score over v2! Saving to final_submission_v3.csv")
        submission_df.to_csv(final_path, index=False)
    else:
        print(f"⚠️ Diverse blend did not improve over v2. final_submission_v3.csv will NOT be saved.")

    # Print final summary
    print('=' * 60)
    print('  DIVERSE METRIC-OPTIMIZED BLENDING SUMMARY')
    print('=' * 60)
    print("Blend weights:")
    for run, w in zip(available_runs, final_weights):
        print(f"  - {run['name']:<20}: {w:.4f}")
    print('-' * 60)
    print(f"Final Blended Metrics:")
    print(f"  MAE            : {final_mae:.6f}")
    print(f"  RMSE           : {final_rmse:.6f}")
    print(f"  Explained Var. : {final_ev:.6f}")
    print(f"  Score          : {best_score:.6f}")
    print(f"  Prediction Range: [{blended_oof.min():.4f}, {blended_oof.max():.4f}]")
    print('=' * 60)


if __name__ == '__main__':
    main()
