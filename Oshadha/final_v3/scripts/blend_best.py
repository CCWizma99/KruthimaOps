from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'outputs_blend_clean'
SAMPLE_SUBMISSION = ROOT / 'data' / 'sample_submission.csv'

# Define the exact clean model run paths
RUNS = [
    {
        'name': 'full_clean',
        'oof_path': ROOT / 'outputs_catboost_full_clean' / 'full' / 'catboost_mae' / 'oof_predictions.csv',
        'sub_path': ROOT / 'outputs_catboost_full_clean' / 'full' / 'catboost_mae' / 'submission.csv',
    },
    {
        'name': 'safe_pseudo',
        'oof_path': ROOT / 'outputs_catboost_safe_pseudo' / 'safe' / 'catboost_mae' / 'oof_predictions.csv',
        'sub_path': ROOT / 'outputs_catboost_safe_pseudo' / 'safe' / 'catboost_mae' / 'submission.csv',
    },
    {
        'name': 'full_pseudo',
        'oof_path': ROOT / 'outputs_catboost_full_pseudo' / 'full' / 'catboost_mae' / 'oof_predictions.csv',
        'sub_path': ROOT / 'outputs_catboost_full_pseudo' / 'full' / 'catboost_mae' / 'submission.csv',
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
    for run in RUNS:
        if not run['oof_path'].exists() or not run['sub_path'].exists():
            raise FileNotFoundError(f"Missing required prediction files for {run['name']}. Run train_models.py first.")
            
        oof_df = pd.read_csv(run['oof_path'])
        sub_df = pd.read_csv(run['sub_path'])

        if not {'record_id', 'oof_pred', 'flood_risk_score'}.issubset(oof_df.columns):
            raise ValueError(f"{run['oof_path']} must contain record_id, oof_pred, and flood_risk_score")
        if not {'record_id', 'flood_risk_score'}.issubset(sub_df.columns):
            raise ValueError(f"{run['sub_path']} must contain record_id and flood_risk_score")

        oof_frames.append(oof_df[['record_id', 'oof_pred', 'flood_risk_score']].rename(columns={'oof_pred': f"pred_{run['name']}"}))
        sub_frames.append(sub_df[['record_id', 'flood_risk_score']].rename(columns={'flood_risk_score': f"pred_{run['name']}"}))

    # Merge OOFs
    merged_oof = oof_frames[0]
    for frame in oof_frames[1:]:
        merged_oof = merged_oof.merge(frame, on='record_id', how='inner')

    if len(merged_oof) == 0:
        raise ValueError('OOF merge produced empty dataframe; record_id alignment failed')

    # Merge submissions
    merged_sub = sub_frames[0]
    for frame in sub_frames[1:]:
        merged_sub = merged_sub.merge(frame, on='record_id', how='inner')

    validate_submission(
        merged_sub[['record_id', f"pred_{RUNS[0]['name']}"]].rename(columns={f"pred_{RUNS[0]['name']}": 'flood_risk_score'}),
        sample,
    )

    y_true = merged_oof['flood_risk_score'].to_numpy(dtype=float)
    
    # Construct OOF predictions matrix (n_samples, n_models)
    preds_matrix = np.column_stack([
        merged_oof[f"pred_{run['name']}"].to_numpy(dtype=float) for run in RUNS
    ])
    
    # Construct test predictions matrix (n_samples, n_models)
    test_preds_matrix = np.column_stack([
        merged_sub[f"pred_{run['name']}"].to_numpy(dtype=float) for run in RUNS
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
    bounds = [(0.0, 1.0)] * len(RUNS)
    
    best_score = np.inf
    best_weights = None
    
    # Multi-start SLSQP
    for i in range(30):
        if i == 0:
            w0 = [1.0 / len(RUNS)] * len(RUNS)
        else:
            w0 = np.random.dirichlet(np.ones(len(RUNS)))
            
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

    # Calculate single model scores for comparison
    single_scores = {}
    for i, run in enumerate(RUNS):
        m_preds = preds_matrix[:, i]
        m_mae = mean_absolute_error(y_true, m_preds)
        m_rmse = np.sqrt(mean_squared_error(y_true, m_preds))
        m_ev = 1.0 - np.var(y_true - m_preds) / np.var(y_true)
        m_score = calculate_comp_score(m_mae, m_rmse, m_ev)
        single_scores[run['name']] = m_score

    best_single_score = single_scores['full_pseudo']
    print(f"Best single clean model score ('full_pseudo'): {best_single_score:.6f}")
    print(f"Blended clean model score:                     {best_score:.6f}")

    # 3. Create submission file
    submission_df = pd.DataFrame({
        'record_id': sample['record_id'],
        'flood_risk_score': blended_test,
    })
    
    # Validate the blend submission
    validate_submission(submission_df, sample)
    
    # Save blend submission in outputs_blend_clean
    submission_df.to_csv(OUTPUT_DIR / 'submission_blend_clean_best.csv', index=False)
    
    # Save blend results details
    results_df = pd.DataFrame([{
        'w_full_clean': final_weights[0],
        'w_safe_pseudo': final_weights[1],
        'w_full_pseudo': final_weights[2],
        'mae': final_mae,
        'rmse': final_rmse,
        'ev': final_ev,
        'score': best_score
    }])
    results_df.to_csv(OUTPUT_DIR / 'blend_results.csv', index=False)

    # 4. Save to final_submission_v2.csv based on condition
    final_path = ROOT / 'final_submission_v2.csv'
    if best_score < best_single_score:
        print(f"🎉 Clean blend improved score over single best. Saving blend to final_submission_v2.csv")
        submission_df.to_csv(final_path, index=False)
    else:
        print(f"⚠️ Clean blend did not improve over single best. Copying full_pseudo submission to final_submission_v2.csv")
        shutil.copy2(RUNS[2]['sub_path'], final_path)

    # Print final validation checks
    final_sub = pd.read_csv(final_path)
    validate_submission(final_sub, sample)
    print("✅ Validation checks passed on final_submission_v2.csv successfully!")

    print('=' * 60)
    print('  CLEAN METRIC-OPTIMIZED BLENDING SUMMARY')
    print('=' * 60)
    print("Clean Blend weights:")
    for run, w in zip(RUNS, final_weights):
        print(f"  - {run['name']:<18}: {w:.4f}")
    print('-' * 60)
    print(f"Final Clean Metrics:")
    print(f"  MAE            : {final_mae:.6f}")
    print(f"  RMSE           : {final_rmse:.6f}")
    print(f"  Explained Var. : {final_ev:.6f}")
    print(f"  Score          : {best_score:.6f}")
    print(f"  Prediction Range: [{blended_oof.min():.4f}, {blended_oof.max():.4f}]")
    print(f"  Destination file: {final_path}")
    print('=' * 60)


if __name__ == '__main__':
    main()