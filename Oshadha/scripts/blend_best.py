from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'outputs_blend'
SAMPLE_SUBMISSION = ROOT / 'data' / 'sample_submission.csv'

RUNS = [
    {
        'name': 'v2',
        'oof_path': ROOT / 'outputs_catboost_full_v2' / 'full' / 'catboost' / 'oof_predictions.csv',
        'sub_path': ROOT / 'outputs_catboost_full_v2' / 'full' / 'catboost' / 'submission.csv',
    },
    {
        'name': 'v3',
        'oof_path': ROOT / 'outputs_catboost_full_v3' / 'full' / 'catboost' / 'oof_predictions.csv',
        'sub_path': ROOT / 'outputs_catboost_full_v3' / 'full' / 'catboost' / 'submission.csv',
    },
    {
        'name': 'rankgauss',
        'oof_path': ROOT / 'outputs_catboost_full_rankgauss' / 'full' / 'catboost' / 'oof_predictions.csv',
        'sub_path': ROOT / 'outputs_catboost_full_rankgauss' / 'full' / 'catboost' / 'submission.csv',
    },
]


def load_predictions(path: Path, pred_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'record_id' not in df.columns:
        raise ValueError(f'Missing record_id in {path}')
    if pred_col not in df.columns:
        raise ValueError(f'Missing {pred_col} in {path}')
    return df[['record_id', pred_col]].copy()


def grid_weights(step: float = 0.05):
    values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)
    for w1, w2 in product(values, repeat=2):
        w3 = np.round(1.0 - w1 - w2, 10)
        if w3 < -1e-9:
            continue
        if abs(w3 / step - round(w3 / step)) > 1e-9:
            continue
        if w3 < 0:
            w3 = 0.0
        yield float(w1), float(w2), float(w3)


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    expected_cols = list(sample.columns)
    if list(submission.columns) != expected_cols:
        raise ValueError(f'Unexpected submission columns: {list(submission.columns)} != {expected_cols}')
    if len(submission) != len(sample):
        raise ValueError(f'Unexpected submission length: {len(submission)} != {len(sample)}')
    if not submission['record_id'].equals(sample['record_id']):
        raise ValueError('Submission record_id order does not match sample_submission.csv')


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(SAMPLE_SUBMISSION)
    if 'record_id' not in sample.columns or 'flood_risk_score' not in sample.columns:
        raise ValueError('sample_submission.csv must contain record_id and flood_risk_score')

    oof_frames = []
    sub_frames = []
    for run in RUNS:
        oof_df = pd.read_csv(run['oof_path'])
        sub_df = pd.read_csv(run['sub_path'])

        if not {'record_id', 'oof_pred', 'flood_risk_score'}.issubset(oof_df.columns):
            raise ValueError(f"{run['oof_path']} must contain record_id, oof_pred, and flood_risk_score")
        if not {'record_id', 'flood_risk_score'}.issubset(sub_df.columns):
            raise ValueError(f"{run['sub_path']} must contain record_id and flood_risk_score")

        oof_frames.append(oof_df[['record_id', 'oof_pred', 'flood_risk_score']].rename(columns={'oof_pred': f"pred_{run['name']}"}))
        sub_frames.append(sub_df[['record_id', 'flood_risk_score']].rename(columns={'flood_risk_score': f"pred_{run['name']}"}))

    merged_oof = oof_frames[0]
    for frame in oof_frames[1:]:
        merged_oof = merged_oof.merge(frame, on='record_id', how='inner')

    if len(merged_oof) == 0:
        raise ValueError('OOF merge produced empty dataframe; record_id alignment failed')

    merged_sub = sub_frames[0]
    for frame in sub_frames[1:]:
        merged_sub = merged_sub.merge(frame, on='record_id', how='inner')

    validate_submission(
        merged_sub[['record_id', 'pred_v2']].rename(columns={'pred_v2': 'flood_risk_score'}),
        sample,
    )

    y_true = merged_oof['flood_risk_score'].to_numpy(dtype=float)
    preds = {
        'v2': merged_oof['pred_v2'].to_numpy(dtype=float),
        'v3': merged_oof['pred_v3'].to_numpy(dtype=float),
        'rankgauss': merged_oof['pred_rankgauss'].to_numpy(dtype=float),
    }
    test_preds = {
        'v2': merged_sub['pred_v2'].to_numpy(dtype=float),
        'v3': merged_sub['pred_v3'].to_numpy(dtype=float),
        'rankgauss': merged_sub['pred_rankgauss'].to_numpy(dtype=float),
    }

    results = []
    for w_v2, w_v3, w_rg in grid_weights(0.05):
        blended_oof = w_v2 * preds['v2'] + w_v3 * preds['v3'] + w_rg * preds['rankgauss']
        blended_oof = np.clip(blended_oof, 0.0, 1.0)

        mae = mean_absolute_error(y_true, blended_oof)
        rmse = np.sqrt(mean_squared_error(y_true, blended_oof))
        ev = 1.0 - np.var(y_true - blended_oof) / np.var(y_true)
        balanced = (rmse + (1.0 - ev)) / 2.0

        results.append({
            'w_v2': w_v2,
            'w_v3': w_v3,
            'w_rankgauss': w_rg,
            'mae': mae,
            'rmse': rmse,
            'ev': ev,
            'balanced_score': balanced,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['mae', 'rmse', 'ev'], ascending=[True, True, False]).reset_index(drop=True)
    results_df.head(20).to_csv(OUTPUT_DIR / 'blend_results.csv', index=False)

    best_mae = results_df.iloc[0]
    best_balanced = results_df.sort_values(['balanced_score', 'mae', 'rmse'], ascending=[True, True, True]).iloc[0]

    best_mae_preds = np.clip(
        best_mae['w_v2'] * test_preds['v2'] + best_mae['w_v3'] * test_preds['v3'] + best_mae['w_rankgauss'] * test_preds['rankgauss'],
        0.0,
        1.0,
    )
    balanced_preds = np.clip(
        best_balanced['w_v2'] * test_preds['v2'] + best_balanced['w_v3'] * test_preds['v3'] + best_balanced['w_rankgauss'] * test_preds['rankgauss'],
        0.0,
        1.0,
    )

    best_mae_submission = pd.DataFrame({
        'record_id': sample['record_id'],
        'flood_risk_score': best_mae_preds,
    })
    balanced_submission = pd.DataFrame({
        'record_id': sample['record_id'],
        'flood_risk_score': balanced_preds,
    })

    validate_submission(best_mae_submission, sample)
    validate_submission(balanced_submission, sample)

    best_mae_submission.to_csv(OUTPUT_DIR / 'submission_blend_best_mae.csv', index=False)
    balanced_submission.to_csv(OUTPUT_DIR / 'submission_blend_balanced.csv', index=False)

    print('Top 20 blends written to', OUTPUT_DIR / 'blend_results.csv')
    print('Best MAE blend:')
    print(
        f"  weights -> v2={best_mae['w_v2']:.2f}, v3={best_mae['w_v3']:.2f}, rankgauss={best_mae['w_rankgauss']:.2f}"
    )
    print(
        f"  metrics -> MAE={best_mae['mae']:.6f}, RMSE={best_mae['rmse']:.6f}, EV={best_mae['ev']:.6f}"
    )
    print('Best balanced blend:')
    print(
        f"  weights -> v2={best_balanced['w_v2']:.2f}, v3={best_balanced['w_v3']:.2f}, rankgauss={best_balanced['w_rankgauss']:.2f}"
    )
    print(
        f"  metrics -> MAE={best_balanced['mae']:.6f}, RMSE={best_balanced['rmse']:.6f}, EV={best_balanced['ev']:.6f}"
    )


if __name__ == '__main__':
    main()