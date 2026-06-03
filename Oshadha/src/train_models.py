import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import QuantileTransformer

import features

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import xgboost as xgb
except Exception:
    xgb = None


DATA_DIR = Path('data')


def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def available_models():
    models = {}
    if CatBoostRegressor is not None:
        models['catboost'] = CatBoostRegressor
    if lgb is not None:
        models['lightgbm'] = lgb.LGBMRegressor
    if xgb is not None:
        models['xgboost'] = xgb.XGBRegressor
    return models


# (helpers above)
def run(
    model_arg: str = 'all',
    feature_set_arg: str = 'both',
    folds_n: int = 5,
    random_state: int = 42,
    output_dir: str = 'outputs',
    data_dir: str = 'data',
    target_transform: str = 'none',
):
    data_dir = Path(data_dir)
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    sample_path = data_dir / 'sample_submission.csv'

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    # mark for concatenation
    train['_is_train'] = 1
    test['_is_train'] = 0
    combined = pd.concat([train, test], sort=False).reset_index(drop=True)

    # create a coarse spatial grid id (used by v20/v21) for GroupKFold if lat/lon are present
    if 'latitude' in combined.columns and 'longitude' in combined.columns:
        lat = combined['latitude'].fillna(combined['latitude'].median())
        lon = combined['longitude'].fillna(combined['longitude'].median())
        combined['lat_bin'] = (lat / 0.5).astype(int)
        combined['lon_bin'] = (lon / 0.5).astype(int)
        combined['grid_id'] = combined['lat_bin'].astype(str) + '_' + combined['lon_bin'].astype(str)

    # prefer spatial GroupKFold on grid_id when available (matches older v20/v21 scripts)
    if 'grid_id' in combined.columns:
        folds = GroupKFold(n_splits=folds_n)
        use_group = True
    else:
        folds = KFold(n_splits=folds_n, shuffle=True, random_state=random_state)
        use_group = False

    summary_rows = []

    # if requested, prepare target transformer (fit on full training target)
    qt = None
    if target_transform == 'rankgauss':
        # fit on raw training target
        if features.TARGET_COL in train.columns:
            qt = QuantileTransformer(output_distribution='normal', random_state=random_state)
            qt.fit(train[features.TARGET_COL].values.reshape(-1, 1))
        else:
            qt = None

    models_available = available_models()
    if len(models_available) == 0:
        raise RuntimeError('No supported model libraries found (catboost, lightgbm or xgboost). Install at least one to run training.')

    # resolve requested models
    if model_arg == 'all':
        model_keys = list(models_available.keys())
    else:
        model_keys = [model_arg]

    # resolve feature sets
    if feature_set_arg == 'both':
        feature_sets = ['safe', 'full']
    else:
        feature_sets = [feature_set_arg]

    for set_name in feature_sets:
        use_safe = (set_name == 'safe')
        # Build base features first
        X_comb, num_cols, cat_cols, encs = features.build_features(combined, use_safe=use_safe, encode_for_tree=True)

        # Add smoothed target-encodings for selected categorical columns (lightweight version of v20/v21)
        target_enc_cols = [c for c in ['district', 'grid_id', 'landcover', 'soil_type'] if c in combined.columns]
        if len(target_enc_cols) > 0 and 'flood_risk_score' in train.columns:
            # compute on training rows only
            train_mask = combined['_is_train'] == 1
            global_mean = float(train[features.TARGET_COL].mean())
            smoothing = 10.0
            for col in target_enc_cols:
                stats = combined[train_mask].groupby(col)[features.TARGET_COL].agg(['mean', 'count']).rename(columns={'mean': 'mean', 'count': 'count'})
                stats['smoothed'] = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
                mapping = stats['smoothed'].to_dict()
                newcol = f"{col}_target_enc"
                X_comb[newcol] = combined[col].astype(str).map(mapping).fillna(global_mean)
                # register as numeric
                if newcol not in num_cols:
                    num_cols.append(newcol)
        cols = num_cols + cat_cols

        print(f'Running feature set: {set_name} ({len(cols)} cols)')

        train_idx = combined[combined['_is_train'] == 1].index
        test_idx = combined[combined['_is_train'] == 0].index

        X_train = X_comb.loc[train_idx, cols].reset_index(drop=True)
        X_test = X_comb.loc[test_idx, cols].reset_index(drop=True)
        y = train[features.TARGET_COL].reset_index(drop=True)
        # precompute transformed target if requested
        if qt is not None:
            y_trans_all = qt.transform(y.values.reshape(-1, 1)).flatten()
        else:
            y_trans_all = None

        # keep record ids for output alignment
        train_ids = train[features.ID_COL] if features.ID_COL in train.columns else train.index
        test_ids = test[features.ID_COL] if features.ID_COL in test.columns else test.index

        for model_key in model_keys:
            if model_key not in models_available:
                print(f' Warning: requested model {model_key} not available in this environment — skipping.')
                continue
            Model = models_available[model_key]

            out_model = out_base / set_name / model_key
            ensure_dir(out_model)

            print(f' Training {model_key} on {set_name}...')

            oof = np.zeros(len(X_train))
            test_preds = np.zeros(len(X_test))
            fold_scores = []

            # iterate folds; use group splits when available
            if use_group:
                groups = combined.loc[train_idx, 'grid_id'].reset_index(drop=True)
                split_iter = folds.split(X_train, y, groups=groups)
            else:
                split_iter = folds.split(X_train, y)

            for fold, (tr_idx, val_idx) in enumerate(split_iter):
                X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

                if model_key == 'catboost':
                    cat_feat_idx = [i for i, c in enumerate(X_train.columns) if c in cat_cols]
                    # align CatBoost params with v20/v21 (more iterations, l2 regularization, depth=5)
                    m = Model(iterations=5000, depth=5, learning_rate=0.03, l2_leaf_reg=5, random_seed=random_state, verbose=0)
                    # use early stopping on validation
                    # select target (transformed or original)
                    if y_trans_all is not None:
                        y_tr_fit = y_trans_all[tr_idx]
                        y_val_fit = y_trans_all[val_idx]
                    else:
                        y_tr_fit = y_tr
                        y_val_fit = y_val

                    m.fit(X_tr, y_tr_fit, eval_set=(X_val, y_val_fit), early_stopping_rounds=100, cat_features=cat_feat_idx)
                    # predictions are on transformed scale if using transformer
                    val_pred_raw = m.predict(X_val)
                    test_fold_pred_raw = m.predict(X_test)
                    if qt is not None:
                        # inverse-transform back to original scale
                        val_pred = qt.inverse_transform(val_pred_raw.reshape(-1, 1)).flatten()
                        test_fold_pred = qt.inverse_transform(test_fold_pred_raw.reshape(-1, 1)).flatten()
                    else:
                        val_pred = val_pred_raw
                        test_fold_pred = test_fold_pred_raw
                    importances = m.get_feature_importance()

                elif model_key == 'lightgbm':
                    m = Model(n_estimators=2000, learning_rate=0.05, random_state=random_state)
                    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, categorical_feature=cat_cols, verbose=False)
                    val_pred = m.predict(X_val)
                    test_fold_pred = m.predict(X_test)
                    importances = m.feature_importances_

                else:  # xgboost
                    m = Model(n_estimators=2000, learning_rate=0.05, random_state=random_state, tree_method='hist')
                    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
                    val_pred = m.predict(X_val)
                    test_fold_pred = m.predict(X_test)
                    try:
                        importances = m.feature_importances_
                    except Exception:
                        importances = np.zeros(X_train.shape[1])

                oof[val_idx] = val_pred
                test_preds += test_fold_pred / folds.n_splits

                fold_rmse = rmse(y_val, val_pred)
                fold_mae = mean_absolute_error(y_val, val_pred)
                fold_scores.append({'fold': fold, 'rmse': float(fold_rmse), 'mae': float(fold_mae)})
                print(f'  fold {fold} rmse={fold_rmse:.5f} mae={fold_mae:.5f}')

            # clip predictions to [0,1]
            oof = np.clip(oof, 0.0, 1.0)
            test_preds = np.clip(test_preds, 0.0, 1.0)

            # save oof
            oof_df = pd.DataFrame({features.ID_COL: train_ids.reset_index(drop=True), 'oof_pred': oof, features.TARGET_COL: y.reset_index(drop=True)})
            oof_df.to_csv(out_model / 'oof_predictions.csv', index=False)

            # save submission
            sub = pd.DataFrame({features.ID_COL: test_ids.reset_index(drop=True), features.TARGET_COL: test_preds})
            sub.to_csv(out_model / 'submission.csv', index=False)

            # save fold scores
            pd.DataFrame(fold_scores).to_csv(out_model / 'fold_scores.csv', index=False)

            # save feature importance
            fi = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
            fi = fi.sort_values('importance', ascending=False)
            fi.to_csv(out_model / 'feature_importance.csv', index=False)

            # record summary (mean across folds)
            mean_rmse = float(np.mean([r['rmse'] for r in fold_scores]))
            mean_mae = float(np.mean([r['mae'] for r in fold_scores]))
            summary_rows.append({'feature_set': set_name, 'model': model_key, 'rmse': mean_rmse, 'mae': mean_mae})

    # write overall summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_base / 'summary_scores.csv', index=False)

    # basic human-readable summary
    if len(summary_df) > 0:
        best = summary_df.sort_values('rmse').iloc[0]
        with open(out_base / 'summary.md', 'w') as f:
            f.write('# CV Summary\n\n')
            try:
                f.write(summary_df.to_markdown(index=False))
            except Exception:
                f.write(summary_df.to_string(index=False))
            f.write('\n\n')
            f.write(f"Best model: {best['model']} with feature set {best['feature_set']} (rmse={best['rmse']:.5f})\n")

    print('Done. Outputs written to', out_base)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['catboost', 'lightgbm', 'xgboost', 'all'], default='all')
    p.add_argument('--feature-set', dest='feature_set', choices=['safe', 'full', 'both'], default='both')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--output-dir', dest='output_dir', default='outputs')
    p.add_argument('--data-dir', dest='data_dir', default='data')
    p.add_argument('--target-transform', dest='target_transform', choices=['none', 'rankgauss'], default='none')
    args = p.parse_args()

    run(
        model_arg=args.model,
        feature_set_arg=args.feature_set,
        folds_n=args.folds,
        random_state=args.random_state,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        target_transform=args.target_transform,
    )
