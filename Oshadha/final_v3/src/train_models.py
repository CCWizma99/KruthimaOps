import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import QuantileTransformer

import features

try:
    from catboost import CatBoostRegressor, Pool
except Exception:
    CatBoostRegressor = None
    Pool = None

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


def calculate_comp_score(mae, rmse, ev):
    """Calculates the true undisclosed competition metric."""
    return (0.583210 * mae + 1.122681 * rmse) * (1.0 + 0.045804 * (1.0 - ev))


def compute_target_encodings(tr_df, val_df, te_df, target_col, cols_to_encode, smoothing=10.0):
    """Computes fold-isolated Bayesian-smoothed target encoding statistics."""
    tr_df = tr_df.copy()
    val_df = val_df.copy()
    te_df = te_df.copy()
    
    global_mean = float(tr_df[target_col].mean())
    global_std = float(tr_df[target_col].std()) if len(tr_df) > 1 else 0.0
    global_med = float(tr_df[target_col].median())
    global_q25 = float(tr_df[target_col].quantile(0.25))
    global_q75 = float(tr_df[target_col].quantile(0.75))
    
    for col in cols_to_encode:
        if col not in tr_df.columns:
            continue
            
        stats = tr_df.groupby(col)[target_col].agg(
            mean='mean',
            std='std',
            median='median',
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            count='count'
        )
        stats['std'] = stats['std'].fillna(0.0)
        
        # Smoothed stats
        smoothed_mean = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        smoothed_std = (stats['count'] * stats['std'] + smoothing * global_std) / (stats['count'] + smoothing)
        smoothed_med = (stats['count'] * stats['median'] + smoothing * global_med) / (stats['count'] + smoothing)
        smoothed_q25 = (stats['count'] * stats['q25'] + smoothing * global_q25) / (stats['count'] + smoothing)
        smoothed_q75 = (stats['count'] * stats['q75'] + smoothing * global_q75) / (stats['count'] + smoothing)
        log_count = np.log1p(stats['count'])
        
        # Create mappings
        mean_map = smoothed_mean.to_dict()
        std_map = smoothed_std.to_dict()
        med_map = smoothed_med.to_dict()
        q25_map = smoothed_q25.to_dict()
        q75_map = smoothed_q75.to_dict()
        cnt_map = log_count.to_dict()
        
        # Apply mappings
        for df in [tr_df, val_df, te_df]:
            col_str = df[col].astype(str)
            df[f"{col}_target_enc"] = col_str.map(mean_map).fillna(global_mean).astype(float)
            df[f"{col}_target_std"] = col_str.map(std_map).fillna(global_std).astype(float)
            df[f"{col}_target_med"] = col_str.map(med_map).fillna(global_med).astype(float)
            df[f"{col}_target_q25"] = col_str.map(q25_map).fillna(global_q25).astype(float)
            df[f"{col}_target_q75"] = col_str.map(q75_map).fillna(global_q75).astype(float)
            df[f"{col}_target_cnt"] = col_str.map(cnt_map).fillna(0.0).astype(float)
            
    return tr_df, val_df, te_df


def train_pipeline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_keys: list,
    set_name: str,
    folds_n: int,
    seeds: list,
    pseudo_labels: np.ndarray = None,
    pseudo_weight: float = 0.2,
    output_dir: str = 'outputs',
    use_safe: bool = True
):
    out_base = Path(output_dir)
    
    # 1. Build features (WITHOUT encoding to preserve categories for fold target encoding)
    X_comb, num_cols, cat_cols, encs = features.build_features(
        pd.concat([train, test], sort=False).reset_index(drop=True), 
        use_safe=use_safe, 
        encode_for_tree=False
    )
    
    # Identify training and test row indexes
    train_idx = train.index
    test_idx = np.arange(len(train), len(train) + len(test))
    
    X_train_base = X_comb.iloc[train_idx].reset_index(drop=True)
    X_test_base = X_comb.iloc[test_idx].reset_index(drop=True)
    y = train[features.TARGET_COL].reset_index(drop=True)
    
    train_ids = train[features.ID_COL] if features.ID_COL in train.columns else train.index
    test_ids = test[features.ID_COL] if features.ID_COL in test.columns else test.index
    
    # Coarse spatial grid id for GroupKFold if present
    if 'latitude' in train.columns and 'longitude' in train.columns:
        lat = train['latitude'].fillna(train['latitude'].median())
        lon = train['longitude'].fillna(train['longitude'].median())
        train_grid_id = (lat / 0.5).astype(int).astype(str) + '_' + (lon / 0.5).astype(int).astype(str)
        folds = GroupKFold(n_splits=folds_n)
        split_iter = folds.split(X_train_base, y, groups=train_grid_id)
    else:
        folds = KFold(n_splits=folds_n, shuffle=True, random_state=42)
        split_iter = folds.split(X_train_base, y)
        
    fold_splits = list(split_iter)
    
    # Categorical columns to encode
    target_enc_cols = [c for c in [
        'district', 'landcover', 'soil_type', 'water_supply',
        'electricity', 'road_quality', 'urban_rural',
        'water_presence_flag', 'flood_occurrence_current_event',
        'is_good_to_live', 'reason_not_good_to_live', 'grid_id'
    ] if c in X_train_base.columns]
    
    # Build list of new target-encoded feature column names
    te_features = []
    for col in target_enc_cols:
        te_features.extend([
            f"{col}_target_enc", f"{col}_target_std", f"{col}_target_med",
            f"{col}_target_q25", f"{col}_target_q75", f"{col}_target_cnt"
        ])
        
    # All features for model training (base numeric + base categorical + target encodings)
    FEATURES = num_cols + cat_cols + te_features
    
    # Model configurations
    MODEL_CONFIGS = {
        'catboost_mae': {'library': 'catboost', 'loss': 'MAE', 'transform': 'none'},
        'catboost_rmse': {'library': 'catboost', 'loss': 'RMSE', 'transform': 'none'},
        'catboost_rankgauss': {'library': 'catboost', 'loss': 'MAE', 'transform': 'rankgauss'},
        'xgboost_mae': {'library': 'xgboost', 'loss': 'reg:absoluteerror', 'transform': 'none'},
        'xgboost_rankgauss': {'library': 'xgboost', 'loss': 'reg:absoluteerror', 'transform': 'rankgauss'},
        'lightgbm_mae': {'library': 'lightgbm', 'loss': 'regression_l1', 'transform': 'none'}
    }
    
    # Results containers
    final_oof_preds = {}
    final_test_preds = {}
    model_summaries = []
    
    for model_key in model_keys:
        if model_key not in MODEL_CONFIGS:
            print(f"Skipping unknown model key: {model_key}")
            continue
            
        config = MODEL_CONFIGS[model_key]
        out_model = out_base / set_name / model_key
        ensure_dir(out_model)
        
        print(f"--- Training {model_key} on {set_name} (Pseudo-labels: {pseudo_labels is not None}) ---")
        
        # Accumulators across seeds
        oof_accum = np.zeros(len(X_train_base))
        test_accum = np.zeros(len(X_test_base))
        fold_scores = []
        
        for fold, (tr_idx, val_idx) in enumerate(fold_splits):
            tr_orig_rows = X_train_base.iloc[tr_idx].copy()
            val_orig_rows = X_train_base.iloc[val_idx].copy()
            test_rows = X_test_base.copy()
            
            y_tr_orig = y.iloc[tr_idx]
            y_val_orig = y.iloc[val_idx]
            
            # Add targets to calculate encoding stats safely
            tr_orig_rows[features.TARGET_COL] = y_tr_orig
            
            # Compute target encodings on training fold, map to training, validation, and test
            tr_df_encoded, val_df_encoded, test_df_encoded = compute_target_encodings(
                tr_orig_rows, val_orig_rows, test_rows,
                features.TARGET_COL, target_enc_cols, smoothing=10.0
            )
            
            # Construct final training and validation features/targets
            if pseudo_labels is not None:
                # Append pseudo-labeled test rows to training fold
                X_tr_fit = pd.concat([tr_df_encoded[FEATURES], test_df_encoded[FEATURES]], ignore_index=True)
                y_tr_fit_orig = pd.concat([y_tr_orig, pd.Series(pseudo_labels)], ignore_index=True)
                w_tr_fit = np.concatenate([np.ones(len(tr_orig_rows)), np.full(len(test_rows), pseudo_weight)])
            else:
                X_tr_fit = tr_df_encoded[FEATURES].copy()
                y_tr_fit_orig = y_tr_orig
                w_tr_fit = np.ones(len(tr_orig_rows))
                
            X_val_fit = val_df_encoded[FEATURES].copy()
            y_val_fit_orig = y_val_orig
            
            # Predict containers for this fold across seeds
            fold_val_preds = np.zeros(len(val_idx))
            fold_test_preds = np.zeros(len(X_test_base))
            
            for seed in seeds:
                if config['library'] == 'catboost':
                    X_tr_cb = X_tr_fit.copy()
                    X_val_cb = X_val_fit.copy()
                    X_te_cb = test_df_encoded[FEATURES].copy()
                    for c in cat_cols:
                        X_tr_cb[c] = X_tr_cb[c].astype(str)
                        X_val_cb[c] = X_val_cb[c].astype(str)
                        X_te_cb[c] = X_te_cb[c].astype(str)
                        
                    if config['transform'] == 'rankgauss':
                        qt = QuantileTransformer(output_distribution='normal', random_state=seed)
                        qt.fit(y_tr_orig.values.reshape(-1, 1))
                        y_tr_fit = qt.transform(y_tr_fit_orig.values.reshape(-1, 1)).flatten()
                        y_val_fit = qt.transform(y_val_fit_orig.values.reshape(-1, 1)).flatten()
                    else:
                        qt = None
                        y_tr_fit = y_tr_fit_orig.values
                        y_val_fit = y_val_fit_orig.values
                        
                    m = CatBoostRegressor(
                        iterations=3000,
                        learning_rate=0.03,
                        depth=5,
                        l2_leaf_reg=5,
                        loss_function=config['loss'],
                        eval_metric=config['loss'],
                        random_seed=seed,
                        verbose=0
                    )
                    
                    train_pool = Pool(X_tr_cb, y_tr_fit, cat_features=cat_cols, weight=w_tr_fit)
                    val_pool = Pool(X_val_cb, y_val_fit, cat_features=cat_cols)
                    m.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=0)
                    
                    val_pred_raw = m.predict(X_val_cb)
                    test_pred_raw = m.predict(X_te_cb)
                    if qt is not None:
                        val_pred = qt.inverse_transform(val_pred_raw.reshape(-1, 1)).flatten()
                        test_pred = qt.inverse_transform(test_pred_raw.reshape(-1, 1)).flatten()
                    else:
                        val_pred = val_pred_raw
                        test_pred = test_pred_raw
                        
                elif config['library'] == 'xgboost':
                    X_tr_xgb = X_tr_fit.copy()
                    X_val_xgb = X_val_fit.copy()
                    X_te_xgb = test_df_encoded[FEATURES].copy()
                    for c in cat_cols:
                        X_tr_xgb[c] = X_tr_xgb[c].astype('category')
                        X_val_xgb[c] = X_val_xgb[c].astype('category')
                        X_te_xgb[c] = X_te_xgb[c].astype('category')
                        
                    if config['transform'] == 'rankgauss':
                        qt = QuantileTransformer(output_distribution='normal', random_state=seed)
                        qt.fit(y_tr_orig.values.reshape(-1, 1))
                        y_tr_fit = qt.transform(y_tr_fit_orig.values.reshape(-1, 1)).flatten()
                        y_val_fit = qt.transform(y_val_fit_orig.values.reshape(-1, 1)).flatten()
                    else:
                        qt = None
                        y_tr_fit = y_tr_fit_orig.values
                        y_val_fit = y_val_fit_orig.values
                        
                    m = xgb.XGBRegressor(
                        n_estimators=3000,
                        learning_rate=0.05,
                        max_depth=7,
                        objective=config['loss'],
                        min_child_weight=3,
                        subsample=0.8,
                        colsample_bytree=0.75,
                        tree_method='hist',
                        early_stopping_rounds=50,
                        enable_categorical=True,
                        random_state=seed,
                        n_jobs=-1
                    )
                    m.fit(
                        X_tr_xgb, y_tr_fit,
                        sample_weight=w_tr_fit,
                        eval_set=[(X_val_xgb, y_val_fit)],
                        verbose=False
                    )
                    
                    val_pred_raw = m.predict(X_val_xgb)
                    test_pred_raw = m.predict(X_te_xgb)
                    if qt is not None:
                        val_pred = qt.inverse_transform(val_pred_raw.reshape(-1, 1)).flatten()
                        test_pred = qt.inverse_transform(test_pred_raw.reshape(-1, 1)).flatten()
                    else:
                        val_pred = val_pred_raw
                        test_pred = test_pred_raw
                        
                elif config['library'] == 'lightgbm':
                    X_tr_lgb = X_tr_fit.copy()
                    X_val_lgb = X_val_fit.copy()
                    X_te_lgb = test_df_encoded[FEATURES].copy()
                    for c in cat_cols:
                        X_tr_lgb[c] = X_tr_lgb[c].astype('category')
                        X_val_lgb[c] = X_val_lgb[c].astype('category')
                        X_te_lgb[c] = X_te_lgb[c].astype('category')
                        
                    if config['transform'] == 'rankgauss':
                        qt = QuantileTransformer(output_distribution='normal', random_state=seed)
                        qt.fit(y_tr_orig.values.reshape(-1, 1))
                        y_tr_fit = qt.transform(y_tr_fit_orig.values.reshape(-1, 1)).flatten()
                        y_val_fit = qt.transform(y_val_fit_orig.values.reshape(-1, 1)).flatten()
                    else:
                        qt = None
                        y_tr_fit = y_tr_fit_orig.values
                        y_val_fit = y_val_fit_orig.values
                        
                    m = lgb.LGBMRegressor(
                        n_estimators=3000,
                        learning_rate=0.05,
                        max_depth=6,
                        num_leaves=31,
                        objective=config['loss'],
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=seed,
                        n_jobs=-1,
                        verbose=-1,
                        early_stopping_rounds=50
                    )
                    m.fit(
                        X_tr_lgb, y_tr_fit,
                        sample_weight=w_tr_fit,
                        eval_set=[(X_val_lgb, y_val_fit)],
                        categorical_feature=cat_cols
                    )
                    
                    val_pred_raw = m.predict(X_val_lgb)
                    test_pred_raw = m.predict(X_te_lgb)
                    if qt is not None:
                        val_pred = qt.inverse_transform(val_pred_raw.reshape(-1, 1)).flatten()
                        test_pred = qt.inverse_transform(test_pred_raw.reshape(-1, 1)).flatten()
                    else:
                        val_pred = val_pred_raw
                        test_pred = test_pred_raw
                        
                fold_val_preds += val_pred / len(seeds)
                fold_test_preds += test_pred / len(seeds)
                
            # Accumulate across folds
            oof_accum[val_idx] = fold_val_preds
            test_accum += fold_test_preds / folds_n
            
            fold_rmse = rmse(y_val_orig, fold_val_preds)
            fold_mae = mean_absolute_error(y_val_orig, fold_val_preds)
            fold_ev = explained_variance_score(y_val_orig, fold_val_preds)
            fold_scores.append({
                'fold': fold, 
                'rmse': float(fold_rmse), 
                'mae': float(fold_mae),
                'ev': float(fold_ev)
            })
            print(f"  Fold {fold} RMSE={fold_rmse:.5f} MAE={fold_mae:.5f} EV={fold_ev:.5f}")
            
        # Post-process predictions
        oof_accum = np.clip(oof_accum, 0.0, 1.0)
        test_accum = np.clip(test_accum, 0.0, 1.0)
        
        # Calculate global OOF scores
        overall_mae = mean_absolute_error(y, oof_accum)
        overall_rmse = rmse(y, oof_accum)
        overall_ev = explained_variance_score(y, oof_accum)
        overall_comp = calculate_comp_score(overall_mae, overall_rmse, overall_ev)
        
        print(f"  --> OVERALL: MAE={overall_mae:.5f} RMSE={overall_rmse:.5f} EV={overall_ev:.5f} Score={overall_comp:.5f}")
        
        # Save OOF & Submission
        oof_df = pd.DataFrame({
            features.ID_COL: train_ids.reset_index(drop=True),
            'oof_pred': oof_accum,
            features.TARGET_COL: y.reset_index(drop=True)
        })
        oof_df.to_csv(out_model / 'oof_predictions.csv', index=False)
        
        sub = pd.DataFrame({
            features.ID_COL: test_ids.reset_index(drop=True),
            features.TARGET_COL: test_accum
        })
        sub.to_csv(out_model / 'submission.csv', index=False)
        
        # Save fold scores
        pd.DataFrame(fold_scores).to_csv(out_model / 'fold_scores.csv', index=False)
        
        final_oof_preds[model_key] = oof_accum
        final_test_preds[model_key] = test_accum
        
        model_summaries.append({
            'feature_set': set_name,
            'model': model_key,
            'mae': overall_mae,
            'rmse': overall_rmse,
            'ev': overall_ev,
            'score': overall_comp
        })
        
    return final_oof_preds, final_test_preds, model_summaries


def run(
    model_arg: str = 'all',
    feature_set_arg: str = 'both',
    folds_n: int = 5,
    random_states: list = [42],
    output_dir: str = 'outputs',
    data_dir: str = 'data',
    use_pseudo: bool = True,
    pseudo_weight: float = 0.2
):
    data_dir = Path(data_dir)
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    
    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')
    
    # Model mapping list
    all_models = [
        'catboost_mae', 'catboost_rmse', 'catboost_rankgauss',
        'xgboost_mae', 'xgboost_rankgauss', 'lightgbm_mae'
    ]
    if model_arg == 'all':
        model_keys = all_models
    else:
        model_keys = [model_arg]
        
    # Feature sets list
    if feature_set_arg == 'both':
        feature_sets = ['safe', 'full']
    else:
        feature_sets = [feature_set_arg]
        
    summary_rows = []
    
    for set_name in feature_sets:
        use_safe = (set_name == 'safe')
        print(f"\n=======================================================")
        # STAGE 1: Standard Training
        print(f"STAGE 1: Training base models on {set_name} features")
        print(f"=======================================================")
        oof_s1, test_s1, summaries_s1 = train_pipeline(
            train, test, model_keys, set_name, folds_n, 
            random_states, pseudo_labels=None, 
            output_dir=output_dir, use_safe=use_safe
        )
        
        if use_pseudo and len(test_s1) > 0:
            print(f"\n=======================================================")
            # STAGE 2: Soft Pseudo-Labeling Training
            print(f"STAGE 2: Training pseudo-labeled models on {set_name} features")
            print(f"=======================================================")
            # Simple average of Stage 1 test predictions as pseudo-labels
            blended_test_s1 = np.mean(list(test_s1.values()), axis=0)
            
            oof_s2, test_s2, summaries_s2 = train_pipeline(
                train, test, model_keys, set_name, folds_n,
                random_states, pseudo_labels=blended_test_s1,
                pseudo_weight=pseudo_weight,
                output_dir=output_dir, use_safe=use_safe
            )
            summary_rows.extend(summaries_s2)
        else:
            summary_rows.extend(summaries_s1)
            
    # Save overall summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_base / 'summary_scores.csv', index=False)
    
    # Write summary markdown report
    if len(summary_df) > 0:
        best = summary_df.sort_values('score').iloc[0]
        with open(out_base / 'summary.md', 'w') as f:
            f.write('# Out-Of-Fold Cross-Validation Summary\n\n')
            f.write('The end-to-end regression pipeline has been trained using fold-isolated target encodings, multi-seed averaging, and pseudo-labeling. Below are the out-of-fold metrics across configurations.\n\n')
            try:
                f.write(summary_df.to_markdown(index=False))
            except Exception:
                f.write(summary_df.to_string(index=False))
            f.write('\n\n')
            f.write(f"**Best Configuration:** {best['model']} with feature set {best['feature_set']} (Est. LB Score={best['score']:.6f}, MAE={best['mae']:.5f}, RMSE={best['rmse']:.5f}, EV={best['ev']:.5f})\n")
            
    print(f"\nDone. Summary scores saved to {out_base / 'summary_scores.csv'}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='all', choices=[
        'catboost_mae', 'catboost_rmse', 'catboost_rankgauss',
        'xgboost_mae', 'xgboost_rankgauss', 'lightgbm_mae', 'all'
    ])
    p.add_argument('--feature-set', dest='feature_set', choices=['safe', 'full', 'both'], default='both')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--output-dir', dest='output_dir', default='outputs')
    p.add_argument('--data-dir', dest='data_dir', default='data')
    p.add_argument('--no-pseudo', dest='use_pseudo', action='store_false')
    p.add_argument('--pseudo-weight', dest='pseudo_weight', type=float, default=0.2)
    args = p.parse_args()
    
    run(
        model_arg=args.model,
        feature_set_arg=args.feature_set,
        folds_n=args.folds,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        use_pseudo=args.use_pseudo,
        pseudo_weight=args.pseudo_weight
    )
