import os
import re

with open('c:/KruthimaOps/scripts/train_v51_kaggle.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Update headers and versions
code = code.replace('v51', 'v53')
code = code.replace('v50', 'v51')

# 2. Update pseudo path
code = code.replace('submission_v42.csv', 'submission_v51_optimized.csv')
code = code.replace('submission_v30.csv', 'submission_v51.csv')

# 3. Update QUANTILE_ALPHAS
code = code.replace('QUANTILE_ALPHAS = [0.40, 0.45, 0.50, 0.55, 0.60]', 'QUANTILE_ALPHAS = [0.25, 0.75]')

# 4. Remove models from MODEL_NAMES
code = re.sub(r'\s*\"CAT-MAE-2 \(d5\)\",', '', code)
code = re.sub(r'\s*\"XGB-MAE-2 \(d5\)\",', '', code)

# 5. Change Early Stopping Rounds to 50
code = code.replace('early_stopping_rounds=100', 'early_stopping_rounds=50')
code = code.replace('early_stopping_rounds=150', 'early_stopping_rounds=50')

# 6. Remove CAT-MAE-2 model definition and fitting
cat_m2_block = """    # 3. CAT-MAE-2 (d5)
    cat_m2 = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.03, depth=5, l2_leaf_reg=12.0,
        bagging_temperature=0.4, random_strength=5.0, border_count=254,
        loss_function=\"MAE\", eval_metric=\"MAE\", task_type=CB_TASK_TYPE,
        random_seed=SEED + 1, verbose=False
    )
    cat_m2.fit(X_tr_cat, y_tr, sample_weight=w_tr, cat_features=cat_cols, eval_set=(X_va_cat, y_va), early_stopping_rounds=50, verbose=False)"""
code = code.replace(cat_m2_block, '')

# 7. Remove XGB-MAE-2 model definition and fitting
xgb_m2_block = """    # 6. XGB-MAE-2 (d5) - Unconstrained, MAE loss
    xgb_m2 = xgb.XGBRegressor(
        n_estimators=4000, learning_rate=0.03, max_depth=5, min_child_weight=6,
        subsample=0.75, colsample_bytree=0.5, colsample_bylevel=0.8,
        reg_alpha=5.0, reg_lambda=10.0, gamma=0.2, max_delta_step=1,
        objective=\"reg:absoluteerror\", eval_metric=\"mae\", tree_method=\"hist\",
        device=XGB_DEVICE,
        enable_categorical=False, early_stopping_rounds=50, random_state=SEED + 3, n_jobs=-1
    )
    xgb_m2.fit(X_tr_xgb, y_tr, sample_weight=w_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)"""
code = code.replace(xgb_m2_block, '')

# 8. Remove predictions
code = code.replace('    oof_preds["CAT-MAE-2 (d5)"][va_idx_clean] = cat_m2.predict(X_va_cat)\n', '')
code = code.replace('    oof_preds["XGB-MAE-2 (d5)"][va_idx_clean] = xgb_m2.predict(X_va_xgb)\n', '')
code = code.replace('    tst_preds["CAT-MAE-2 (d5)"] += cat_m2.predict(X_te_cat) / N_FOLDS\n', '')
code = code.replace('    tst_preds["XGB-MAE-2 (d5)"] += xgb_m2.predict(X_te_xgb) / N_FOLDS\n', '')

# 9. Clean up print statement for iterations
code = code.replace('  CAT2_it={cat_m2.best_iteration_}', '')

# 10. Update Model indexing for optimization constraint
# Original: init_guess[:len(MODEL_NAMES)] = 1.0 / len(MODEL_NAMES) -> This is dynamically scaling based on length, so it's fine!

with open('c:/KruthimaOps/scripts/train_v53_kaggle.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('Successfully generated train_v53_kaggle.py')
