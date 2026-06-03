import os

def build_v27(src_file, dst_file, is_kaggle=False):
    with open(src_file, "r") as f:
        content = f.read()
        
    # 1. Update Header
    content = content.replace("v26 - Adversarial Metric Alignment", "v27 - Dampened Newton Optimization")
    content = content.replace("v26 - ADVERSARIAL METRIC ALIGNMENT", "v27 - DAMPENED NEWTON OPTIMIZATION")
    content = content.replace("7. Positive constraint on Level-2 stacker via LinearRegression", "7. Unconstrained Ridge Level-2 stacker (Negative Bias Corrector)")
    content = content.replace("v26.csv", "v27.csv")
    content = content.replace("v26.npy", "v27.npy")
    
    # 2. Update XGBoost Hyperparameters
    xgb_old = """        # === 1. XGBoost with Custom Metric-Driven Objective ===
        xgb_mae = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.05, max_depth=7,
            objective=joint_mae_rmse_objective, 
            min_child_weight=30, subsample=0.8, colsample_bytree=0.75,
            tree_method="hist", early_stopping_rounds=100, random_state=seed, n_jobs=-1,
            eval_metric='mae'
        )"""
    
    xgb_new = """        # === 1. XGBoost with Custom Metric-Driven Objective ===
        xgb_mae = xgb.XGBRegressor(
            n_estimators=10000, learning_rate=0.01, max_depth=7,
            objective=joint_mae_rmse_objective, 
            min_child_weight=30, reg_lambda=10.0,
            subsample=0.8, colsample_bytree=0.75,
            tree_method="hist", early_stopping_rounds=200, random_state=seed, n_jobs=-1,
            eval_metric='mae'
        )"""
    content = content.replace(xgb_old, xgb_new)
    
    # 3. Update Stacker
    ridge_old = """    # 7. Positive constraint on stacker
    ridge_seed = LinearRegression(positive=True, fit_intercept=True)
    ridge_seed.fit(oof_meta_seed, y_arr)"""
    
    ridge_new = """    # 7. Unconstrained Ridge Level-2 stacker (Negative Bias Corrector)
    ridge_seed = Ridge(alpha=1.0, fit_intercept=True)
    ridge_seed.fit(oof_meta_seed, y_arr)"""
    content = content.replace(ridge_old, ridge_new)
    
    # Kaggle specific dataset update logic is already in v26_kaggle, so we just propagate it!
    # Write to destination
    with open(dst_file, "w") as f:
        f.write(content)

if __name__ == "__main__":
    build_v27("scripts/train_v26.py", "scripts/train_v27.py", is_kaggle=False)
    build_v27("scripts/train_v26_kaggle.py", "scripts/train_v27_kaggle.py", is_kaggle=True)
    print("Built v27 and v27_kaggle")
