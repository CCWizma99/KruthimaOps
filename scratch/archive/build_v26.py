import sys

with open("scripts/train_v26.py", "r") as f:
    content = f.read()

# 1. Update Header
content = content.replace("ML OPSIDIAN v25 - PURE STATISTICAL FOUNDATION", "ML OPSIDIAN v26 - ADVERSARIAL METRIC ALIGNMENT")
content = content.replace("v25 - Pure Statistical Foundation", "v26 - Adversarial Metric Alignment")
content = content.replace("v25.csv", "v26.csv")
content = content.replace("v25.npy", "v26.npy")

# 2. Add Imports
imports_str = """import numpy as np
import pandas as pd
import warnings
import time
import os
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import catboost as cb
from scipy.spatial import cKDTree
"""
# Replace the top imports
content = content.replace("""import numpy as np
import pandas as pd
import warnings
import time
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import catboost as cb""", imports_str.strip())

# 3. Strategy 1 & 2: Feature Engineering
feat_block_old = """def engineer_features(df):
    df = df.copy()"""

feat_block_new = """# --- Strategy 1: "Snap Features" Reconstruction Matrix ---
# We isolate the real records first
real_nodes = train_df[train_df['is_synthetic'].isna()][['latitude', 'longitude']].dropna().values
spatial_tree = cKDTree(real_nodes)

def engineer_features(df):
    df = df.copy()
    
    # 1. Snap Features
    coords = df[['latitude', 'longitude']].values
    distances, indices = spatial_tree.query(coords, k=1)
    
    df['snapped_lat'] = real_nodes[indices, 0]
    df['snapped_lon'] = real_nodes[indices, 1]
    
    df['lat_perturbation_noise'] = df['latitude'] - df['snapped_lat']
    df['lon_perturbation_noise'] = df['longitude'] - df['snapped_lon']
    df['spatial_perturbation_magnitude'] = distances

    # 2. Strategy 2: Multi-Scale Decimal Digit Extraction
    for col in ['inundation_area_sqm', 'latitude', 'longitude', 'ndvi_qmap', 'ndwi_qmap']:
        if col in df.columns:
            frac = np.abs(df[col] - np.floor(df[col]))
            df[f'{col}_dec_d1'] = np.floor(frac * 10)
            df[f'{col}_dec_d2'] = np.floor(frac * 100) % 10
            df[f'{col}_dec_d3'] = np.floor(frac * 1000) % 10
            df[f'{col}_is_perfect_round'] = ((frac < 0.001) | (frac > 0.999)).astype(int)
            df[f'{col}_mod_quarter'] = frac % 0.25
            df[f'{col}_mod_tenth'] = frac % 0.10
"""
content = content.replace(feat_block_old, feat_block_new)

# Drop `is_synthetic` from train_df / test_df if it's there (it's already handled in `IGNORE_COLS`)

# 4. Strategy 4: Custom Objective
custom_obj = """
def joint_mae_rmse_objective(y_true, y_pred):
    \"\"\"
    Custom objective for XGBoost. 
    Signature for XGBRegressor in Scikit-Learn API is (y_true, y_pred).
    \"\"\"
    residual = y_pred - y_true

    
    alpha = 13.2460  # MAE Weight
    beta = 4.6735    # RMSE/MSE Weight
    delta = 1e-3     # Pseudo-Huber smoothing
    
    grad_mae = alpha * (residual / np.sqrt(residual**2 + delta))
    grad_rmse = 2 * beta * residual
    gradient = grad_mae + grad_rmse
    
    hess_mae = alpha * (delta / (residual**2 + delta)**(1.5))
    hess_rmse = 2 * beta * np.ones_like(residual)
    hessian = hess_mae + hess_rmse
    
    return gradient, hessian
"""

# We'll inject the custom objective right before the MODEL CONFIGS starts
cv_loop_str = """# -----------------------------------------------------------------
# 5. MODEL CONFIGS & MULTI-SEED SETUP
# -----------------------------------------------------------------"""
content = content.replace(cv_loop_str, custom_obj + "\n" + cv_loop_str)


# Update the XGBoost definition
xgb_old = """        # === 1. XGBoost with MAE loss ===
        xgb_mae = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.05, max_depth=7,
            objective='reg:absoluteerror', 
            min_child_weight=3, subsample=0.8, colsample_bytree=0.75,
            tree_method="hist", early_stopping_rounds=100, random_state=seed, n_jobs=-1,
            eval_metric='mae'
        )"""

xgb_new = """        # === 1. XGBoost with Custom Metric-Driven Objective ===
        xgb_mae = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.05, max_depth=7,
            objective=joint_mae_rmse_objective, 
            min_child_weight=30, subsample=0.8, colsample_bytree=0.75,
            tree_method="hist", early_stopping_rounds=100, random_state=seed, n_jobs=-1,
            eval_metric='mae'
        )"""
content = content.replace(xgb_old, xgb_new)

# In MODEL_NAMES change 'XGB-MAE (d7)' to 'XGB-Custom (d7)'? 
# Let's keep the key as 'XGB-MAE (d7)' for now to not break oof_preds assignments, 
# or change all of them. Let's change the name carefully.
content = content.replace('"XGB-MAE (d7)",', '"XGB-Custom (d7)",')
content = content.replace('oof_preds["XGB-MAE (d7)"]', 'oof_preds["XGB-Custom (d7)"]')
content = content.replace('tst_preds["XGB-MAE (d7)"]', 'tst_preds["XGB-Custom (d7)"]')

with open("scripts/train_v26.py", "w") as f:
    f.write(content)
