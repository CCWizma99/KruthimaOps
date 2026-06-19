import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, explained_variance_score
import sys
import os
sys.path.append('c:/KruthimaOps/production')

from app.inference import v1000_engine

# Need to set os environment for config maybe? Let's just import
v1000_engine.load_artifacts()

print("Loading data...")
df = pd.read_csv('c:/KruthimaOps/data/train_v1002_desinventar.csv')

# The 1500 historical rows are the ones where record_id is missing
hist_df = df[df['record_id'].isna()].copy()
print(f"Found {len(hist_df)} historical records for verification.")

# The physically calculated target is in 'flood_risk_score'
y_true = hist_df['flood_risk_score'].values

y_pred = []
print("Running Inference on Historical Data...")
for i, row in hist_df.iterrows():
    pred, var = v1000_engine.infer(row.to_dict())
    y_pred.append(pred)

y_pred = np.array(y_pred)

mae = mean_absolute_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
ev = explained_variance_score(y_true, y_pred)

print("\n" + "="*50)
print("  HISTORICAL VERIFICATION RESULTS")
print("="*50)
print(f"Target Distribution: Mean={y_true.mean():.4f}, Std={y_true.std():.4f}")
print(f"Pred   Distribution: Mean={y_pred.mean():.4f}, Std={y_pred.std():.4f}")
print("-" * 50)
print(f"MAE  : {mae:.5f}")
print(f"RMSE : {rmse:.5f}")
print(f"EV   : {ev:.5f}")
print("="*50)
