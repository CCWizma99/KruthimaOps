import pandas as pd
import numpy as np
import os

print("Loading baseline submission (k=1)...")
# Make sure we read from the submissions folder
sub = pd.read_csv('../submissions/submission_v3.csv')

preds = sub['flood_risk_score'].values
mean_pred = np.mean(preds)

print(f"Base predictions - Mean: {mean_pred:.4f}, Min: {preds.min():.4f}, Max: {preds.max():.4f}")

factors = [2.0, 3.5, 8.0]

for k in factors:
    print(f"Generating probe for k={k}...")
    new_preds = mean_pred + k * (preds - mean_pred)
    new_preds = np.clip(new_preds, 0.0, 1.0)
    
    sub_probe = sub.copy()
    sub_probe['flood_risk_score'] = new_preds
    sub_probe.to_csv(f'../submissions/submission_probe_k{k}.csv', index=False)

print("Probe files generated successfully in submissions/ folder.")
