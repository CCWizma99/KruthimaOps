import pandas as pd
import numpy as np

# Load baseline submission and train targets
print("Loading data...")
sub = pd.read_csv('submission_v3.csv')
train = pd.read_csv('train.csv')

preds = sub['flood_risk_score'].values
mean_pred = np.mean(preds)

print(f"Base predictions - Mean: {mean_pred:.4f}, Min: {preds.min():.4f}, Max: {preds.max():.4f}")

# --- 1. VARIANCE LADDER ---
factors = [1.05, 1.10, 1.20]

for f in factors:
    print(f"Generating {f}x Variance Submission...")
    new_preds = mean_pred + f * (preds - mean_pred)
    new_preds = np.clip(new_preds, 0.0, 1.0)
    
    sub_ladder = sub.copy()
    sub_ladder['flood_risk_score'] = new_preds
    sub_ladder.to_csv(f'submission_v3_{int(f*100)}x.csv', index=False)

# --- 2. TARGET DISTRIBUTION MATCHING ---
print("Generating Quantile Matched Submission...")
# Get exact distribution of training targets
sorted_targets = np.sort(train['flood_risk_score'].dropna().values)

# Calculate percentiles of our predictions
from scipy.stats import rankdata
# rankdata returns 1 to N. Divide by N to get 0 to 1 percentiles.
pred_ranks = rankdata(preds, method='average') - 1
pred_percentiles = pred_ranks / (len(preds) - 1) * 100

# Map percentiles to the training target percentiles
matched_preds = np.percentile(sorted_targets, pred_percentiles)
matched_preds = np.clip(matched_preds, 0.0, 1.0)

sub_quantile = sub.copy()
sub_quantile['flood_risk_score'] = matched_preds
sub_quantile.to_csv('submission_quantile.csv', index=False)
print("Files generated successfully.")
