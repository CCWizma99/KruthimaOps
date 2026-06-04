import pandas as pd
import numpy as np

print("Loading submission files...")
sub_safe = pd.read_csv('submission_v3.csv')
sub_high_var = pd.read_csv('submission_v3_120x.csv')
sub_quantile = pd.read_csv('submission_quantile.csv')

# Verify lengths
assert len(sub_safe) == len(sub_high_var) == len(sub_quantile)

print("Blending models: 60% Safe, 20% High-Variance, 20% Quantile-Matched...")
blend_preds = (
    0.6 * sub_safe['flood_risk_score'] + 
    0.2 * sub_high_var['flood_risk_score'] + 
    0.2 * sub_quantile['flood_risk_score']
)

blend_preds = np.clip(blend_preds, 0.0, 1.0)

sub_blend = sub_safe.copy()
sub_blend['flood_risk_score'] = blend_preds
sub_blend.to_csv('submission_blend.csv', index=False)
print("Saved submission_blend.csv")
