import pandas as pd
import numpy as np

# Load submissions
sub13 = pd.read_csv("submissions/submission_v13.csv")
sub16 = pd.read_csv("submissions/submission_v16.csv")
sub17 = pd.read_csv("submissions/submission_v17.csv")

assert (sub13['record_id'] == sub16['record_id']).all()
assert (sub13['record_id'] == sub17['record_id']).all()

# Blend v13 (best LB) + v16 (MAE insight + algo diversity)
blend_13_16 = sub13.copy()
blend_13_16['flood_risk_score'] = 0.5 * sub13['flood_risk_score'] + 0.5 * sub16['flood_risk_score']
blend_13_16.to_csv("submissions/submission_blend_v13_v16.csv", index=False)

# Also make v13+v17 blend for comparison
blend_13_17 = sub13.copy()
blend_13_17['flood_risk_score'] = 0.5 * sub13['flood_risk_score'] + 0.5 * sub17['flood_risk_score']
blend_13_17.to_csv("submissions/submission_blend_v13_v17.csv", index=False)

# And v13-heavy blend (60/40)
blend_13h_16 = sub13.copy()
blend_13h_16['flood_risk_score'] = 0.6 * sub13['flood_risk_score'] + 0.4 * sub16['flood_risk_score']
blend_13h_16.to_csv("submissions/submission_blend_v13h_v16.csv", index=False)

print("=== Blend Analysis ===")
print(f"\nv13 range:  [{sub13['flood_risk_score'].min():.4f}, {sub13['flood_risk_score'].max():.4f}]")
print(f"v16 range:  [{sub16['flood_risk_score'].min():.4f}, {sub16['flood_risk_score'].max():.4f}]")
print(f"v17 range:  [{sub17['flood_risk_score'].min():.4f}, {sub17['flood_risk_score'].max():.4f}]")

print(f"\nBlend v13+v16 (50/50): [{blend_13_16['flood_risk_score'].min():.4f}, {blend_13_16['flood_risk_score'].max():.4f}]")
print(f"Blend v13+v17 (50/50): [{blend_13_17['flood_risk_score'].min():.4f}, {blend_13_17['flood_risk_score'].max():.4f}]")
print(f"Blend v13+v16 (60/40): [{blend_13h_16['flood_risk_score'].min():.4f}, {blend_13h_16['flood_risk_score'].max():.4f}]")

# Pairwise correlation between submissions
corr_13_16 = np.corrcoef(sub13['flood_risk_score'], sub16['flood_risk_score'])[0,1]
corr_13_17 = np.corrcoef(sub13['flood_risk_score'], sub17['flood_risk_score'])[0,1]
corr_16_17 = np.corrcoef(sub16['flood_risk_score'], sub17['flood_risk_score'])[0,1]

print(f"\nPairwise correlations:")
print(f"  v13 vs v16: {corr_13_16:.5f}")
print(f"  v13 vs v17: {corr_13_17:.5f}")
print(f"  v16 vs v17: {corr_16_17:.5f}")
print(f"\n  Lower correlation = more diversity = better blend potential")

# Mean absolute difference between submissions
mad_13_16 = np.abs(sub13['flood_risk_score'] - sub16['flood_risk_score']).mean()
mad_13_17 = np.abs(sub13['flood_risk_score'] - sub17['flood_risk_score']).mean()
print(f"\nMean absolute difference:")
print(f"  v13 vs v16: {mad_13_16:.5f}")
print(f"  v13 vs v17: {mad_13_17:.5f}")
