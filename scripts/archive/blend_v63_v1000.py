import pandas as pd
import numpy as np
import os

print("=" * 60)
print("  EVIL ML: VARIANCE-FORCED ENSEMBLE (v63 + v1000)")
print("=" * 60)

DATA_DIR = "/kaggle/input/competitions/ml-opsidian-genesis-initial-round-26"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "data"

# 1. Load Submissions and Ground Truth
print("\n[LOAD] Loading submission files and ground truth...")
sub_63 = pd.read_csv("submission_v63_optimized.csv")
sub_1000 = pd.read_csv("submission_v1000_optimized.csv")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train_df = train_df.drop_duplicates()
real_mask = train_df['is_synthetic'] == 0 if 'is_synthetic' in train_df.columns else np.ones(len(train_df), dtype=bool)

# The absolute ground truth to copy the histogram from
true_y = train_df['flood_risk_score'].values

# 2. Raw Blend
print("\n[BLEND] Creating raw 50/50 average blend...")
raw_blend = 0.5 * sub_63['flood_risk_score'] + 0.5 * sub_1000['flood_risk_score']

# Demonstrate the Variance Shrinkage
var_63 = sub_63['flood_risk_score'].var()
var_1000 = sub_1000['flood_risk_score'].var()
var_blend = raw_blend.var()

print(f"   Variance v63   : {var_63:.6f}")
print(f"   Variance v1000 : {var_1000:.6f}")
print(f"   Variance BLEND : {var_blend:.6f} <-- MASSIVE SHRINKAGE DETECTED!")

# 3. Histogram Forcing (Quantile Mapping)
print("\n[MAD SCIENCE] Applying Histogram Forcing to restore Variance...")

# Sort the true training labels
y_train_sorted = np.sort(true_y)

# Find percentiles for the blended test predictions
percentiles = np.linspace(0, 100, len(raw_blend))
forced_values = np.percentile(y_train_sorted, percentiles)

# Sort the blended predictions to get mapping indices
sort_idx = np.argsort(raw_blend.values)

# Create a new array and map the exact train percentiles back to the blend ordering
forced_blend = np.zeros_like(raw_blend.values)
forced_blend[sort_idx] = forced_values

print(f"   Variance FORCED: {forced_blend.var(ddof=1):.6f} <-- FULL VARIANCE RESTORED!")

# 4. Save Submission
submission = sub_63.copy()
submission['flood_risk_score'] = forced_blend
out_path = "submissions/blend_v63_v1000_forced.csv"
submission.to_csv(out_path, index=False)

print(f"\n[DONE] Saved perfectly mapped blend to {out_path}")
print("=" * 60)
