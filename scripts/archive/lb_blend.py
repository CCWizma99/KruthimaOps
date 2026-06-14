import numpy as np
import pandas as pd
import os

SUB_DIR = "submissions"

# Top 3 Models on actual Kaggle Public LB
# 1. v703_optimized (0.38203) - 50% weight
# 2. v70_optimized  (0.38216) - 25% weight
# 3. v67_optimized  (0.38216) - 25% weight

weights = {
    "v703_optimized": 0.50,
    "v70_optimized": 0.25,
    "v67_optimized": 0.25
}

print("Loading OOF arrays and Submission files...")
oof_blend = None
sub_blend = None

for name, w in weights.items():
    print(f"  Adding {name} (Weight: {w})")
    
    # OOF
    oof_path = os.path.join(SUB_DIR, f"oof_{name}.npy")
    oof_arr = np.load(oof_path)
    if oof_blend is None:
        oof_blend = oof_arr * w
    else:
        oof_blend += oof_arr * w
        
    # Submission
    sub_path = os.path.join(SUB_DIR, f"submission_{name}.csv")
    sub_df = pd.read_csv(sub_path)
    if sub_blend is None:
        sub_blend = sub_df.copy()
        sub_blend['flood_risk_score'] = sub_df['flood_risk_score'] * w
    else:
        sub_blend['flood_risk_score'] += sub_df['flood_risk_score'] * w

oof_blend = np.clip(oof_blend, 0.0, 1.0)
sub_blend['flood_risk_score'] = np.clip(sub_blend['flood_risk_score'], 0.0, 1.0)

# Save
oof_out = os.path.join(SUB_DIR, "oof_lb_blend.npy")
sub_out = os.path.join(SUB_DIR, "submission_lb_blend.csv")

np.save(oof_out, oof_blend)
sub_blend.to_csv(sub_out, index=False)

print(f"\n[DONE] Saved {sub_out} and {oof_out}")
print(f"Prediction range: [{sub_blend['flood_risk_score'].min():.4f}, {sub_blend['flood_risk_score'].max():.4f}]")
