import numpy as np
import pandas as pd
from evaluate import evaluate_predictions, print_evaluation

# Load ground truth
train_df = pd.read_csv("data/train.csv")
train_df = train_df.drop_duplicates()
y_true = train_df['flood_risk_score'].values

# Load OOFs
oof13 = np.load("submissions/oof_v13.npy")
oof17 = np.load("submissions/oof_v17.npy")
oof19 = np.load("submissions/oof_v19.npy")

print("\n" + "="*60)
print("  EVALUATING BLENDS USING NEW COMPETITION METRIC")
print("="*60)

# Blend 1: v13 (best LB prior) + v17 (pure MAE) 50/50
blend_13_17 = 0.5 * oof13 + 0.5 * oof17
res = evaluate_predictions(y_true, blend_13_17, label="Blend 50% v13 + 50% v17")
print_evaluation(res, verbose=False)
print(f"  Gap to Rank 1: {res['est_LB'] - 0.38215:+.5f}")
print(f"  vs v19: {res['est_LB'] - 0.38386:+.5f}")

# Blend 2: v13 + v19 (the two strongest LB candidates)
blend_13_19 = 0.5 * oof13 + 0.5 * oof19
res2 = evaluate_predictions(y_true, blend_13_19, label="Blend 50% v13 + 50% v19")
print_evaluation(res2, verbose=False)
print(f"  Gap to Rank 1: {res2['est_LB'] - 0.38215:+.5f}")
print(f"  vs v19: {res2['est_LB'] - 0.38386:+.5f}")

# Blend 3: 33% v13, 33% v17, 33% v19
blend_13_17_19 = (oof13 + oof17 + oof19) / 3.0
res3 = evaluate_predictions(y_true, blend_13_17_19, label="Blend 33% v13 + 33% v17 + 33% v19")
print_evaluation(res3, verbose=False)
print(f"  Gap to Rank 1: {res3['est_LB'] - 0.38215:+.5f}")
print(f"  vs v19: {res3['est_LB'] - 0.38386:+.5f}")
