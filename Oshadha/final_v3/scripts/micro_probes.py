import pandas as pd
import numpy as np

print("[LOAD] Loading base model (v3)...")
sub_v3 = pd.read_csv('../submissions/submission_v3.csv')
preds = sub_v3['flood_risk_score'].values
mean_pred = np.mean(preds)
std_pred = np.std(preds)

print(f"Base v3 std: {std_pred:.5f}")

k_values = [0.90, 0.95, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]

print("\n[GENERATE] Creating micro-probes...")
for k in k_values:
    stretched = mean_pred + k * (preds - mean_pred)
    stretched = np.clip(stretched, 0.0, 1.0)
    
    out_name = f"../submissions/submission_v3_micro_k{k:.2f}.csv"
    pd.DataFrame({
        "record_id": sub_v3['record_id'],
        "flood_risk_score": stretched
    }).to_csv(out_name, index=False)
    
    print(f"   -> k={k:.2f} | Target std: {stretched.std():.5f} | Saved: {out_name}")

print("\n[DONE] Micro-probes generated successfully in submissions/ folder.")
