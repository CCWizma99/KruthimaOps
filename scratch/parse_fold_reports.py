import glob
import os
import pandas as pd

reports = glob.glob("submissions/fold_report_v*.csv") + glob.glob("fold_report_v*.csv")

# Remove duplicates
seen = set()
unique_reports = []
for r in reports:
    basename = os.path.basename(r)
    if basename not in seen:
        seen.add(basename)
        unique_reports.append(r)

print(f"{'File':<30} | {'MAE':<10} | {'RMSE':<10} | {'EV':<10}")
print("-" * 68)

for r in sorted(unique_reports):
    try:
        df = pd.read_csv(r)
        # Some fold reports have columns: fold, MAE, RMSE, EV
        # Let's take the mean
        mae = df["MAE"].mean()
        rmse = df["RMSE"].mean()
        ev = df["EV"].mean()
        print(f"{os.path.basename(r):<30} | {mae:.6f} | {rmse:.6f} | {ev:.6f}")
    except Exception as e:
        print(f"Error reading {r}: {e}")
