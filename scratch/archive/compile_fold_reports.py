import os
import pandas as pd
import re

submissions_dir = "submissions"
files = sorted(os.listdir(submissions_dir))

report_files = []
for f in files:
    if f.startswith("fold_report_v") and f.endswith(".csv"):
        report_files.append(f)

# Sort reports by version number
def get_num(filename):
    nums = re.findall(r'\d+', filename)
    if not nums:
        return 999
    if "3_5" in filename:
        return 3.5
    return float(nums[0])

report_files = sorted(report_files, key=get_num)

summary_rows = []
for f in report_files:
    version = f.replace("fold_report_", "").replace(".csv", "")
    path = os.path.join(submissions_dir, f)
    try:
        df = pd.read_csv(path)
        # Check column names
        # Normalize column names to uppercase
        df.columns = [c.upper() for c in df.columns]
        
        # If there is an 'OVERALL' row, use it. Otherwise, average the folds.
        # Check if 'FOLD' has 'overall' or 'OVERALL' or 'Overall'
        overall_mask = df['FOLD'].astype(str).str.upper().str.contains('OVERALL')
        if overall_mask.any():
            overall_df = df[overall_mask]
            mae = overall_df['MAE'].values[0]
            rmse = overall_df['RMSE'].values[0]
            ev = overall_df['EV'].values[0]
        else:
            # Average of all folds
            # Drop overall row if it exists under a different name
            clean_df = df[~df['FOLD'].astype(str).str.upper().str.contains('OVERALL')]
            # Convert metric columns to float
            mae = pd.to_numeric(clean_df['MAE'], errors='coerce').mean()
            rmse = pd.to_numeric(clean_df['RMSE'], errors='coerce').mean()
            ev = pd.to_numeric(clean_df['EV'], errors='coerce').mean()
            
        summary_rows.append({
            "Version": version,
            "MAE": mae,
            "RMSE": rmse,
            "EV": ev
        })
    except Exception as e:
        print(f"Error reading {f}: {e}")

summary_df = pd.DataFrame(summary_rows)
# Compute the estimated LB score using the reverse engineered formula:
# LB = -22.87 * MAE + 6.60 * RMSE + 3.03 * (1 - EV)
summary_df["Est_LB"] = -22.87 * summary_df["MAE"] + 6.60 * summary_df["RMSE"] + 3.03 * (1.0 - summary_df["EV"])

print(summary_df.to_string(index=False))
summary_df.to_csv("scratch/compiled_fold_reports.csv", index=False)
