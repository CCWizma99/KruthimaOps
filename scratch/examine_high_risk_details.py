import pandas as pd
df = pd.read_csv("c:/KruthimaOps/data/train.csv")

high_rows = df[df["flood_risk_score"] > 0.95].head(3)
low_rows = df[df["flood_risk_score"] < 0.05].head(3)

print("--- HIGH RISK ROWS (ALL COLUMNS) ---")
for idx, row in high_rows.iterrows():
    print(f"\nRow {idx} (Score: {row['flood_risk_score']}):")
    print({k: v for k, v in row.to_dict().items() if not pd.isna(v)})

print("\n--- LOW RISK ROWS (ALL COLUMNS) ---")
for idx, row in low_rows.iterrows():
    print(f"\nRow {idx} (Score: {row['flood_risk_score']}):")
    print({k: v for k, v in row.to_dict().items() if not pd.isna(v)})
