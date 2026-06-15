import pandas as pd
df = pd.read_csv("c:/KruthimaOps/data/train.csv")

cols = ["district", "rainfall_7d_mm", "inundation_area_sqm", "flood_occurrence_current_event", "is_good_to_live", "reason_not_good_to_live", "distance_to_river_m", "elevation_m", "flood_risk_score"]

print("--- 5 HIGH RISK ROWS (>0.95) ---")
print(df[df["flood_risk_score"] > 0.95][cols].head(5))

print("\n--- 5 LOW RISK ROWS (<0.05) ---")
print(df[df["flood_risk_score"] < 0.05][cols].head(5))
