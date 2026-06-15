import pandas as pd
import numpy as np

df = pd.read_csv("c:/KruthimaOps/data/train.csv")

print("Value counts of is_synthetic:")
print(df["is_synthetic"].value_counts())

for is_syn in [True, False]:
    sub = df[df["is_synthetic"] == is_syn]
    print(f"\n--- is_synthetic == {is_syn} (count={len(sub)}) ---")
    print("Mean target:", sub["flood_risk_score"].mean())
    print("Std target:", sub["flood_risk_score"].std())
    print("Rainfall correlation with target:", sub["rainfall_7d_mm"].corr(sub["flood_risk_score"]))
    print("Inundation correlation with target:", sub["inundation_area_sqm"].corr(sub["flood_risk_score"]))
    print("River distance correlation with target:", sub["distance_to_river_m"].corr(sub["flood_risk_score"]))
    print("Flood occurrence correlation (mean risk by occurrence):")
    print(sub.groupby("flood_occurrence_current_event")["flood_risk_score"].mean())
    print("Is good to live correlation (mean risk by is_good):")
    print(sub.groupby("is_good_to_live")["flood_risk_score"].mean())
