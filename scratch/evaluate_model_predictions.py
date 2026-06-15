import sys
sys.path.append('c:/KruthimaOps/production')
from app.inference.v703_engine import load_artifacts, infer
import numpy as np
import pandas as pd

load_artifacts()

df = pd.read_csv("c:/KruthimaOps/data/train.csv")
print("Evaluating serialized model on a sample of train data...")

# Drop rows with missing district for testing
df_clean = df.dropna(subset=["district"]).copy()

# Select 100 random rows from train
sample_df = df_clean.sample(100, random_state=42).copy()
predictions = []

for idx, row in sample_df.iterrows():
    payload = {
        "district": str(row["district"]),
        "rainfall_7d_mm": float(row["rainfall_7d_mm"]) if not pd.isna(row["rainfall_7d_mm"]) else 0.0,
        "inundation_area_sqm": float(row["inundation_area_sqm"]) if not pd.isna(row["inundation_area_sqm"]) else 0.0,
        "flood_occurrence_current_event": str(row["flood_occurrence_current_event"]) if not pd.isna(row["flood_occurrence_current_event"]) else "No",
        "is_good_to_live": str(row["is_good_to_live"]) if not pd.isna(row["is_good_to_live"]) else "Yes",
        "reason_not_good_to_live": str(row["reason_not_good_to_live"]) if not pd.isna(row["reason_not_good_to_live"]) else "None"
    }
    pred = infer(payload)
    predictions.append(pred)

sample_df["pred"] = predictions
print("\nTrue target stats in sample:")
print(sample_df["flood_risk_score"].describe())
print("\nPredicted target stats in sample:")
print(sample_df["pred"].describe())

# Let's print top 10 rows with highest difference
sample_df["abs_diff"] = (sample_df["flood_risk_score"] - sample_df["pred"]).abs()
print("\nTop 10 rows with largest prediction error:")
print(sample_df.sort_values("abs_diff", ascending=False)[["district", "rainfall_7d_mm", "inundation_area_sqm", "flood_occurrence_current_event", "flood_risk_score", "pred", "abs_diff"]].head(10))
