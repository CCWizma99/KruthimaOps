import sys
sys.path.append('c:/KruthimaOps/production')
from app.inference.v1000_engine import load_artifacts, infer
import numpy as np
import pandas as pd
import json

load_artifacts()

print("--- TESTING EXTREME WET/FLOOD SCENARIO ---")
payload_extreme = {
  "district": "Colombo",
  "rainfall_7d_mm": 500.0,
  "inundation_area_sqm": 100000.0,
  "flood_occurrence_current_event": "Yes",
  "is_good_to_live": "No",
  "reason_not_good_to_live": "High flood risk"
}
out_extreme = infer(payload_extreme)
print(f"Extreme Scenario score: {out_extreme}\n")

print("--- TESTING DRY/SAFE SCENARIO ---")
payload_safe = {
  "district": "Colombo",
  "rainfall_7d_mm": 0.0,
  "inundation_area_sqm": 0.0,
  "flood_occurrence_current_event": "No",
  "is_good_to_live": "Yes",
  "reason_not_good_to_live": "Other"
}
out_safe = infer(payload_safe)
print(f"Safe Scenario score: {out_safe}\n")
