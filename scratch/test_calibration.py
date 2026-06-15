import sys
sys.path.append('c:/KruthimaOps/production')
from app.inference.v703_engine import load_artifacts, infer
import numpy as np
import pandas as pd

load_artifacts()

def get_calibrated_score(payload):
    raw_score = infer(payload)
    
    rain = float(payload.get("rainfall_7d_mm", 0.0))
    inund = float(payload.get("inundation_area_sqm", 0.0))
    flood = str(payload.get("flood_occurrence_current_event", "No")).strip().lower()
    is_good = str(payload.get("is_good_to_live", "Yes")).strip().lower()
    
    R = min(rain / 300.0, 1.0) if rain > 0 else 0.0
    I = min(inund / 25000.0, 1.0) if inund > 0 else 0.0
    F = 1.0 if flood == "yes" else 0.0
    U = 1.0 if is_good == "no" else 0.0
    
    pri = 0.3 * R + 0.3 * I + 0.2 * F + 0.2 * U
    
    # Let's adjust range dynamically:
    # 0.58 is safe baseline, 0.38 is extreme wet baseline
    raw_risk = (0.58 - raw_score) / (0.58 - 0.38)
    raw_risk = np.clip(raw_risk, 0.0, 1.0)
    
    blended = 0.6 * raw_risk + 0.4 * pri
    
    final_score = 0.05 + blended * 0.90
    return float(np.clip(final_score, 0.02, 0.99))

districts = ["Colombo", "Anuradhapura", "Nuwara Eliya", "Jaffna", "Monaragala"]
for dist in districts:
    payload = {
        "district": dist,
        "rainfall_7d_mm": 0.0,
        "inundation_area_sqm": 0.0,
        "flood_occurrence_current_event": "No",
        "is_good_to_live": "Yes",
        "reason_not_good_to_live": "None"
    }
    print(f"Safe dry {dist}: calibrated={get_calibrated_score(payload):.4f} (raw={infer(payload):.4f})")
