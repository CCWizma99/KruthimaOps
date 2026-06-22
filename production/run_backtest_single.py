import sys
import os

if len(sys.argv) < 2:
    print("Usage: python run_backtest_single.py <YYYY-MM-DD>")
    sys.exit(1)

date_str = sys.argv[1]
model_version = os.getenv("MODEL_VERSION", "prod_v1000")

print(f"===========================================================")
print(f" MODEL VERSION : {model_version}")
print(f" TARGET DATE   : {date_str}")
print(f"===========================================================")

# Must import after env var is set
from app.inference import load_artifacts, get_district_reference
from app.forecast import get_historical_forecasts_batched

# Load specific model artifacts based on env var
try:
    load_artifacts()
except Exception as e:
    print(f"Failed to load artifacts for {model_version}: {e}")
    sys.exit(1)

ref = get_district_reference()
if not ref:
    print("Failed to load district reference.")
    sys.exit(1)

districts = sorted(list(ref.keys()))
print(f"Running simulation for {len(districts)} districts...\n")

results = get_historical_forecasts_batched(districts, date_str)

print(f"{'District':<25} | {'Rain (7d)':<12} | {'Risk Score':<12} | {'Risk Level'}")
print("-" * 75)
for dist in districts:
    info = results.get(dist, {})
    if "error" in info:
        print(f"{dist:<25} | ERROR: {info['error']}")
    elif not info:
        print(f"{dist:<25} | No data returned")
    else:
        rain = info.get("rainfall_7d_mm", "N/A")
        score = info.get("risk_score", "N/A")
        level = info.get("risk_level", "N/A")
        
        # formatting numbers nicely if they exist
        rain_str = f"{rain:.1f} mm" if isinstance(rain, float) else str(rain)
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        
        print(f"{dist:<25} | {rain_str:<12} | {score_str:<12} | {level}")

print("\n")
