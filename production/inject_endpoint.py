import re

path = 'c:/KruthimaOps/production/app/main.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

endpoint_code = """
# ── Subdivisions ───────────────────────────────────────────────────────
import json
import urllib.request
@app.get("/api/predict/subdivisions/{district_name}", tags=["Inference"])
async def predict_subdivisions(district_name: str):
    file_path = os.path.join(static_dir, "subdivisions", f"{district_name.replace(' ', '_')}.geojson")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Subdivisions not found")
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    subs = []
    for feat in data["features"]:
        name = feat["properties"]["shapeName"]
        coords = feat["geometry"]["coordinates"]
        
        pts = []
        def ext(arr):
            if isinstance(arr[0], (int, float)): pts.append(arr)
            else:
                for x in arr: ext(x)
        ext(coords)
        if not pts: continue
        lat = sum(p[1] for p in pts) / len(pts)
        lon = sum(p[0] for p in pts) / len(pts)
        # Avoid duplicate names in multipolygons
        if not any(s["name"] == name for s in subs):
            subs.append({"name": name, "lat": lat, "lon": lon})
            
    if not subs: return []
    
    lats_str = ",".join(str(s["lat"]) for s in subs)
    lons_str = ",".join(str(s["lon"]) for s in subs)
    
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lats_str}&longitude={lons_str}"
        f"&daily=precipitation_sum"
        f"&past_days=7&forecast_days=1"
        f"&timezone=Asia%2FColombo"
    )
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Flood TimelineSL"})
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode())
    except Exception as e:
        logger.error(f"Open-meteo failed for subdivisions: {e}")
        res_data = None
        
    results = []
    for idx, sub in enumerate(subs):
        sum_7d = 0.0
        if res_data and isinstance(res_data, list) and idx < len(res_data):
            daily = res_data[idx].get("daily", {})
            precip = daily.get("precipitation_sum", [])
            sum_7d = sum(p for p in precip[:7] if p is not None)
        elif res_data and isinstance(res_data, dict):
            daily = res_data.get("daily", {})
            precip = daily.get("precipitation_sum", [])
            sum_7d = sum(p for p in precip[:7] if p is not None)
            
        payload = {
            "district": district_name,
            "place_name": sub["name"],
            "latitude": sub["lat"],
            "longitude": sub["lon"],
            "rainfall_7d_mm": sum_7d,
            "inundation_area_sqm": 0.0,
            "flood_occurrence_current_event": "No",
            "is_good_to_live": "Yes",
            "reason_not_good_to_live": "None"
        }
        
        try:
            score_tuple = infer(payload)
            score = score_tuple[0] if isinstance(score_tuple, tuple) else score_tuple
        except Exception:
            score = 0.1
            
        level = "LOW"
        if score >= 0.75: level = "EXTREME"
        elif score >= 0.5: level = "HIGH"
        elif score >= 0.25: level = "MEDIUM"
        
        results.append({
            "place_name": sub["name"],
            "lat": sub["lat"],
            "lon": sub["lon"],
            "rainfall_7d_mm": round(sum_7d, 2),
            "risk_score": round(score, 4),
            "risk_level": level
        })
        
    return results

"""

if "predict_subdivisions(" not in content:
    content += "\n" + endpoint_code
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Endpoint injected")
else:
    print("Endpoint already exists")
