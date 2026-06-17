import re
import os

main_path = "c:/KruthimaOps/production/app/main.py"
app_js_path = "c:/KruthimaOps/production/app/static/app.js"

# Patch main.py
with open(main_path, "r", encoding="utf-8") as f:
    main_content = f.read()

main_patch = """async def predict_subdivisions(district_name: str, date: str = None):
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
    
    if date:
        from datetime import datetime, timedelta
        target_dt = datetime.strptime(date, "%Y-%m-%d")
        start_date = (target_dt - timedelta(days=7)).strftime("%Y-%m-%d")
        end_date = (target_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lats_str}&longitude={lons_str}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=precipitation_sum"
            f"&timezone=Asia%2FColombo"
        )
    else:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lats_str}&longitude={lons_str}"
            f"&daily=precipitation_sum"
            f"&past_days=7&forecast_days=1"
            f"&timezone=Asia%2FColombo"
        )"""

main_content = re.sub(
    r'async def predict_subdivisions\(district_name: str\):.*?f"&timezone=Asia%2FColombo"\n    \)',
    main_patch,
    main_content,
    flags=re.DOTALL
)

with open(main_path, "w", encoding="utf-8") as f:
    f.write(main_content)

# Patch app.js
with open(app_js_path, "r", encoding="utf-8") as f:
    app_js_content = f.read()

app_js_patch = """
    // Add custom Pins logic for subdivisions
    const dateQuery = state.simulationDate ? `?date=${state.simulationDate}` : '';
    const res = await fetch(`/api/predict/subdivisions/${districtName}${dateQuery}`);
"""

app_js_content = re.sub(
    r'// Add custom Pins logic for subdivisions\s*const res = await fetch\(`/api/predict/subdivisions/\$\{districtName\}`\);',
    app_js_patch,
    app_js_content
)

with open(app_js_path, "w", encoding="utf-8") as f:
    f.write(app_js_content)

print("Patched main.py and app.js")
