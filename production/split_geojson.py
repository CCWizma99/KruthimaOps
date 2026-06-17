import json
import pandas as pd
import math
import os

print("Loading DesInventar training data for mapping...")
df = pd.read_csv("C:/KruthimaOps/data/train_v1002_desinventar.csv")

# Map place_name to district
mapping = {}
for _, row in df.iterrows():
    if pd.notna(row['place_name']) and pd.notna(row['district']):
        mapping[str(row['place_name']).lower().strip()] = str(row['district']).strip()

print("Loading ADM3 GeoJSON...")
with open("C:/KruthimaOps/production/data/dsd.geojson", "r", encoding="utf-8") as f:
    adm3 = json.load(f)

print("Loading District Reference...")
with open("C:/KruthimaOps/production/data/district_reference.json", "r", encoding="utf-8") as f:
    dist_ref = json.load(f)

# Helper for centroid distance mapping as fallback
def dist(lat1, lon1, lat2, lon2):
    return math.hypot(lat1 - lat2, lon1 - lon2)

def get_centroid(coords):
    # Flatten the coords to compute a rough centroid
    pts = []
    def extract(arr):
        if isinstance(arr[0], (int, float)):
            pts.append(arr)
        else:
            for item in arr: extract(item)
    extract(coords)
    if not pts: return 0, 0
    lats = [p[1] for p in pts]
    lons = [p[0] for p in pts]
    return sum(lats)/len(lats), sum(lons)/len(lons)

district_features = {d: [] for d in dist_ref.keys()}

matched_direct = 0
matched_spatial = 0

for feat in adm3['features']:
    name = feat['properties']['shapeName'].lower().strip()
    
    # Try direct name match from training data
    district = None
    if name in mapping:
        district = mapping[name]
    else:
        # Fuzzy match
        for k, v in mapping.items():
            if name in k or k in name:
                district = v
                break
                
    if district and district in district_features:
        district_features[district].append(feat)
        matched_direct += 1
    else:
        # Spatial fallback
        lat, lon = get_centroid(feat['geometry']['coordinates'])
        best_dist = float('inf')
        best_d = None
        for d_name, d_data in dist_ref.items():
            d_lat = d_data.get('center_lat', d_data['latitude'])
            d_lon = d_data.get('center_lon', d_data['longitude'])
            curr_dist = dist(lat, lon, d_lat, d_lon)
            if curr_dist < best_dist:
                best_dist = curr_dist
                best_d = d_name
        if best_d:
            district_features[best_d].append(feat)
            matched_spatial += 1

print(f"Matched Direct: {matched_direct}, Matched Spatial: {matched_spatial}")

# Save separated geojsons
out_dir = "C:/KruthimaOps/production/app/static/subdivisions"
os.makedirs(out_dir, exist_ok=True)

for d, feats in district_features.items():
    if not feats: continue
    out_dict = {
        "type": "FeatureCollection",
        "features": feats
    }
    with open(f"{out_dir}/{d.replace(' ', '_')}.geojson", "w") as f:
        json.dump(out_dict, f)

print("Split complete!")
