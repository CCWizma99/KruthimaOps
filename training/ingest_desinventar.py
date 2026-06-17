import pandas as pd
import json
import time
import requests
import os
import datetime
import numpy as np

DI_FILE = "C:/KruthimaOps/data/SriLankaOldData/DI_report70416.xls"
TRAIN_BASE_FILE = "C:/KruthimaOps/data/train_v1001_gee.csv"
OUTPUT_FILE = "C:/KruthimaOps/data/train_v1002_desinventar.csv"
DIST_REF = "C:/KruthimaOps/production/data/district_reference.json"

def fetch_rainfall(lat, lon, date_str):
    """Fetch 7 day rainfall before the date from Open-Meteo"""
    try:
        # date_str is YYYY/MM/DD
        dt = datetime.datetime.strptime(date_str, "%Y/%m/%d")
        end_date = dt.strftime("%Y-%m-%d")
        start_date = (dt - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=precipitation_sum&timezone=auto"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            precip = data.get("daily", {}).get("precipitation_sum", [])
            precip = [p for p in precip if p is not None]
            if precip:
                return sum(precip), sum(precip) * 4 # rough monthly estimate
    except Exception as e:
        print(f"Error fetching for {date_str}: {e}")
    return None, None

def run():
    print("Loading district references...")
    with open(DIST_REF, "r") as f:
        dist_ref = json.load(f)
        
    print("Loading base training data...")
    base_df = pd.read_csv(TRAIN_BASE_FILE)
    
    print("Loading DesInventar data...")
    # It's a TSV. skip bad lines
    di_df = pd.read_csv(DI_FILE, sep="\t", on_bad_lines='skip')
    print(f"Loaded {len(di_df)} rows from DesInventar.")
    
    # Clean column names
    di_df.columns = [c.replace('"', '').strip() for c in di_df.columns]
    
    # Filter Event
    allowed_events = ["FLOOD", "HEAVY RAINS", "CYCLONE", "STORM"]
    di_df = di_df[di_df["Event"].isin(allowed_events)]
    
    # Filter Date >= 2010
    di_df = di_df[di_df["Date (YMD)"].str.startswith("201") | di_df["Date (YMD)"].str.startswith("202")]
    
    print(f"Filtered to {len(di_df)} relevant events since 2010.")
    
    if len(di_df) > 1500:
        di_df = di_df.sample(1500, random_state=42)
        print("Sampled down to 1500 records to respect API limits.")
        
    new_rows = []
    
    # Normalize district names in dict
    dist_map = {k.lower().strip(): v for k, v in dist_ref.items()}
    
    for idx, row in di_df.iterrows():
        dist_raw = str(row.get("District", "")).replace('"', '').strip()
        date_str = str(row.get("Date (YMD)", ""))
        
        # Match district
        dist_key = dist_raw.lower()
        if dist_key not in dist_map:
            # Fallback mapping
            found = False
            for k in dist_map:
                if k in dist_key or dist_key in k:
                    dist_key = k
                    found = True
                    break
            if not found:
                continue
                
        ref_data = dist_map[dist_key]
        
        houses_destroyed = pd.to_numeric(row.get("Houses Destroyed", 0), errors='coerce')
        houses_damaged = pd.to_numeric(row.get("Houses Damaged", 0), errors='coerce')
        relocated = pd.to_numeric(row.get("Relocated", 0), errors='coerce')
        crops_ha = pd.to_numeric(row.get("Damages in crops Ha.", 0), errors='coerce')
        
        # NaN handling
        houses_destroyed = houses_destroyed if not pd.isna(houses_destroyed) else 0
        houses_damaged = houses_damaged if not pd.isna(houses_damaged) else 0
        relocated = relocated if not pd.isna(relocated) else 0
        crops_ha = crops_ha if not pd.isna(crops_ha) else 0
        
        is_good_to_live = "No" if (houses_destroyed > 0 or relocated > 0) else "Yes"
        inundation_area = crops_ha * 10000
        if inundation_area == 0 and houses_damaged > 0:
            inundation_area = houses_damaged * 200 # guess 200sqm per house damaged
            
        lat = ref_data.get("center_lat", ref_data.get("latitude", 7.8))
        lon = ref_data.get("center_lon", ref_data.get("longitude", 80.7))
        
        rain_7d, rain_month = fetch_rainfall(lat, lon, date_str)
        if rain_7d is None:
            rain_7d = ref_data.get("rainfall_7d_mm", 50.0)
            rain_month = ref_data.get("monthly_rainfall_mm", 200.0)
            
        new_row = {
            "district": ref_data.get("district", dist_raw),
            "place_name": str(row.get("Location", "")) or str(row.get("Division", "Unknown")),
            "latitude": lat,
            "longitude": lon,
            "flood_occurrence_current_event": "Yes",
            "is_good_to_live": is_good_to_live,
            "reason_not_good_to_live": "Flood damage" if is_good_to_live == "No" else "None",
            "inundation_area_sqm": inundation_area,
            "rainfall_7d_mm": rain_7d,
            "monthly_rainfall_mm": rain_month,
            "elevation_m": ref_data.get("elevation_m"),
            "distance_to_river_m": ref_data.get("distance_to_river_m"),
            "landcover": ref_data.get("landcover", "Mixed"),
            "soil_type": ref_data.get("soil_type", "Loamy"),
            "drainage_index": ref_data.get("drainage_index", 0.5),
            "water_supply": ref_data.get("water_supply", "Well"),
            "electricity": ref_data.get("electricity", "Grid"),
            "road_quality": ref_data.get("road_quality", "Fair (unpaved)"),
            "urban_rural": ref_data.get("urban_rural", "Rural"),
            "water_presence_flag": "Likely",
            "generation_date": date_str.replace("/", "-")
        }
        new_rows.append(new_row)
        
        if len(new_rows) % 100 == 0:
            print(f"Processed {len(new_rows)} rows...")
            time.sleep(0.5) # Be nice to open-meteo
            
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Ensure all columns in base_df are present
        for col in base_df.columns:
            if col not in new_df.columns:
                new_df[col] = base_df[col].median() if pd.api.types.is_numeric_dtype(base_df[col]) else None
        
        # Keep real data as not synthetic
        new_df["is_synthetic"] = np.nan
        new_df["is_pseudo"] = np.nan
        
        new_df = new_df[base_df.columns]
        combined_df = pd.concat([base_df, new_df], ignore_index=True)
        combined_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved to {OUTPUT_FILE}! Added {len(new_df)} new records.")
    else:
        print("No rows generated.")

if __name__ == "__main__":
    run()
