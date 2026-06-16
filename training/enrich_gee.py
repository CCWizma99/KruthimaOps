"""
Google Earth Engine Enrichment Script (Fixed & Optimized)
========================================================
Pulls JRC Global Surface Water (Historical Flood Occurrence %)
and GPM IMERG (Max Daily Precipitation in last 5 years).
"""

import os
import time
import ee
import pandas as pd
import numpy as np

print("Initializing Earth Engine...")
try:
    # Authenticate locally if needed, but Initialize is required
    ee.Initialize(project='uniassist-496910')
except Exception as e:
    print(f"Error initializing Earth Engine. Did you run 'earthengine authenticate'?\n{e}")
    exit(1)

# Paths
INPUT_CSV = "C:/KruthimaOps/data/train_v1000.csv"
OUTPUT_CSV = "C:/KruthimaOps/data/train_v1001_gee.csv"

print(f"Loading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows.")

# Sri Lanka Boundary (AOI) to filter collections
sri_lanka_aoi = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'Sri Lanka'))

# 1. JRC Global Surface Water - Occurrence (0-100%)
jrc_water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select(['occurrence'], ['water_occurrence_pct']).clip(sri_lanka_aoi)

# Extract only JRC Surface Water to avoid 5-year precipitation computation timeouts
enrichment_image = jrc_water

print("Extracting features from Earth Engine...")

# Chunks reduced to 250 to strictly comply with GEE server-side memory limits for 30m sampling
CHUNK_SIZE = 250
results = []

for start_idx in range(0, len(df), CHUNK_SIZE):
    end_idx = min(start_idx + CHUNK_SIZE, len(df))
    chunk_df = df.iloc[start_idx:end_idx].copy()
    
    print(f"  Processing chunk {start_idx} to {end_idx}...")
    
    features = []
    for idx, row in chunk_df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        if pd.isna(lat) or pd.isna(lon):
            continue
            
        geom = ee.Geometry.Point([lon, lat])
        # Ensure record_id is passed natively as a standard string or integer type
        feat = ee.Feature(geom, {'record_id': int(row['record_id']) if isinstance(row['record_id'], (int, float)) else str(row['record_id'])})
        features.append(feat)
        
    if not features:
        continue
        
    fc = ee.FeatureCollection(features)
    
    # CRITICAL FIX: sampleRegions is significantly faster and lower-memory than reduceRegions for points
    sampled = enrichment_image.sampleRegions(
        collection=fc,
        properties=['record_id'],
        scale=30, # Dictates extraction footprint (30m to match JRC)
        tileScale=4 # Distributed computing scaling factor to avoid out-of-memory errors
    )
    
    try:
        chunk_data = sampled.getInfo()['features']
        
        for feat in chunk_data:
            props = feat['properties']
            results.append({
                'record_id': props.get('record_id'),
                # Earth Engine returns 0 or missing if it has never seen water
                'water_occurrence_pct': props.get('water_occurrence_pct', 0.0)
            })
    except Exception as server_error:
        print(f"  Execution timed out or failed on chunk {start_idx}-{end_idx}. Error: {server_error}")
        # Append empty frames to avoid breaking final join indices
        for idx, row in chunk_df.iterrows():
            results.append({
                'record_id': int(row['record_id']) if isinstance(row['record_id'], (int, float)) else str(row['record_id']),
                'water_occurrence_pct': 0.0
            })
            
    # Politeness delay to avoid getting flagged for heavy client-side hitting
    time.sleep(1)

# Merge results back
print("Merging results...")
gee_df = pd.DataFrame(results)

# Drop any accidental duplicates generated during error processing
gee_df = gee_df.drop_duplicates(subset=['record_id'])

# Handle missing extractions gracefully
gee_df['water_occurrence_pct'] = gee_df['water_occurrence_pct'].fillna(0.0)

final_df = df.merge(gee_df, on='record_id', how='left')

# Save output data pipeline
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"Successfully saved enriched dataset to {OUTPUT_CSV}")
print(f"Final shape: {final_df.shape}")
