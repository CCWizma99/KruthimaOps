import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
import elevation
import time

import requests
import sys
import zipfile
import glob
from rasterio.merge import merge

def download_dem(output_path):
    print("Downloading SRTM 90m DEM tiles for Sri Lanka from CGIAR...")
    if os.path.exists(output_path):
        print(f"DEM already exists at {output_path}")
        return
    
    urls = [
        "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_52_11.zip",
        "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_53_11.zip"
    ]
    
    temp_dir = os.path.dirname(output_path)
    tifs = []
    
    for url in urls:
        zip_path = os.path.join(temp_dir, os.path.basename(url))
        if not os.path.exists(zip_path):
            print(f"Downloading {url}...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
            else:
                print(f"Failed to download {url}")
                sys.exit(1)
        
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            for name in zip_ref.namelist():
                if name.endswith('.tif'):
                    tifs.append(os.path.join(temp_dir, name))
    
    print("Merging tiles...")
    src_files_to_mosaic = []
    for tif in tifs:
        src = rasterio.open(tif)
        src_files_to_mosaic.append(src)
        
    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })
    
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
        
    for src in src_files_to_mosaic:
        src.close()
        
    print("DEM merge complete.")

def calculate_slope(data, resolution):
    # data is a 3x3 array. resolution is the grid size in meters (approx 30m for SRTM)
    if data.shape != (3, 3):
        return 0.0
    
    dz_dx = ((data[2, 0] + 2*data[2, 1] + data[2, 2]) - (data[0, 0] + 2*data[0, 1] + data[0, 2])) / (8 * resolution)
    dz_dy = ((data[0, 2] + 2*data[1, 2] + data[2, 2]) - (data[0, 0] + 2*data[1, 0] + data[2, 0])) / (8 * resolution)
    
    slope_rad = np.sqrt(dz_dx**2 + dz_dy**2)
    slope_deg = np.arctan(slope_rad) * (180.0 / np.pi)
    return float(slope_deg)

def process_dataset(input_csv, output_csv, dem_path):
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Loading DEM from {dem_path}...")
    with rasterio.open(dem_path) as src:
        # approx meters per degree at equator
        meters_per_degree = 111320.0
        
        # Spatial resolution in degrees
        res_x, res_y = src.res
        resolution_m = (res_x + res_y) / 2.0 * meters_per_degree
        
        hand_metrics = []
        slopes = []
        
        print("Extracting HAND and Slope metrics...")
        for idx, row in df.iterrows():
            lon = row['longitude']
            lat = row['latitude']
            dist_to_river = row['distance_to_river_m']
            
            # Map lon/lat to pixel coordinates
            try:
                py, px = src.index(lon, lat)
                if px < 0 or px >= src.width or py < 0 or py >= src.height:
                    hand_metrics.append(np.nan)
                    slopes.append(np.nan)
                    continue
            except Exception:
                hand_metrics.append(np.nan)
                slopes.append(np.nan)
                continue
                
            # 1. Calculate HAND
            # Define window size based on distance to river (minimum 1 pixel, max 200 pixels to avoid massive memory usage)
            window_radius_pixels = int(max(1, min(200, dist_to_river / resolution_m)))
            
            # Make sure we don't go out of bounds
            w_min_x = max(0, px - window_radius_pixels)
            w_min_y = max(0, py - window_radius_pixels)
            w_max_x = min(src.width, px + window_radius_pixels + 1)
            w_max_y = min(src.height, py + window_radius_pixels + 1)
            
            width = w_max_x - w_min_x
            height = w_max_y - w_min_y
            
            window = Window(w_min_x, w_min_y, width, height)
            try:
                local_data = src.read(1, window=window)
                local_data = np.where(local_data < -500, np.nan, local_data) # mask out nodata
                
                if local_data.size == 0 or np.all(np.isnan(local_data)):
                    min_elev = row['elevation_m']
                else:
                    min_elev = np.nanmin(local_data)
                
                hand = max(0, row['elevation_m'] - min_elev)
                hand_metrics.append(hand)
            except Exception:
                hand_metrics.append(0)
            
            # 2. Calculate Slope
            # Get 3x3 window around the point
            s_min_x = max(0, px - 1)
            s_min_y = max(0, py - 1)
            s_width = min(src.width, px + 2) - s_min_x
            s_height = min(src.height, py + 2) - s_min_y
            
            s_window = Window(s_min_x, s_min_y, s_width, s_height)
            try:
                slope_data = src.read(1, window=s_window)
                slope_data = np.where(slope_data < -500, np.nan, slope_data)
                if slope_data.shape == (3,3) and not np.isnan(slope_data).any():
                    slp = calculate_slope(slope_data, resolution_m)
                    slopes.append(slp)
                else:
                    slopes.append(0.0)
            except Exception:
                slopes.append(0.0)
                
            if idx % 5000 == 0 and idx > 0:
                print(f"Processed {idx} rows...")

    df['hand_metric'] = hand_metrics
    df['slope_deg'] = slopes
    
    print(f"Saving enriched dataset to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print("Done!")

if __name__ == "__main__":
    os.makedirs('C:/KruthimaOps/data/dem', exist_ok=True)
    dem_file = 'C:/KruthimaOps/data/dem/srilanka_srtm.tif'
    
    # Note: If download_dem fails, we will need to fetch the file manually.
    try:
        download_dem(dem_file)
    except Exception as e:
        print(f"Error downloading DEM with elevation package: {e}")
        print("Please ensure GDAL and curl are available, or download SRTM manually.")
        
    if os.path.exists(dem_file):
        process_dataset('C:/KruthimaOps/data/train.csv', 'C:/KruthimaOps/data/train_v1000.csv', dem_file)
