import json
import logging
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

logger = logging.getLogger(__name__)

_XGB = None
_LGB = None
_CAT = None
_FEATURE_INFO = None
_METADATA = None
_DEM_PATH = "C:/KruthimaOps/data/dem/srilanka_srtm.tif"

def load_artifacts() -> None:
    """Load the v1000 models and feature mappings."""
    global _XGB, _LGB, _CAT, _FEATURE_INFO, _METADATA
    
    base = "C:/KruthimaOps/production/models/prod_v1000"
    logger.info(f"[Inference] Loading v1000 artifacts from {base} ...")

    if not os.path.isdir(base):
        logger.error(f"Model directory '{base}' not found. Run serialize_pipeline.py first.")
        return

    # Load feature lists and medians
    with open(os.path.join(base, "feature_info.json"), "r") as f:
        _FEATURE_INFO = json.load(f)

    # Load metadata
    with open(os.path.join(base, "model_metadata.json"), "r") as f:
        _METADATA = json.load(f)

    # Load Models
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    
    _XGB = xgb.XGBRegressor()
    _XGB.load_model(os.path.join(base, "xgb.json"))
    
    _LGB = lgb.Booster(model_file=os.path.join(base, "lgb.txt"))
    
    _CAT = cb.CatBoostRegressor()
    _CAT.load_model(os.path.join(base, "cat.cbm"))
    
    logger.info("[Inference] v1000 models (XGB, LGB, CAT) loaded successfully.")

def _calculate_slope(data, resolution):
    if data.shape != (3, 3): return 0.0
    dz_dx = ((data[2, 0] + 2*data[2, 1] + data[2, 2]) - (data[0, 0] + 2*data[0, 1] + data[0, 2])) / (8 * resolution)
    dz_dy = ((data[0, 2] + 2*data[1, 2] + data[2, 2]) - (data[0, 0] + 2*data[1, 0] + data[2, 0])) / (8 * resolution)
    slope_rad = np.sqrt(dz_dx**2 + dz_dy**2)
    return float(np.arctan(slope_rad) * (180.0 / np.pi))

def _get_topography_metrics(lat, lon, dist_to_river, base_elevation):
    """Dynamically calculates HAND and Slope from the SRTM DEM."""
    if not os.path.exists(_DEM_PATH):
        return 0.0, 0.0 # fallback if DEM missing
        
    try:
        with rasterio.open(_DEM_PATH) as src:
            py, px = src.index(lon, lat)
            if px < 0 or px >= src.width or py < 0 or py >= src.height:
                return 0.0, 0.0
                
            # Resolution in meters
            res_m = ((src.res[0] + src.res[1]) / 2.0) * 111320.0
            
            # 1. Slope
            s_min_x, s_min_y = max(0, px - 1), max(0, py - 1)
            s_window = Window(s_min_x, s_min_y, 3, 3)
            slope_data = src.read(1, window=s_window)
            slope_data = np.where(slope_data < -500, np.nan, slope_data)
            slope = _calculate_slope(slope_data, res_m) if slope_data.shape == (3,3) else 0.0
            
            # 2. HAND
            radius = int(max(1, min(200, dist_to_river / res_m)))
            w_min_x, w_min_y = max(0, px - radius), max(0, py - radius)
            w_max_x, w_max_y = min(src.width, px + radius + 1), min(src.height, py + radius + 1)
            
            window = Window(w_min_x, w_min_y, w_max_x - w_min_x, w_max_y - w_min_y)
            local_data = src.read(1, window=window)
            local_data = np.where(local_data < -500, np.nan, local_data)
            
            min_elev = np.nanmin(local_data) if local_data.size > 0 and not np.all(np.isnan(local_data)) else base_elevation
            hand = max(0, base_elevation - min_elev)
            
            return hand, slope
    except Exception:
        return 0.0, 0.0

def infer(features_dict: dict) -> float:
    if _XGB is None or _LGB is None or _CAT is None:
        raise RuntimeError("v1000 Inference engine not initialised. Call load_artifacts() first.")

    row = dict(features_dict)
    
    # 1. Extract dynamic Topography features
    lat = float(row.get("latitude", 7.8731))
    lon = float(row.get("longitude", 80.7718))
    dist = float(row.get("distance_to_river_m", 500.0))
    elev = float(row.get("elevation_m", 10.0))
    
    hand, slope = _get_topography_metrics(lat, lon, dist, elev)
    row["hand_metric"] = hand
    row["slope_deg"] = slope

    # 2. Fill missing values based on medians from training
    medians = _FEATURE_INFO["medians"]
    features_list = _FEATURE_INFO["features"]
    cat_cols = _FEATURE_INFO["cat_cols"]
    
    df = pd.DataFrame([row])
    
    for col in features_list:
        if col not in df.columns:
            df[col] = medians.get(col, 0.0)
        else:
            if pd.isna(df.loc[0, col]):
                df.loc[0, col] = medians.get(col, 0.0)
                
        # Cast categorical columns to strings for LGBM / XGB / CatBoost with training categories alignment
        if col in cat_cols:
            valid_cats = _FEATURE_INFO.get("categories", {}).get(col, [])
            val = str(df.loc[0, col])
            if val not in valid_cats:
                df.loc[0, col] = "missing" if "missing" in valid_cats else np.nan
            
            cdt = pd.CategoricalDtype(categories=valid_cats, ordered=False)
            df[col] = df[col].astype(cdt)

    # 3. Model Alignment
    X = df[features_list]

    # 4. Predict
    p_xgb = float(_XGB.predict(X)[0])
    p_lgb = float(_LGB.predict(X)[0])
    p_cat = float(_CAT.predict(X)[0])
    
    # Average and clip (1/3 weight each)
    raw_score = float(np.clip((p_xgb + p_lgb + p_cat) / 3.0, 0.0, 1.0))

    # 5. Dashboard Calibration (Physical Index)
    rain = float(row.get("rainfall_7d_mm", 0.0))
    inund = float(row.get("inundation_area_sqm", 0.0))
    flood = str(row.get("flood_occurrence_current_event", "No")).strip().lower()
    live = str(row.get("is_good_to_live", "Yes")).strip().lower()

    R = min(rain / 300.0, 1.0)
    I = min(inund / 25000.0, 1.0)
    F = 1.0 if flood == "yes" else 0.0
    U = 1.0 if live == "no" else 0.0
    pri = 0.3 * R + 0.3 * I + 0.2 * F + 0.2 * U

    # Blend 60% ML, 40% Physical rules to ensure web dashboard acts logically
    calibrated = 0.05 + ((0.6 * raw_score) + (0.4 * pri)) * 0.90
    return float(np.clip(calibrated, 0.02, 0.99))

def get_model_metadata() -> dict:
    if _METADATA is None:
        return {"status": "not_loaded"}
    return dict(_METADATA)

def get_district_reference() -> dict:
    # Read the same baseline reference JSON used by v703
    try:
        with open("C:/KruthimaOps/production/data/district_reference.json", "r") as f:
            return json.load(f)
    except:
        return {}
