import json
import logging
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from app.config import MODEL_VERSION, MODELS_BASE_DIR, DISTRICT_REFERENCE_PATH
import app.inference.geospatial_mapper as geospatial_mapper

logger = logging.getLogger(__name__)

_MODELS = {}
_FEATURE_INFO = None
_DEM_PATH = "C:/KruthimaOps/data/dem/srilanka_srtm.tif"
_STATION_COORDS = {}
_OFFLINE_GAUGES = {}

def load_artifacts() -> None:
    global _MODELS, _FEATURE_INFO, _STATION_COORDS, _OFFLINE_GAUGES
    
    try:
        with open('c:/KruthimaOps/data/station_coords.json') as f:
            _STATION_COORDS = json.load(f)
        with open('c:/KruthimaOps/data/offline_river_gauges.json') as f:
            _OFFLINE_GAUGES = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load offline gauges: {e}")
    
    base = os.path.join(MODELS_BASE_DIR, MODEL_VERSION)
    logger.info(f"[Inference] Loading lightweight {MODEL_VERSION} artifacts from {base} ...")

    if not os.path.isdir(base):
        logger.error(f"Model directory '{base}' not found. Run serialize_pipeline.py first.")
        return

    try:
        with open(os.path.join(base, "feature_info.json"), "r") as f:
            _FEATURE_INFO = json.load(f)
            
        _MODELS['xgb'] = xgb.XGBRegressor()
        _MODELS['xgb'].load_model(os.path.join(base, "xgb.json"))
        
        _MODELS['cat'] = cb.CatBoostRegressor()
        _MODELS['cat'].load_model(os.path.join(base, "cat.cbm"))
        
        _MODELS['lgb'] = lgb.Booster(model_file=os.path.join(base, "lgb.txt"))

        logger.info("[Inference] Lightweight v1000 models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")

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
                
            res_m = ((src.res[0] + src.res[1]) / 2.0) * 111320.0
            
            s_min_x, s_min_y = max(0, px - 1), max(0, py - 1)
            s_window = Window(s_min_x, s_min_y, 3, 3)
            slope_data = src.read(1, window=s_window)
            slope_data = np.where(slope_data < -500, np.nan, slope_data)
            slope = _calculate_slope(slope_data, res_m) if slope_data.shape == (3,3) else 0.0
            
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


def to_cat_fmt_local(df, cat_cols):
    df = df.copy()
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def infer(features_dict: dict) -> tuple[float, float]:
    if not _MODELS or not _FEATURE_INFO:
        raise RuntimeError("v1000 Inference engine not initialised. Call load_artifacts() first.")

    row = dict(features_dict)
    
    # 0. Historic river gauge integration
    if "generation_date" in row and row["generation_date"]:
        date_str = str(row["generation_date"]).split("T")[0]
        if date_str in _OFFLINE_GAUGES:
            # Fallbacks for lat/lon if not provided
            meds = _FEATURE_INFO["medians"] if _FEATURE_INFO else {}
            lat = float(row.get("latitude") if row.get("latitude") is not None else meds.get("latitude", 7.8731))
            lon = float(row.get("longitude") if row.get("longitude") is not None else meds.get("longitude", 80.7718))
            
            gauge_dict = _OFFLINE_GAUGES[date_str]
            station, dist, g_info = geospatial_mapper.find_closest_gauge(lat, lon, gauge_dict, _STATION_COORDS)
            
            if g_info:
                row["nearest_gauge_distance_km"] = dist
                row["gauge_water_level_m"] = g_info['water_level']
                row["gauge_flood_ratio"] = g_info['water_level'] / (g_info['minor_flood'] + 0.001)
                

    # 1. Fill missing features with medians
    features = _FEATURE_INFO["features"]
    cat_cols = _FEATURE_INFO["cat_cols"]
    medians = _FEATURE_INFO["medians"]
    cat_dtype_map = _FEATURE_INFO["categories"]
    
    # Dynamic calculation for HAND and Slope if possible
    if "hand_metric" not in row or "slope_deg" not in row:
        lat = float(row.get("latitude") if row.get("latitude") is not None else medians.get("latitude", 7.8731))
        lon = float(row.get("longitude") if row.get("longitude") is not None else medians.get("longitude", 80.7718))
        dist = float(row.get("distance_to_river_m") if row.get("distance_to_river_m") is not None else medians.get("distance_to_river_m", 500.0))
        elev = float(row.get("elevation_m") if row.get("elevation_m") is not None else medians.get("elevation_m", 10.0))
        
        hand, slope = _get_topography_metrics(lat, lon, dist, elev)
        if "hand_metric" not in row: row["hand_metric"] = hand
        if "slope_deg" not in row: row["slope_deg"] = slope

    for col in features:
        if col not in row or pd.isna(row[col]) or row[col] is None:
            row[col] = medians.get(col, "missing" if col in cat_cols else 0.0)

    df = pd.DataFrame([row])
    
    # 2. Filter to exact columns and types
    df = df[features].copy()
    
    for col in cat_cols:
        df[col] = df[col].astype(str)
        all_vals = cat_dtype_map.get(col, [])
        val = df.loc[0, col]
        if val not in all_vals:
            df.loc[0, col] = "missing" if "missing" in all_vals else (all_vals[0] if all_vals else "missing")
            
        cdt = pd.CategoricalDtype(categories=all_vals, ordered=False)
        df[col] = df[col].astype(cdt)

    # 3. Format for each model family
    X_xgb = df.copy()
    X_cat = to_cat_fmt_local(df, cat_cols)
    X_lgb = df.copy()

    # 4. Predictions
    try:
        p_xgb = float(_MODELS['xgb'].predict(X_xgb)[0])
        p_cat = float(_MODELS['cat'].predict(X_cat)[0])
        p_lgb = float(_MODELS['lgb'].predict(X_lgb)[0])
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise e

    # 5. Average and calibrate
    preds = np.array([p_xgb, p_cat, p_lgb])
    raw_score = float(np.clip(np.mean(preds), 0.0, 1.0))
    variance = float(np.var(preds))

    # 6. Physical Risk Calibration (from v703)
    rain = float(features_dict.get("rainfall_7d_mm", 0.0))
    inund = float(features_dict.get("inundation_area_sqm", 0.0))
    flood_occurrence = str(features_dict.get("flood_occurrence_current_event", "No")).strip().lower()
    is_good_to_live = str(features_dict.get("is_good_to_live", "Yes")).strip().lower()

    R = min(rain / 300.0, 1.0) if rain > 0.0 else 0.0
    I = min(inund / 25000.0, 1.0) if inund > 0.0 else 0.0
    F = 1.0 if flood_occurrence == "yes" else 0.0
    U = 1.0 if is_good_to_live == "no" else 0.0
    pri = 0.3 * R + 0.3 * I + 0.2 * F + 0.2 * U

    raw_risk = (0.58 - raw_score) / (0.58 - 0.38)
    raw_risk = float(np.clip(raw_risk, 0.0, 1.0))

    blended = 0.6 * raw_risk + 0.4 * pri
    calibrated = 0.05 + blended * 0.90

    return float(np.clip(calibrated, 0.02, 0.99)), variance

def get_model_metadata() -> dict:
    try:
        base = os.path.join(MODELS_BASE_DIR, MODEL_VERSION)
        with open(os.path.join(base, "model_metadata.json"), "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load model metadata: {e}")
        return {"status": "not_loaded"}

def get_district_reference() -> dict:
    try:
        with open(DISTRICT_REFERENCE_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load district reference: {e}")
        return {}
