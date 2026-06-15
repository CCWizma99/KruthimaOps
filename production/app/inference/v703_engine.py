"""
Inference Layer — v703 Engine
Loads all serialized artifacts from models/prod_v1/ at startup (once).
Implements: infer(features_dict: dict) -> float [0, 1]

Inference flow:
  1. Merge input with district baseline (from district_reference.json)
  2. Apply freq_maps → freq count features
  3. engineer_features() — same function as training
  4. Apply full-dataset TE maps
  5. Align & cast feature columns
  6. Run 6 production models (XGB × 2, CAT × 3, LGB × 1)
  7. Apply stacking weights → intercept
  8. Post-hoc power transform: a * pred^b + c
  9. clip(0, 1) → return float
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Runtime state ────────────────────────────────────────────────────
_ARTIFACTS:         Optional[Dict[str, Any]] = None
_TE_MAPS:           Optional[Dict]           = None
_FEATURE_LISTS:     Optional[Dict]           = None
_STACKER:           Optional[Dict]           = None
_POSTHOC:           Optional[Dict]           = None
_METADATA:          Optional[Dict]           = None
_DISTRICT_REF:      Optional[Dict]           = None

# Models
_XGB1 = _XGB2 = None
_CAT1 = _CAT2 = _CATRMSE = None
_LGB1 = None


def load_artifacts() -> None:
    """Load all serialized artifacts. Called once at FastAPI startup."""
    global _ARTIFACTS, _TE_MAPS, _FEATURE_LISTS, _STACKER, _POSTHOC, _METADATA
    global _DISTRICT_REF, _XGB1, _XGB2, _CAT1, _CAT2, _CATRMSE, _LGB1

    from app.config import MODEL_VERSION, MODELS_BASE_DIR, DISTRICT_REFERENCE_PATH

    base = os.path.join(MODELS_BASE_DIR, MODEL_VERSION)
    logger.info(f"[Inference] Loading artifacts from {base} ...")

    if not os.path.isdir(base):
        logger.error(
            f"Model directory '{base}' not found. "
            f"Run training/serialize_pipeline.py first."
        )
        return

    # Preprocessing
    with open(os.path.join(base, "preprocessing.pkl"), "rb") as f:
        _ARTIFACTS = pickle.load(f)

    # TE maps
    with open(os.path.join(base, "te_maps.pkl"), "rb") as f:
        _TE_MAPS = pickle.load(f)

    # Feature lists
    with open(os.path.join(base, "feature_lists.json")) as f:
        _FEATURE_LISTS = json.load(f)

    # Stacker
    with open(os.path.join(base, "stacker.json")) as f:
        _STACKER = json.load(f)

    # Post-hoc params
    with open(os.path.join(base, "posthoc.json")) as f:
        _POSTHOC = json.load(f)

    # Metadata
    with open(os.path.join(base, "model_metadata.json")) as f:
        _METADATA = json.load(f)

    # District reference
    if os.path.exists(DISTRICT_REFERENCE_PATH):
        with open(DISTRICT_REFERENCE_PATH) as f:
            _DISTRICT_REF = json.load(f)
        logger.info(f"[Inference] District reference: {len(_DISTRICT_REF)} districts.")

    # Models
    import xgboost as xgb
    import catboost as cb
    import lightgbm as lgb

    _XGB1 = xgb.XGBRegressor(); _XGB1.load_model(os.path.join(base, "xgb1.json"))
    _XGB2 = xgb.XGBRegressor(); _XGB2.load_model(os.path.join(base, "xgb2.json"))
    _CAT1 = cb.CatBoostRegressor(); _CAT1.load_model(os.path.join(base, "cat1.cbm"))
    _CAT2 = cb.CatBoostRegressor(); _CAT2.load_model(os.path.join(base, "cat2.cbm"))
    _CATRMSE = cb.CatBoostRegressor(); _CATRMSE.load_model(os.path.join(base, "catrmse.cbm"))
    _LGB1 = lgb.Booster(model_file=os.path.join(base, "lgb1.txt"))

    logger.info("[Inference] All 6 models loaded. Engine ready.")


def _is_ready() -> bool:
    return all(x is not None for x in [_ARTIFACTS, _TE_MAPS, _FEATURE_LISTS,
                                        _STACKER, _POSTHOC, _XGB1, _LGB1])


# ── Feature engineering (mirrors training) ───────────────────────────

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    a = _ARTIFACTS
    dist_elev_std         = a["district_elev_std"]
    lc_inund_mean         = a["landcover_mean_inundation"]
    comb_inund_mean       = a["combined_inundation_mean"]
    soil_map              = a["soil_infilt_map"]
    cyc_d                 = set(a["cyclone_districts"])
    wet_d                 = set(a["wet_zone_districts"])

    df = df.copy()
    df["confirmed_severe_risk"] = (
        (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes") &
        (df["is_good_to_live"].astype(str).str.strip().str.lower() == "no") &
        (df["inundation_area_sqm"] > 0)
    ).astype(int)
    df["no_flood_confirmed"] = (
        (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "no") &
        (df["is_good_to_live"].astype(str).str.strip().str.lower() == "yes")
    ).astype(int)
    df["inundation_per_capita"] = df["inundation_area_sqm"] / (
        np.expm1(df["population_density_per_km2_log1p"].fillna(0)) + 1.0
    )
    has_reason = (~df["reason_not_good_to_live"].astype(str).str.strip().str.lower().isin(
        ["nan", "none", "", "missing", "n/a"])).astype(int)
    df["downstream_risk_count"] = (
        (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes").astype(int) +
        (df["is_good_to_live"].astype(str).str.strip().str.lower() == "no").astype(int) +
        has_reason + (df["inundation_area_sqm"] > 0).astype(int)
    )
    df["downstream_sig"] = (
        df["flood_occurrence_current_event"].astype(str).str.strip() + "_" +
        df["is_good_to_live"].astype(str).str.strip() + "_" +
        df["reason_not_good_to_live"].astype(str).str.strip()
    )
    has_inundation = (df["inundation_area_sqm"] > 0).astype(int)
    df["downstream_quad_sig"] = (
        df["flood_occurrence_current_event"].astype(str).str.strip() + "_" +
        df["is_good_to_live"].astype(str).str.strip() + "_" +
        df["reason_not_good_to_live"].astype(str).str.strip() + "_" +
        has_inundation.astype(str)
    )
    date_series = pd.to_datetime(df["generation_date"].fillna("2024-06-15"))
    df["month"]   = date_series.dt.month
    df["is_yala"] = df["month"].isin([5, 6, 7, 8, 9]).astype(int)
    df["is_maha"] = df["month"].isin([11, 12, 1]).astype(int)
    df["zone_code"] = df["district"].astype(str).map(lambda x: 1 if x in wet_d else 2)
    df["monsoon_impact"] = (
        df["rainfall_7d_mm"] * df["is_yala"] * (df["zone_code"] == 1).astype(int) +
        df["rainfall_7d_mm"] * df["is_maha"] * (df["zone_code"] == 2).astype(int)
    )
    df["urban_runoff_potential"]   = df["rainfall_7d_mm"] * df["built_up_percent"].fillna(50) * (1.0 / (df["drainage_index"].fillna(0.5) + 1e-5))
    df["fluvial_risk_score_feat"]  = df["rainfall_7d_mm"] * (1.0 / (df["distance_to_river_m"].fillna(500) + 1.0))
    df["soil_infiltration"]        = df["soil_type"].astype(str).map(soil_map).fillna(0.4)
    df["soil_saturation_limit"]    = df["rainfall_7d_mm"] / (df["soil_infiltration"] + 0.1)
    df["pseudo_twi"]               = np.log1p((df["distance_to_river_m"].fillna(500) + 1.0) / (df["elevation_m"].fillna(10).clip(lower=0.0) + 1.0))
    df["flatness_index"]           = df["district"].astype(str).map(dist_elev_std).fillna(df["elevation_m"].std() if len(df) > 1 else 10.0)
    df["in_cyclone_path"]          = df["district"].astype(str).map(lambda x: 1 if x in cyc_d else 0)
    df["cyclone_vulnerability"]    = df["in_cyclone_path"] * df["extreme_weather_index"].fillna(0)
    df["slope_proxy"]              = df["elevation_m"].fillna(10) / (df["distance_to_river_m"].fillna(500) + 1.0)
    df["isolation_index"]          = np.log1p(df["nearest_hospital_km"].fillna(5)) + np.log1p(df["nearest_evac_km"].fillna(5))
    df["vulnerability"]            = df["isolation_index"] / (df["infrastructure_score"].fillna(3) + 1.0)
    df["elevation_divergence"]     = df["elevation_m"].fillna(10) - df["elevation_m_yeojohnson"].fillna(10)
    df["infra_deficit_sig"] = (
        df["water_supply"].astype(str).str.strip() + "_" +
        df["electricity"].astype(str).str.strip() + "_" +
        df["road_quality"].astype(str).str.strip()
    )
    df["inundation_area_log"]          = np.log1p(df["inundation_area_sqm"])
    df["flood_occurrence_yes"]         = (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes").astype(int)
    df["inundation_flood_interaction"]  = df["flood_occurrence_yes"] * df["inundation_area_log"]
    df["river_rain_interaction"]        = df["distance_to_river_m_log1p"].fillna(0) * df["rainfall_7d_mm_log1p"].fillna(0)
    df["river_monthly_exposure"]        = df["distance_to_river_m_log1p"].fillna(0) * df["monthly_rainfall_mm_log1p"].fillna(0)
    df["elev_rain_risk"]               = df["elevation_m_yeojohnson"].fillna(10) / (df["rainfall_7d_mm_log1p"].fillna(0) + 1e-6)
    df["water_signal"]                 = df["ndwi_qmap"].fillna(0).clip(lower=0)
    df["drainage_deficit"]             = (df["rainfall_7d_mm_log1p"].fillna(0) + 1) * (1.0 - df["drainage_index_yeojohnson"].fillna(0.5).clip(0, 1))
    df["infra_resilience"]             = df["infrastructure_score"].fillna(3) / (df["population_density_per_km2_log1p"].fillna(0) + 1e-6)
    df["evacuation_difficulty"]        = df["nearest_hospital_km_log1p"].fillna(0) + df["nearest_evac_km_log1p"].fillna(0)
    df["inundation_density_risk"]      = df["inundation_area_log"] / (df["population_density_per_km2_log1p"].fillna(0) + 1e-6)
    df["terrain_veg_risk"]             = df["terrain_roughness_index"].fillna(0) * (1.0 - df["ndvi_qmap"].fillna(0).clip(-1, 1))
    df["flood_pressure"]               = df["extreme_weather_index"].fillna(0) * df["seasonal_index"].fillna(0).clip(lower=0)
    df["is_repeat_flood_zone"]         = (df["historical_flood_count"].fillna(0) > 2).astype(int)
    df["rain_spike_ratio"]             = df["rainfall_7d_mm"] / (df["monthly_rainfall_mm"].fillna(100) + 1e-6)
    df["confirmed_risk"]               = (
        (df["flood_occurrence_current_event"].astype(str).str.strip().str.lower() == "yes") &
        (df["is_good_to_live"].astype(str).str.strip().str.lower() == "no")
    ).astype(int)
    df["landcover_mean_inundation_val"] = df["landcover"].astype(str).map(lc_inund_mean).fillna(comb_inund_mean)
    df["inundation_ratio"]              = df["inundation_area_sqm"] / (df["landcover_mean_inundation_val"] + 1.0)
    ndwi_clip = df["ndwi_qmap"].fillna(0).clip(lower=0.0)
    ndvi_clip = df["ndvi_qmap"].fillna(0).clip(-1.0, 1.0).clip(lower=0.0)
    df["pooling_vulnerability"]         = ndwi_clip * (1.0 - ndvi_clip)
    df["soil_drainage_saturation"]      = df["soil_saturation_limit"] * (1.0 - df["drainage_index_yeojohnson"].fillna(0.5).clip(0.0, 1.0))

    lat = df["latitude"].fillna(7.8731)
    lon = df["longitude"].fillna(80.7718)
    df["grid_id_100"] = (lat / 1.0).astype(int).astype(str) + "_" + (lon / 1.0).astype(int).astype(str)
    df["grid_id_050"] = (lat / 0.5).astype(int).astype(str) + "_" + (lon / 0.5).astype(int).astype(str)
    df["grid_id_025"] = (lat / 0.25).astype(int).astype(str) + "_" + (lon / 0.25).astype(int).astype(str)
    df["grid_id_012"] = (lat / 0.125).astype(int).astype(str) + "_" + (lon / 0.125).astype(int).astype(str)
    df["lat_bin"]     = (lat / 0.5).astype(int)
    df["lon_bin"]     = (lon / 0.5).astype(int)
    df["grid_id"]     = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)

    df = df.drop(columns=["inundation_area_sqm", "landcover_mean_inundation_val"], errors="ignore")
    return df


# ── TE map application ───────────────────────────────────────────────

def _apply_te_maps(df: pd.DataFrame) -> pd.DataFrame:
    gs = _ARTIFACTS["global_stats"]
    fl = _FEATURE_LISTS
    SMOOTHING_COMPOSITE = fl["SMOOTHING_COMPOSITE"]

    df = df.copy()
    all_te_cols = fl["TARGET_ENC_COLS"] + fl["COMPOSITE_ENC_COLS"]

    for col in all_te_cols:
        if col not in _TE_MAPS or col not in df.columns:
            continue
        maps    = _TE_MAPS[col]
        col_str = df[col].astype(str)
        is_composite = col in fl["COMPOSITE_ENC_COLS"]

        df[f"{col}_target_enc"] = col_str.map(maps["smoothed_median"]).fillna(gs["GLOBAL_MEDIAN"]).astype(float)
        if not is_composite:
            df[f"{col}_target_q25"] = col_str.map(maps["smoothed_q25"]).fillna(gs["GLOBAL_Q25"]).astype(float)
            df[f"{col}_target_q75"] = col_str.map(maps["smoothed_q75"]).fillna(gs["GLOBAL_Q75"]).astype(float)
            df[f"{col}_target_cnt"] = col_str.map(maps["log_count"]).fillna(0.0).astype(float)
        if col in fl["STD_ENC_COLS"]:
            df[f"{col}_target_std"] = col_str.map(maps["smoothed_std"]).fillna(gs["GLOBAL_STD"]).astype(float)

    return df


# ── dtype alignment ──────────────────────────────────────────────────

def _align_dtypes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return three views: XGB (label-encoded ints), CAT (strings), LGB (pandas cat)."""
    cat_dtype_map     = _ARTIFACTS["cat_dtype_map"]
    cat_feature_names = _FEATURE_LISTS["cat_feature_names"]

    df_lgb = df.copy()
    df_cat = df.copy()
    df_xgb = df.copy()

    for col in cat_feature_names:
        if col not in df.columns:
            continue
        cats = cat_dtype_map[col]
        cdt  = pd.CategoricalDtype(categories=cats, ordered=False)

        # LGB: pandas categorical
        df_lgb[col] = df_lgb[col].astype(str).astype(cdt)

        # CAT: plain strings
        df_cat[col] = df_cat[col].astype(str)

        # XGB: integer codes
        df_xgb[col] = df_xgb[col].astype(str).astype(cdt).cat.codes.astype("int32")

    return df_xgb, df_cat, df_lgb


# ── Main inference function ──────────────────────────────────────────

def infer(features_dict: dict) -> float:
    """
    Run v703 inference on a single feature dict.
    Returns a float in [0, 1].
    """
    if not _is_ready():
        raise RuntimeError("Inference engine not initialised. Call load_artifacts() first.")

    # Copy input payload to prevent mutating caller state
    features_dict = dict(features_dict)

    # Normalize category values to match training set vocabulary
    reason_map = {
        "none": "None",
        "flood risk": "High flood risk",
        "water contamination": "Other",
        "infrastructure damage": "Poor infrastructure",
        "landslide risk": "Other"
    }
    raw_reason = str(features_dict.get("reason_not_good_to_live", "None")).strip()
    if raw_reason.lower() in reason_map:
        features_dict["reason_not_good_to_live"] = reason_map[raw_reason.lower()]

    a  = _ARTIFACTS
    fl = _FEATURE_LISTS

    # 1. Merge with district baseline
    district = features_dict.get("district", "")
    base_row: Dict[str, Any] = {}
    if _DISTRICT_REF and district in _DISTRICT_REF:
        base_row = dict(_DISTRICT_REF[district])

    # Override with user inputs
    base_row.update({k: v for k, v in features_dict.items() if v is not None})

    # Compute derived log1p transforms for dynamic inputs
    rainfall = float(base_row.get("rainfall_7d_mm", 0))
    base_row["rainfall_7d_mm_log1p"] = float(np.log1p(rainfall))

    inundation = float(base_row.get("inundation_area_sqm", 0))

    # Decimal precision features
    lat = base_row.get("latitude", 7.8731)
    lon = base_row.get("longitude", 80.7718)
    base_row["lat_decimal_len"] = len(str(lat).split('.')[1]) if '.' in str(lat) else 0
    base_row["lon_decimal_len"] = len(str(lon).split('.')[1]) if '.' in str(lon) else 0

    # 2. Apply freq_maps (0 for unseen values — correct by training convention)
    for col in a["freq_cols"]:
        freq_map = a["freq_maps"].get(col, {})
        raw_val  = base_row.get(col)
        base_row[f"{col}_freq"] = float(freq_map.get(raw_val, 0))

    # Ensure inundation_area_sqm is available for engineer_features
    base_row["inundation_area_sqm"] = inundation

    # 3. Build single-row DataFrame
    df = pd.DataFrame([base_row])

    # 4. Engineer features
    df = _engineer_features(df)

    # 5. Apply full-dataset TE maps
    df = _apply_te_maps(df)

    # 6. Align to model feature list
    FEATURES = fl["FEATURES"]
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    df = df[FEATURES]

    # Numeric fillna
    cat_feature_names = fl["cat_feature_names"]
    for col in df.columns:
        if col not in cat_feature_names:
            if df[col].dtype in ["float64", "int64", "float32", "int32"]:
                df[col] = df[col].fillna(0.0)

    # 7. Format for each model family
    X_xgb, X_cat, X_lgb = _align_dtypes(df)

    # 8. Get predictions from each model
    cat_cols = [c for c in cat_feature_names if c in X_cat.columns]
    p_xgb1   = float(_XGB1.predict(X_xgb)[0])
    p_cat1   = float(_CAT1.predict(X_cat)[0])
    p_cat2   = float(_CAT2.predict(X_cat)[0])
    p_catrmse = float(_CATRMSE.predict(X_cat)[0])
    p_lgb1   = float(_LGB1.predict(X_lgb)[0])
    p_xgb2   = float(_XGB2.predict(X_xgb)[0])

    # 9. Stack
    preds  = np.array([p_xgb1, p_cat1, p_cat2, p_catrmse, p_lgb1, p_xgb2])
    w      = np.array(_STACKER["weights"])
    b      = float(_STACKER["bias"])
    stacked = float(np.clip(np.dot(preds, w) + b, 0.0, 1.0))

    # 10. Post-hoc transform
    ph  = _POSTHOC
    out = float(ph["a"] * (max(stacked, 1e-6) ** ph["b"]) + ph["c"])
    raw_score = float(np.clip(out, 0.0, 1.0))

    # 11. Web Dashboard Calibration
    # Align score with physical/meteorological expectations (rain/flooding -> high risk)
    rain = float(features_dict.get("rainfall_7d_mm", 0.0))
    inund = float(features_dict.get("inundation_area_sqm", 0.0))
    flood_occurrence = str(features_dict.get("flood_occurrence_current_event", "No")).strip().lower()
    is_good_to_live = str(features_dict.get("is_good_to_live", "Yes")).strip().lower()

    # Calculate Physical Risk Index (PRI)
    R = min(rain / 300.0, 1.0) if rain > 0.0 else 0.0
    I = min(inund / 25000.0, 1.0) if inund > 0.0 else 0.0
    F = 1.0 if flood_occurrence == "yes" else 0.0
    U = 1.0 if is_good_to_live == "no" else 0.0
    pri = 0.3 * R + 0.3 * I + 0.2 * F + 0.2 * U

    # Invert and scale the model's raw score: 0.58 is safe dry baseline, 0.38 is extreme wet baseline
    raw_risk = (0.58 - raw_score) / (0.58 - 0.38)
    raw_risk = float(np.clip(raw_risk, 0.0, 1.0))

    # Blend model-derived risk and physical parameters (60% ML, 40% Physical)
    blended = 0.6 * raw_risk + 0.4 * pri
    calibrated = 0.05 + blended * 0.90

    return float(np.clip(calibrated, 0.02, 0.99))


def get_model_metadata() -> Dict[str, Any]:
    """Return model registry metadata for the /api/models endpoint."""
    if _METADATA is None:
        return {"status": "not_loaded"}
    return dict(_METADATA)


def get_district_reference() -> Dict[str, Any]:
    """Return the district reference lookup for the /api/district endpoint."""
    if _DISTRICT_REF is None:
        return {}
    return dict(_DISTRICT_REF)
