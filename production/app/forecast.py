import logging
import urllib.request
import json
from datetime import datetime, date, timedelta
from typing import List, Dict, Any

from app.inference import get_district_reference, infer
from app.monitoring import get_cached_forecast, save_cached_forecast

logger = logging.getLogger(__name__)


def get_district_forecast(district: str) -> List[Dict[str, Any]]:
    """
    Get 7-day risk outlook for a district (today + next 6 days).
    Uses SQLite cache if calculated today, otherwise fetches from Open-Meteo
    and calculates risk scores via v703 model.

    Rolling 7-day rainfall window:
      - For each target date T, we sum precipitation from [T-7 .. T-1].
      - Past days come from Open-Meteo `past_days` observations (actual readings).
      - Future days come from Open-Meteo `forecast_days` values.
    """
    today = date.today()
    today_str = today.isoformat()

    # Target dates: [today, today+1, ..., today+6]
    target_dates = [(today + timedelta(days=i)).isoformat() for i in range(7)]

    # 1. Try to read ALL 7 slots from today's cache first
    forecasts = []
    all_cached = True
    for t_date in target_dates:
        cached = get_cached_forecast(district, t_date)
        if cached and cached["calculation_date"] == today_str:
            forecasts.append({
                "date":           cached["forecast_date"],
                "rainfall_7d_mm": cached["rainfall_7d_mm"],
                "risk_score":     cached["risk_score"],
                "risk_level":     cached["risk_level"],
                "cached":         True,
            })
        else:
            all_cached = False
            break

    if all_cached:
        logger.info(f"[Forecast] Full cache hit for {district}.")
        return forecasts

    # 2. Cache miss — fetch from Open-Meteo
    logger.info(f"[Forecast] Cache miss for {district}. Fetching weather data...")
    ref = get_district_reference()
    if district not in ref:
        raise ValueError(f"District '{district}' not found in reference.")

    district_info = ref[district]
    lat = district_info.get("center_lat", district_info.get("latitude", 7.87))
    lon = district_info.get("center_lon", district_info.get("longitude", 80.77))

    # We need at minimum 7 past days (for today's lookback) up to 13 past days
    # (for today+6's lookback which goes [today-1 .. today+5]).
    # past_days=7 covers the actual observed days; forecast_days=7 covers future.
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum"
        f"&past_days=7&forecast_days=7"
        f"&timezone=Asia%2FColombo"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FloodGuardSL-Predictor/1.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            res_data = json.loads(response.read().decode())
        daily_time   = res_data["daily"]["time"]
        daily_precip = res_data["daily"]["precipitation_sum"]
        logger.info(f"[Forecast] Received {len(daily_time)} days of precipitation from Open-Meteo.")
    except Exception as e:
        logger.warning(
            f"[Forecast] Open-Meteo call failed for {district}: {e}. "
            f"Falling back to district baseline."
        )
        # Flat-spread fallback using district baseline
        fallback_rain = float(district_info.get("rainfall_7d_mm", 50.0))
        daily_time   = [
            (today + timedelta(days=i - 7)).isoformat() for i in range(14)
        ]
        daily_precip = [fallback_rain / 7.0] * 14

    # Build date → daily precipitation map
    precip_map: Dict[str, float] = {}
    for t, p in zip(daily_time, daily_precip):
        precip_map[t] = float(p) if p is not None else 0.0

    logger.debug(f"[Forecast] Precip map range: {min(precip_map)} → {max(precip_map)}")

    # 3. Compute risk for each target date
    forecasts = []
    for t_date in target_dates:
        t_dt = datetime.strptime(t_date, "%Y-%m-%d").date()

        # Rolling sum: preceding 7 days [T-7 .. T-1] (NOT including T itself)
        sum_7d = 0.0
        for offset in range(1, 8):
            prev = (t_dt - timedelta(days=offset)).isoformat()
            sum_7d += precip_map.get(prev, 0.0)

        # Inject generation_date = forecast target date so seasonal features
        # (is_yala, is_maha, month) are computed correctly for THAT date,
        # not defaulting to the June 2024 fallback in the engine.
        payload = {
            "district":                       district,
            "rainfall_7d_mm":                 sum_7d,
            "inundation_area_sqm":            0.0,   # baseline: no active inundation
            "flood_occurrence_current_event": "No",  # baseline
            "is_good_to_live":                "Yes", # baseline
            "reason_not_good_to_live":        "None",
            "generation_date":                t_date, # ← critical: drives seasonal features
        }

        score = infer(payload)

        if score < 0.25:
            level = "LOW"
        elif score < 0.50:
            level = "MEDIUM"
        elif score < 0.75:
            level = "HIGH"
        else:
            level = "EXTREME"

        save_cached_forecast(
            district=district,
            forecast_date=t_date,
            calculation_date=today_str,
            rainfall_7d_mm=sum_7d,
            risk_score=score,
            risk_level=level,
        )

        forecasts.append({
            "date":           t_date,
            "rainfall_7d_mm": round(sum_7d, 2),
            "risk_score":     round(score, 6),
            "risk_level":     level,
            "cached":         False,
        })

        logger.info(
            f"[Forecast] {district} {t_date}: "
            f"rain={sum_7d:.1f}mm score={score:.4f} level={level}"
        )

    return forecasts


def get_historical_forecast(district: str, target_date_str: str) -> Dict[str, Any]:
    """
    Run a historical simulation for a single district on a specific past date.
    Fetches ACTUAL observed weather data from Open-Meteo's archive API
    and calculates the flood risk score using the v703 model.

    This lets us backtest model predictions against actual past weather conditions.
    """
    from datetime import datetime

    target = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    today = date.today()

    if target >= today:
        raise ValueError(f"Historical simulation requires a past date. Got: {target_date_str}")

    ref = get_district_reference()
    if district not in ref:
        raise ValueError(f"District '{district}' not found in reference.")

    district_info = ref[district]
    lat = district_info.get("center_lat", district_info.get("latitude", 7.87))
    lon = district_info.get("center_lon", district_info.get("longitude", 80.77))

    # We need precipitation for [target-7 .. target-1] → fetch a 14-day window
    fetch_start = (target - timedelta(days=10)).isoformat()
    fetch_end   = (target + timedelta(days=3)).isoformat()

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum"
        f"&start_date={fetch_start}&end_date={fetch_end}"
        f"&timezone=Asia%2FColombo"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FloodGuardSL-Historical/1.0"})
        with urllib.request.urlopen(req, timeout=12) as response:
            res_data = json.loads(response.read().decode())
        daily_time   = res_data["daily"]["time"]
        daily_precip = res_data["daily"]["precipitation_sum"]
        logger.info(
            f"[Historical] Archive API returned {len(daily_time)} days "
            f"for {district} around {target_date_str}."
        )
    except Exception as e:
        logger.warning(f"[Historical] Archive API failed for {district}: {e}")
        # Fallback: use district baseline rainfall
        fallback_rain = float(district_info.get("rainfall_7d_mm", 50.0))
        daily_time   = [
            (target + timedelta(days=i - 10)).isoformat() for i in range(14)
        ]
        daily_precip = [fallback_rain / 7.0] * 14

    # Build precipitation map
    precip_map: Dict[str, float] = {}
    for t, p in zip(daily_time, daily_precip):
        precip_map[t] = float(p) if p is not None else 0.0

    # Rolling 7-day sum: [T-7 .. T-1]
    t_dt = target
    sum_7d = 0.0
    for offset in range(1, 8):
        prev = (t_dt - timedelta(days=offset)).isoformat()
        sum_7d += precip_map.get(prev, 0.0)

    # Run inference
    payload = {
        "district":                       district,
        "rainfall_7d_mm":                 sum_7d,
        "inundation_area_sqm":            0.0,
        "flood_occurrence_current_event": "No",
        "is_good_to_live":                "Yes",
        "reason_not_good_to_live":        "None",
        "generation_date":                target_date_str,
    }

    score = infer(payload)

    if score < 0.25:
        level = "LOW"
    elif score < 0.50:
        level = "MEDIUM"
    elif score < 0.75:
        level = "HIGH"
    else:
        level = "EXTREME"

    logger.info(
        f"[Historical] {district} {target_date_str}: "
        f"rain={sum_7d:.1f}mm score={score:.4f} level={level}"
    )

    return {
        "district":       district,
        "date":           target_date_str,
        "rainfall_7d_mm": round(sum_7d, 2),
        "risk_score":     round(score, 6),
        "risk_level":     level,
        "source":         "archive",
    }
