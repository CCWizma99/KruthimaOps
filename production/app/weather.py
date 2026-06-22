"""
Weather Data Layer — Multi-Point Sampling Engine
=================================================
Fetches rainfall data from Open-Meteo using multiple sample points
per district (Center, N, S, E, W) and returns the MAXIMUM 7-day
cumulative rainfall across all points.

Supports:
  - Historical dates → Open-Meteo Archive API
  - Future dates     → Open-Meteo Forecast API
  - Today            → Open-Meteo Forecast API (with past_days)
"""
import logging
import urllib.request
import json
import ssl
from datetime import date, timedelta, datetime
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

# ── District Multi-Point Sample Coordinates ──────────────────────────
# For each district: [center, north, south, east, west]
# These approximate the geographic spread of the district boundary
# to capture localized rainfall extremes that a single center point misses.

DISTRICT_SAMPLE_POINTS: Dict[str, List[Tuple[float, float]]] = {
    "Ampara": [
        (7.2959, 81.6750),   # Center
        (7.5500, 81.6200),   # North
        (6.9500, 81.7800),   # South
        (7.3000, 81.8500),   # East (coastal)
        (7.3000, 81.3500),   # West (inland)
    ],
    "Anuradhapura": [
        (8.3114, 80.4037),   # Center
        (8.7000, 80.4000),   # North (Medawachchiya)
        (7.9500, 80.5000),   # South (Kekirawa)
        (8.3000, 80.7500),   # East (Padaviya)
        (8.2000, 80.1000),   # West
    ],
    "Badulla": [
        (6.9934, 81.0550),   # Center
        (7.2500, 81.0000),   # North (Mahiyanganaya)
        (6.7500, 81.0500),   # South (Ella/Wellawaya)
        (7.0000, 81.2500),   # East
        (6.9500, 80.8500),   # West (Bandarawela)
    ],
    "Batticaloa": [
        (7.7310, 81.6747),   # Center
        (8.0000, 81.6000),   # North (Vakarai)
        (7.4500, 81.7500),   # South (Kalmunai side)
        (7.7300, 81.8500),   # East (coastal)
        (7.7000, 81.4500),   # West (interior)
    ],
    "Colombo": [
        (6.9271, 79.8612),   # Center
        (7.0000, 79.9000),   # North (Kaduwela)
        (6.8200, 79.8700),   # South (Moratuwa)
        (6.9300, 80.0500),   # East (Avissawella edge)
        (6.9200, 79.8200),   # West (coastal)
    ],
    "Galle": [
        (6.0535, 80.2210),   # Center
        (6.2000, 80.2500),   # North (Elpitiya)
        (5.9500, 80.2000),   # South (coastal)
        (6.0500, 80.4500),   # East (Imaduwa)
        (6.0500, 80.1000),   # West (Hikkaduwa)
    ],
    "Gampaha": [
        (7.0840, 80.0098),   # Center
        (7.2500, 80.0000),   # North (Minuwangoda)
        (6.9800, 79.9500),   # South (Wattala)
        (7.1000, 80.2000),   # East (Attanagalla)
        (7.0500, 79.8500),   # West (Negombo coast)
    ],
    "Hambantota": [
        (6.1241, 81.1185),   # Center
        (6.3000, 81.0000),   # North
        (6.0500, 81.1000),   # South (coastal)
        (6.1500, 81.4000),   # East (Yala side)
        (6.1000, 80.8500),   # West (Tangalle)
    ],
    "Jaffna": [
        (9.6615, 80.0255),   # Center
        (9.7500, 80.0000),   # North (Point Pedro)
        (9.5500, 80.0500),   # South (Chavakachcheri)
        (9.6500, 80.2000),   # East
        (9.6500, 79.8500),   # West (Kayts)
    ],
    "Kalutara": [
        (6.5854, 80.1616),   # Center
        (6.7500, 80.2000),   # North (Horana)
        (6.4500, 80.1500),   # South (Aluthgama)
        (6.6000, 80.3500),   # East (Bulathsinhala)
        (6.5800, 80.0500),   # West (coastal)
    ],
    "Kandy": [
        (7.2906, 80.6337),   # Center
        (7.4500, 80.6000),   # North (Matale border)
        (7.1500, 80.6500),   # South (Peradeniya)
        (7.3000, 80.8500),   # East (Teldeniya)
        (7.2800, 80.4500),   # West
    ],
    "Kegalle": [
        (7.2513, 80.3464),   # Center
        (7.4000, 80.3000),   # North (Mawanella)
        (7.1000, 80.3500),   # South (Deraniyagala)
        (7.2500, 80.5000),   # East
        (7.2500, 80.2000),   # West (Ruwanwella)
    ],
    "Kilinochchi": [
        (9.3803, 80.3770),   # Center
        (9.5000, 80.3500),   # North
        (9.2500, 80.4000),   # South
        (9.3800, 80.5500),   # East
        (9.3800, 80.2000),   # West
    ],
    "Kurunegala": [
        (7.4863, 80.3623),   # Center
        (7.7000, 80.3000),   # North (Nikaweratiya)
        (7.3000, 80.4000),   # South (Alawwa)
        (7.5000, 80.6000),   # East (Dambulla border)
        (7.5000, 80.1000),   # West (Chilaw border)
    ],
    "Mannar": [
        (8.9810, 79.9044),   # Center
        (9.1500, 79.9000),   # North
        (8.8000, 79.9500),   # South
        (8.9800, 80.1500),   # East (mainland interior)
        (8.9800, 79.7500),   # West (island side)
    ],
    "Matale": [
        (7.4675, 80.6234),   # Center
        (7.7000, 80.6000),   # North (Dambulla)
        (7.3000, 80.6500),   # South (Kandy border)
        (7.4700, 80.8500),   # East (Laggala)
        (7.4700, 80.4000),   # West
    ],
    "Matara": [
        (5.9549, 80.5550),   # Center
        (6.1000, 80.5000),   # North (Akuressa)
        (5.9000, 80.5500),   # South (coastal)
        (5.9500, 80.7500),   # East (Devinuwara)
        (5.9500, 80.3500),   # West (Weligama)
    ],
    "Monaragala": [
        (6.8728, 81.3507),   # Center
        (7.1000, 81.3000),   # North (Bibile)
        (6.6500, 81.4000),   # South (Buttala)
        (6.8700, 81.5500),   # East
        (6.8700, 81.1000),   # West
    ],
    "Mullaitivu": [
        (9.2671, 80.5881),   # Center
        (9.4500, 80.5000),   # North
        (9.0500, 80.6500),   # South
        (9.2700, 80.8000),   # East (coastal)
        (9.2700, 80.4000),   # West
    ],
    "Nuwara Eliya": [
        (6.9497, 80.7891),   # Center
        (7.1500, 80.7500),   # North
        (6.7500, 80.8000),   # South
        (6.9500, 80.9500),   # East
        (6.9500, 80.6000),   # West
    ],
    "Polonnaruwa": [
        (7.9403, 81.0188),   # Center
        (8.1500, 81.0000),   # North (Medirigiriya)
        (7.7500, 81.0500),   # South (Dimbulagala)
        (7.9400, 81.2500),   # East
        (7.9400, 80.8000),   # West
    ],
    "Puttalam": [
        (8.0362, 79.8283),   # Center
        (8.3000, 79.8500),   # North (Kalpitiya)
        (7.7500, 79.9000),   # South (Chilaw)
        (8.0400, 80.1000),   # East (interior)
        (8.0400, 79.7000),   # West (coastal)
    ],
    "Ratnapura": [
        (6.6828, 80.3992),   # Center
        (6.9000, 80.3500),   # North (Eheliyagoda)
        (6.4500, 80.4000),   # South (Embilipitiya)
        (6.7000, 80.6500),   # East (Balangoda)
        (6.7000, 80.2000),   # West
    ],
    "Trincomalee": [
        (8.5874, 81.2152),   # Center
        (8.8000, 81.1500),   # North (Padavi Sripura)
        (8.3500, 81.2500),   # South (Kantale)
        (8.5900, 81.4000),   # East (coastal)
        (8.5900, 80.9500),   # West (interior)
    ],
    "Vavuniya": [
        (8.7514, 80.4971),   # Center
        (8.9500, 80.5000),   # North
        (8.5500, 80.5000),   # South
        (8.7500, 80.7000),   # East
        (8.7500, 80.3000),   # West
    ],
}


def _build_ssl_context() -> ssl.SSLContext:
    """Lenient SSL context for environments with outdated certs."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _fetch_json(url: str, timeout: int = 25) -> Any:
    """HTTP GET → parsed JSON, with SSL fallback."""
    req = urllib.request.Request(url, headers={"User-Agent": "FloodTimeline-SL/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError:
        ctx = _build_ssl_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.loads(resp.read().decode())


def _get_sample_points(district: str, district_ref: dict) -> List[Tuple[float, float]]:
    """Return sample points for a district, falling back to center only."""
    if district in DISTRICT_SAMPLE_POINTS:
        return DISTRICT_SAMPLE_POINTS[district]
    # Fallback: just the center point
    info = district_ref.get(district, {})
    lat = info.get("center_lat", info.get("latitude", 7.87))
    lon = info.get("center_lon", info.get("longitude", 80.77))
    return [(lat, lon)]


def fetch_max_rainfall_7d(
    district: str,
    target_date: date,
    district_ref: dict,
) -> float:
    """
    Fetch 7-day cumulative rainfall for a district using multi-point
    sampling and return the MAXIMUM across all sample points.

    Automatically selects:
      - Archive API for past dates
      - Forecast API for today / future dates
    """
    points = _get_sample_points(district, district_ref)
    today = date.today()

    max_rain = 0.0
    for lat, lon in points:
        try:
            rain = _fetch_7d_rainfall_at_point(lat, lon, target_date, today)
            if rain > max_rain:
                max_rain = rain
        except Exception as e:
            logger.warning(
                f"[Weather] Failed point ({lat},{lon}) for {district}: {e}"
            )
            continue

    if max_rain == 0.0:
        # All points failed — use district baseline as last resort
        fallback = float(district_ref.get(district, {}).get("rainfall_7d_mm", 50.0))
        logger.warning(
            f"[Weather] All sample points failed for {district}. "
            f"Using baseline fallback: {fallback:.1f} mm"
        )
        return fallback

    logger.info(
        f"[Weather] {district} {target_date}: "
        f"max 7d rain = {max_rain:.1f} mm across {len(points)} points"
    )
    return max_rain


def fetch_max_rainfall_7d_batched(
    districts: List[str],
    target_date: date,
    district_ref: dict,
) -> Dict[str, float]:
    """
    Batched multi-point sampling for multiple districts.
    Returns {district_name: max_7d_rainfall_mm}.

    Packs ALL sample points from ALL districts into a SINGLE API call
    (e.g. 25 districts × 5 points = 125 coordinates in one request)
    to avoid rate-limiting.
    """
    today = date.today()

    # Build flat list of all coordinates + track which district each belongs to
    all_lats: List[str] = []
    all_lons: List[str] = []
    point_to_district: List[str] = []  # index → district name

    for district in districts:
        points = _get_sample_points(district, district_ref)
        for lat, lon in points:
            all_lats.append(str(lat))
            all_lons.append(str(lon))
            point_to_district.append(district)

    lat_str = ",".join(all_lats)
    lon_str = ",".join(all_lons)

    logger.info(
        f"[Weather] Fetching {len(all_lats)} sample points "
        f"for {len(districts)} districts in a single API call..."
    )

    try:
        if target_date < today:
            fetch_start = (target_date - timedelta(days=10)).isoformat()
            fetch_end = (target_date + timedelta(days=3)).isoformat()
            url = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={lat_str}&longitude={lon_str}"
                f"&daily=precipitation_sum"
                f"&start_date={fetch_start}&end_date={fetch_end}"
                f"&timezone=Asia%2FColombo"
            )
        else:
            past_days = 7
            future_offset = (target_date - today).days + 3
            forecast_days = max(future_offset, 7)
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat_str}&longitude={lon_str}"
                f"&daily=precipitation_sum"
                f"&past_days={past_days}&forecast_days={forecast_days}"
                f"&timezone=Asia%2FColombo"
            )

        data = _fetch_json(url, timeout=60)

        # Normalize: single coordinate returns dict, multiple returns list
        if isinstance(data, dict) and "latitude" in data:
            data = [data]

        # Compute 7-day sum for each point and group by district (take max)
        district_max: Dict[str, float] = {d: 0.0 for d in districts}

        for i, point_data in enumerate(data):
            district = point_to_district[i]
            daily = point_data.get("daily", {})
            times = daily.get("time", [])
            precip = daily.get("precipitation_sum", [])
            precip_map = {
                t: (float(p) if p is not None else 0.0)
                for t, p in zip(times, precip)
            }

            sum_7d = 0.0
            for offset in range(1, 8):
                prev = (target_date - timedelta(days=offset)).isoformat()
                sum_7d += precip_map.get(prev, 0.0)

            if sum_7d > district_max[district]:
                district_max[district] = sum_7d

        for d in districts:
            logger.info(f"[Weather] {d}: max 7d rain = {district_max[d]:.1f} mm")

        return district_max

    except Exception as e:
        logger.warning(f"[Weather] Single-call batch failed: {e}. Falling back to baselines.")
        results: Dict[str, float] = {}
        for district in districts:
            fallback = float(district_ref.get(district, {}).get("rainfall_7d_mm", 50.0))
            results[district] = fallback
        return results


def _fetch_7d_rainfall_at_point(
    lat: float,
    lon: float,
    target_date: date,
    today: date,
) -> float:
    """
    Fetch 7-day cumulative rainfall for a single lat/lon point.
    Chooses Archive vs Forecast API automatically.
    """
    if target_date < today:
        # Historical
        fetch_start = (target_date - timedelta(days=10)).isoformat()
        fetch_end = (target_date + timedelta(days=3)).isoformat()
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=precipitation_sum"
            f"&start_date={fetch_start}&end_date={fetch_end}"
            f"&timezone=Asia%2FColombo"
        )
    else:
        # Today or future — use forecast API
        past_days = 7
        future_offset = (target_date - today).days + 3
        forecast_days = max(future_offset, 7)
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=precipitation_sum"
            f"&past_days={past_days}&forecast_days={forecast_days}"
            f"&timezone=Asia%2FColombo"
        )

    data = _fetch_json(url)
    daily = data.get("daily", {})
    times = daily.get("time", [])
    precip = daily.get("precipitation_sum", [])

    precip_map = {
        t: (float(p) if p is not None else 0.0)
        for t, p in zip(times, precip)
    }

    # Rolling 7-day sum: [T-7 .. T-1]
    sum_7d = 0.0
    for offset in range(1, 8):
        prev = (target_date - timedelta(days=offset)).isoformat()
        sum_7d += precip_map.get(prev, 0.0)

    return sum_7d
