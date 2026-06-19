"""
Monitoring Layer — PostgreSQL Adapter
Implements the same signatures as sqlite_adapter.py but translates them to PostgreSQL / Supabase,
utilizing psycopg2 and connection pooling.
"""
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.database import get_db_cursor

logger = logging.getLogger(__name__)

# ── Schema (PostgreSQL DDL) ──────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   VARCHAR PRIMARY KEY,
    timestamp       VARCHAR NOT NULL,
    district        VARCHAR,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    rainfall_7d     DOUBLE PRECISION,
    flood_occurrence VARCHAR,
    inundation_area DOUBLE PRECISION,
    is_good_to_live VARCHAR,
    risk_score      DOUBLE PRECISION,
    risk_level      VARCHAR,
    latency_ms      INTEGER,
    has_warnings    INTEGER DEFAULT 0,
    warning_text    TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id     VARCHAR PRIMARY KEY,
    prediction_id   VARCHAR NOT NULL,
    timestamp       VARCHAR NOT NULL,
    feedback_type   VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS errors (
    error_id        VARCHAR PRIMARY KEY,
    timestamp       VARCHAR NOT NULL,
    endpoint        VARCHAR,
    error_type      VARCHAR,
    trace           TEXT
);

CREATE TABLE IF NOT EXISTS forecast_cache (
    district          VARCHAR NOT NULL,
    forecast_date     VARCHAR NOT NULL,
    calculation_date  VARCHAR NOT NULL,
    rainfall_7d_mm    DOUBLE PRECISION NOT NULL,
    risk_score        DOUBLE PRECISION NOT NULL,
    risk_level        VARCHAR NOT NULL,
    PRIMARY KEY (district, forecast_date)
);
"""

def init_db() -> None:
    """Create PostgreSQL tables on Supabase if they don't exist."""
    logger.info("[Database] Ensuring PostgreSQL schemas are initialized...")
    with get_db_cursor() as cur:
        cur.execute(_DDL)
    logger.info("[Database] Schemas initialized successfully.")

# ── Write operations ──────────────────────────────────────────────────

def log_prediction(
    prediction_id: str,
    district: str,
    latitude: float,
    longitude: float,
    rainfall_7d: float,
    flood_occurrence: str,
    inundation_area: float,
    is_good_to_live: str,
    risk_score: float,
    risk_level: str,
    latency_ms: int,
    warnings: List[str],
) -> None:
    with get_db_cursor() as cur:
        cur.execute(
            """INSERT INTO predictions VALUES
               (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (prediction_id) DO NOTHING""",
            (
                prediction_id,
                datetime.now(timezone.utc).isoformat(),
                district,
                latitude,
                longitude,
                rainfall_7d,
                flood_occurrence,
                inundation_area,
                is_good_to_live,
                risk_score,
                risk_level,
                latency_ms,
                1 if warnings else 0,
                " | ".join(warnings) if warnings else None,
            ),
        )

def log_error(endpoint: str, error_type: str, trace: str) -> None:
    with get_db_cursor() as cur:
        cur.execute(
            "INSERT INTO errors VALUES (%s,%s,%s,%s,%s)",
            (str(uuid.uuid4()), datetime.now(timezone.utc).isoformat(),
             endpoint, error_type, trace),
        )

def log_feedback(prediction_id: str, feedback_type: str) -> str:
    feedback_id = str(uuid.uuid4())
    with get_db_cursor() as cur:
        cur.execute(
            "INSERT INTO feedback VALUES (%s,%s,%s,%s)",
            (feedback_id, prediction_id,
             datetime.now(timezone.utc).isoformat(), feedback_type),
        )
    return feedback_id

# ── Read operations ───────────────────────────────────────────────────

def get_metrics() -> Dict[str, Any]:
    with get_db_cursor() as cur:
        # Total predictions count
        cur.execute("SELECT COUNT(*) as cnt FROM predictions")
        total = cur.fetchone()["cnt"] or 0

        # Avg latency
        cur.execute("SELECT AVG(latency_ms) as avg_lat FROM predictions")
        avg_lat = cur.fetchone()["avg_lat"] or 0

        # Total errors
        cur.execute("SELECT COUNT(*) as cnt FROM errors")
        err_count = cur.fetchone()["cnt"] or 0

        # Risk distribution
        cur.execute("SELECT risk_level, COUNT(*) as cnt FROM predictions GROUP BY risk_level")
        rows = cur.fetchall()
        risk_dist_raw = {r["risk_level"]: r["cnt"] for r in rows}
        risk_dist = {lvl: round(cnt / max(total, 1), 4) for lvl, cnt in risk_dist_raw.items()}

        # Feedback stats
        cur.execute("SELECT feedback_type, COUNT(*) as cnt FROM feedback GROUP BY feedback_type")
        fb_rows = cur.fetchall()
        fb = {r["feedback_type"]: r["cnt"] for r in fb_rows}
        fb_total = sum(fb.values())
        fb_acc = round(fb.get("accurate", 0) / max(fb_total, 1), 4) if fb_total else None

        # Predictions per hour (last 24h)
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        cur.execute(
            """SELECT to_char(date_trunc('hour', timestamp::timestamptz), 'YYYY-MM-DD"T"HH24:00:00') as hr,
                      COUNT(*) as cnt
               FROM predictions
               WHERE timestamp >= %s
               GROUP BY hr
               ORDER BY hr""",
            (cutoff,),
        )
        hourly = cur.fetchall()
        predictions_last_24h = [{"hour": r["hr"], "count": r["cnt"]} for r in hourly]

    return {
        "total_predictions":    total,
        "avg_latency_ms":       round(float(avg_lat), 2),
        "error_rate":           round(err_count / max(total + err_count, 1), 4),
        "feedback_accuracy":    fb_acc,
        "risk_distribution":    risk_dist,
        "predictions_last_24h": predictions_last_24h,
    }

def get_recent_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    with get_db_cursor() as cur:
        cur.execute(
            """SELECT prediction_id, timestamp, district, rainfall_7d,
                      risk_score, risk_level, latency_ms, has_warnings
               FROM predictions
               ORDER BY timestamp DESC
               LIMIT %s""",
            (limit,),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]

def get_prediction_by_id(prediction_id: str) -> Optional[Dict[str, Any]]:
    with get_db_cursor() as cur:
        cur.execute(
            """SELECT prediction_id, timestamp, district, latitude, longitude,
                      rainfall_7d, flood_occurrence, inundation_area,
                      is_good_to_live, risk_score, risk_level, latency_ms,
                      has_warnings, warning_text
               FROM predictions
               WHERE prediction_id = %s""",
            (prediction_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None

def get_cached_forecast(district: str, forecast_date: str) -> Optional[Dict[str, Any]]:
    with get_db_cursor() as cur:
        cur.execute(
            """SELECT district, forecast_date, calculation_date, rainfall_7d_mm, risk_score, risk_level
               FROM forecast_cache
               WHERE district = %s AND forecast_date = %s""",
            (district, forecast_date),
        )
        row = cur.fetchone()
    return dict(row) if row else None

def save_cached_forecast(
    district: str,
    forecast_date: str,
    calculation_date: str,
    rainfall_7d_mm: float,
    risk_score: float,
    risk_level: str,
) -> None:
    with get_db_cursor() as cur:
        cur.execute(
            """INSERT INTO forecast_cache 
               (district, forecast_date, calculation_date, rainfall_7d_mm, risk_score, risk_level)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (district, forecast_date) DO UPDATE SET
                   calculation_date = EXCLUDED.calculation_date,
                   rainfall_7d_mm = EXCLUDED.rainfall_7d_mm,
                   risk_score = EXCLUDED.risk_score,
                   risk_level = EXCLUDED.risk_level""",
            (district, forecast_date, calculation_date, rainfall_7d_mm, risk_score, risk_level),
        )

def get_all_today_forecasts(today: str) -> Dict[str, List[Dict[str, Any]]]:
    with get_db_cursor() as cur:
        cur.execute(
            """SELECT district, forecast_date, rainfall_7d_mm, risk_score, risk_level
               FROM forecast_cache
               WHERE calculation_date = %s
               ORDER BY district, forecast_date ASC""",
            (today,),
        )
        rows = cur.fetchall()
    
    result = {}
    for r in rows:
        d = r["district"]
        if d not in result:
            result[d] = []
          # Ensure dict fields match expected format
        result[d].append({
            "forecast_date":  r["forecast_date"],
            "rainfall_7d_mm": float(r["rainfall_7d_mm"]),
            "risk_score":     float(r["risk_score"]),
            "risk_level":     r["risk_level"]
        })
    return result
