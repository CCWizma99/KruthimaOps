"""
Monitoring Layer — SQLite Adapter
Implements: log_prediction, log_error, log_feedback, get_metrics,
            get_recent_predictions, init_db
To swap for PostgreSQL: write a postgres_adapter.py with the same
function signatures, then change MONITOR_BACKEND in config.py.
"""
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.config import SQLITE_DB_PATH

# ── Schema ────────────────────────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    district        TEXT,
    latitude        REAL,
    longitude       REAL,
    rainfall_7d     REAL,
    flood_occurrence TEXT,
    inundation_area REAL,
    is_good_to_live TEXT,
    risk_score      REAL,
    risk_level      TEXT,
    latency_ms      INTEGER,
    has_warnings    INTEGER DEFAULT 0,
    warning_text    TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id     TEXT PRIMARY KEY,
    prediction_id   TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    feedback_type   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS errors (
    error_id        TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    endpoint        TEXT,
    error_type      TEXT,
    trace           TEXT
);

CREATE TABLE IF NOT EXISTS forecast_cache (
    district          TEXT NOT NULL,
    forecast_date     TEXT NOT NULL,
    calculation_date  TEXT NOT NULL,
    rainfall_7d_mm    REAL NOT NULL,
    risk_score        REAL NOT NULL,
    risk_level        TEXT NOT NULL,
    PRIMARY KEY (district, forecast_date)
);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(SQLITE_DB_PATH, timeout=15.0)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    """Create tables if they don't exist. Called once at app startup."""
    with _conn() as c:
        c.executescript(_DDL)


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
    with _conn() as c:
        c.execute(
            """INSERT OR IGNORE INTO predictions VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
    with _conn() as c:
        c.execute(
            "INSERT INTO errors VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), datetime.now(timezone.utc).isoformat(),
             endpoint, error_type, trace),
        )


def log_feedback(prediction_id: str, feedback_type: str) -> str:
    feedback_id = str(uuid.uuid4())
    with _conn() as c:
        c.execute(
            "INSERT INTO feedback VALUES (?,?,?,?)",
            (feedback_id, prediction_id,
             datetime.now(timezone.utc).isoformat(), feedback_type),
        )
    return feedback_id


# ── Read operations ───────────────────────────────────────────────────

def get_metrics() -> Dict[str, Any]:
    with _conn() as c:
        total     = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        avg_lat   = c.execute("SELECT AVG(latency_ms) FROM predictions").fetchone()[0] or 0
        err_count = c.execute("SELECT COUNT(*) FROM errors").fetchone()[0]

        rows = c.execute(
            "SELECT risk_level, COUNT(*) as cnt FROM predictions GROUP BY risk_level"
        ).fetchall()
        risk_dist_raw = {r["risk_level"]: r["cnt"] for r in rows}
        risk_dist     = {lvl: round(cnt / max(total, 1), 4)
                         for lvl, cnt in risk_dist_raw.items()}

        fb_rows = c.execute(
            "SELECT feedback_type, COUNT(*) FROM feedback GROUP BY feedback_type"
        ).fetchall()
        fb      = {r[0]: r[1] for r in fb_rows}
        fb_total = sum(fb.values())
        fb_acc   = round(fb.get("accurate", 0) / max(fb_total, 1), 4) if fb_total else None

        # Predictions per hour (last 24h)
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        hourly = c.execute(
            """SELECT strftime('%Y-%m-%dT%H:00:00', timestamp) as hr,
                      COUNT(*) as cnt
               FROM predictions
               WHERE timestamp >= ?
               GROUP BY hr
               ORDER BY hr""",
            (cutoff,),
        ).fetchall()
        predictions_last_24h = [{"hour": r["hr"], "count": r["cnt"]} for r in hourly]

    return {
        "total_predictions":    total,
        "avg_latency_ms":       round(avg_lat, 2),
        "error_rate":           round(err_count / max(total + err_count, 1), 4),
        "feedback_accuracy":    fb_acc,
        "risk_distribution":    risk_dist,
        "predictions_last_24h": predictions_last_24h,
    }


def get_recent_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            """SELECT prediction_id, timestamp, district, rainfall_7d,
                      risk_score, risk_level, latency_ms, has_warnings
               FROM predictions
               ORDER BY timestamp DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_prediction_by_id(prediction_id: str) -> Optional[Dict[str, Any]]:
    """Return a single logged prediction row for PDF report generation."""
    with _conn() as c:
        row = c.execute(
            """SELECT prediction_id, timestamp, district, latitude, longitude,
                      rainfall_7d, flood_occurrence, inundation_area,
                      is_good_to_live, risk_score, risk_level, latency_ms,
                      has_warnings, warning_text
               FROM predictions
               WHERE prediction_id = ?""",
            (prediction_id,),
        ).fetchone()
    return dict(row) if row else None


def get_cached_forecast(district: str, forecast_date: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute(
            """SELECT district, forecast_date, calculation_date, rainfall_7d_mm, risk_score, risk_level
               FROM forecast_cache
               WHERE district = ? AND forecast_date = ?""",
            (district, forecast_date),
        ).fetchone()
        return dict(row) if row else None


def save_cached_forecast(
    district: str,
    forecast_date: str,
    calculation_date: str,
    rainfall_7d_mm: float,
    risk_score: float,
    risk_level: str,
) -> None:
    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO forecast_cache 
               (district, forecast_date, calculation_date, rainfall_7d_mm, risk_score, risk_level)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (district, forecast_date, calculation_date, rainfall_7d_mm, risk_score, risk_level),
        )


def get_all_today_forecasts(today: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return all 7-day forecast days for all districts calculated today.
    """
    with _conn() as c:
        rows = c.execute(
            """SELECT district, forecast_date, rainfall_7d_mm, risk_score, risk_level
               FROM forecast_cache
               WHERE calculation_date = ?
               ORDER BY district, forecast_date ASC""",
            (today,),
        ).fetchall()
    
    result = {}
    for r in rows:
        d = r["district"]
        if d not in result:
            result[d] = []
        result[d].append({
            "forecast_date":  r["forecast_date"],
            "rainfall_7d_mm": r["rainfall_7d_mm"],
            "risk_score":     r["risk_score"],
            "risk_level":     r["risk_level"]
        })
    return result
