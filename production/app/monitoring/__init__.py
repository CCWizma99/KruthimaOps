"""
Monitoring Layer — Public Interface
Exposes: log_prediction(), log_error(), get_metrics()
The API orchestrator calls ONLY these functions — it never imports
the adapter internals directly.
"""
from app.config import MONITOR_BACKEND

if MONITOR_BACKEND == "sqlite":
    from app.monitoring.sqlite_adapter import (
        log_prediction,
        log_error,
        log_feedback,
        get_metrics,
        get_recent_predictions,
        init_db,
        get_cached_forecast,
        save_cached_forecast,
        get_all_today_forecasts,
    )
else:
    raise ValueError(f"Unknown MONITOR_BACKEND: {MONITOR_BACKEND}")

__all__ = [
    "log_prediction",
    "log_error",
    "log_feedback",
    "get_metrics",
    "get_recent_predictions",
    "init_db",
    "get_cached_forecast",
    "save_cached_forecast",
    "get_all_today_forecasts",
]
