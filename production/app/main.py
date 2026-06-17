"""
FloodGuard SL — FastAPI Orchestrator
Thin layer that wires the 4 independent modules:
  validate → infer → monitor → brief → respond

All business logic lives in the respective modules.
"""
from __future__ import annotations

import logging
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app import config
from app.briefing import brief
from app.inference import get_district_reference, get_model_metadata, infer, load_artifacts
from app.monitoring import get_metrics, get_prediction_by_id, get_recent_predictions, init_db, log_error, log_feedback, log_prediction
from app.reports import build_prediction_report_pdf
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    DistrictInfoResponse,
    FeedbackRequest,
    FeedbackResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from app.validation import validate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Precompute State ──────────────────────────────────────────────────
_precompute_status = {"total": 0, "ready": 0, "complete": False}


def _bg_precompute_all_forecasts() -> None:
    """
    Background thread: ensure all district forecasts are cached for today.
    Runs 0.5 seconds after startup so uvicorn can print its binding logs.
    Progress is tracked in _precompute_status for the frontend to poll.
    """
    global _precompute_status
    import time
    time.sleep(0.5)
    from app.forecast import get_district_forecast
    ref = get_district_reference()
    districts = sorted(ref.keys())
    _precompute_status["total"] = len(districts)
    _precompute_status["ready"] = 0
    _precompute_status["complete"] = False
    logger.info(f"[Precompute] Starting for {len(districts)} districts...")
    for district in districts:
        try:
            get_district_forecast(district)
            _precompute_status["ready"] += 1
            logger.info(
                f"[Precompute] {district} done "
                f"({_precompute_status['ready']}/{_precompute_status['total']})"
            )
        except Exception as e:
            logger.warning(f"[Precompute] {district} failed: {e}")
    _precompute_status["complete"] = True
    logger.info("[Precompute] All district forecasts ready.")


# ── Risk Level Classifier ────────────────────────────────────────────

def _risk_level(score: float) -> str:
    if score < 0.25:   return "LOW"
    if score < 0.50:   return "MEDIUM"
    if score < 0.75:   return "HIGH"
    return "EXTREME"


# ── Startup / Shutdown ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("[Startup] Initialising monitoring DB...")
    init_db()
    logger.info("[Startup] Loading ML artifacts...")
    load_artifacts()
    logger.info("[Startup] Launching background precompute thread...")
    threading.Thread(target=_bg_precompute_all_forecasts, daemon=True).start()
    logger.info("[Startup] FloodGuard SL is ready.")
    yield
    # Shutdown (nothing to do)


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
)

# Static files (frontend dashboard)
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ── Global error handler ──────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    trace = traceback.format_exc()
    log_error(str(request.url.path), type(exc).__name__, trace[:2000])
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. The incident has been logged."},
    )


# ── Health ────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": config.API_VERSION}


# ── Frontend ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    index = os.path.join(static_dir, "index.html")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>FloodGuard SL API is running.</h1><p>Frontend not built yet.</p>")

@app.get("/diagnostics", response_class=HTMLResponse, include_in_schema=False)
async def diagnostics_page():
    page = os.path.join(static_dir, "diagnostics.html")
    if os.path.exists(page):
        with open(page, encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Diagnostics UI not found.</h1>")

@app.get("/api/config/cesium-token", tags=["System"])
async def get_cesium_token():
    token = os.getenv("CESIUM_ION_TOKEN", "")
    return {"token": token}

from app.diagnostics import run_diagnostics

@app.get("/api/diagnostics", tags=["System"])
async def api_diagnostics():
    return run_diagnostics()


# ── Single Prediction ────────────────────────────────────────────────

@app.post("/api/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(payload: PredictRequest):
    prediction_id = str(uuid.uuid4())
    t0 = perf_counter()

    # 1. Validate
    try:
        clean_data, warnings = validate(payload)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # 2. Infer
    try:
        score = infer(clean_data)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    latency_ms = int((perf_counter() - t0) * 1000)
    risk_lvl   = _risk_level(score)

    # 3. Monitor
    lat = clean_data.get("latitude") or 7.8731
    lon = clean_data.get("longitude") or 80.7718
    log_prediction(
        prediction_id=prediction_id,
        district=clean_data["district"],
        latitude=lat,
        longitude=lon,
        rainfall_7d=clean_data["rainfall_7d_mm"],
        flood_occurrence=clean_data["flood_occurrence_current_event"],
        inundation_area=clean_data["inundation_area_sqm"],
        is_good_to_live=clean_data["is_good_to_live"],
        risk_score=score,
        risk_level=risk_lvl,
        latency_ms=latency_ms,
        warnings=warnings,
    )

    # 4. AI Briefing (non-blocking — return even if it fails)
    briefing_text = ""
    try:
        briefing_text = brief(score, clean_data)
    except Exception:
        pass

    return PredictResponse(
        prediction_id=prediction_id,
        risk_score=round(score, 6),
        risk_level=risk_lvl,
        district=clean_data["district"],
        rainfall_7d_mm=clean_data["rainfall_7d_mm"],
        latency_ms=latency_ms,
        warnings=warnings,
        briefing=briefing_text or None,
    )


# ── Batch Prediction ─────────────────────────────────────────────────

@app.post("/api/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(payload: BatchPredictRequest):
    results = []
    for row in payload.rows:
        try:
            response = await predict(row)
            results.append(response)
        except HTTPException as e:
            results.append({"error": e.detail, "district": row.district})
    return BatchPredictResponse(results=results, total=len(results))


# ── Feedback ──────────────────────────────────────────────────────────

@app.post("/api/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def feedback(payload: FeedbackRequest):
    if payload.feedback_type not in {"accurate", "inaccurate"}:
        raise HTTPException(status_code=422, detail="feedback_type must be 'accurate' or 'inaccurate'.")
    fid = log_feedback(payload.prediction_id, payload.feedback_type)
    return FeedbackResponse(status="recorded", feedback_id=fid)


# ── Metrics ───────────────────────────────────────────────────────────

@app.get("/api/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def metrics():
    data = get_metrics()
    return MetricsResponse(**data)


# ── Recent Predictions Log ────────────────────────────────────────────

@app.get("/api/log", tags=["Monitoring"])
async def activity_log(limit: int = 50):
    return {"predictions": get_recent_predictions(limit=limit)}


# ── PDF Report Generation ────────────────────────────────────────────

@app.get("/api/report/{prediction_id}", tags=["Reports"])
async def prediction_report(prediction_id: str):
    """Generate a downloadable PDF report from a logged prediction."""
    prediction = get_prediction_by_id(prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction ID not found in monitoring log.")

    # Generate an AI operational comment for the PDF using the same briefing
    # layer used by the live prediction flow. Gemini is used when configured;
    # the briefing layer provides a deterministic fallback when the API key is
    # unavailable so report generation still works during demos/CI.
    ai_comment = ""
    try:
        report_features = {
            "district": prediction.get("district"),
            "rainfall_7d_mm": prediction.get("rainfall_7d") or 0.0,
            "inundation_area_sqm": prediction.get("inundation_area") or 0.0,
            "flood_occurrence_current_event": prediction.get("flood_occurrence") or "No",
            "is_good_to_live": prediction.get("is_good_to_live") or "Yes",
            "latitude": prediction.get("latitude"),
            "longitude": prediction.get("longitude"),
        }
        ai_comment = brief(float(prediction.get("risk_score") or 0.0), report_features)
    except Exception as exc:
        logger.warning("[Report] AI comment generation failed: %s", exc)

    prediction_with_ai = {**prediction, "ai_comment": ai_comment}

    try:
        pdf_bytes = build_prediction_report_pdf(
            prediction=prediction_with_ai,
            model_metadata=get_model_metadata(),
            metrics=get_metrics(),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    short_id = prediction_id.split("-")[0]
    headers = {
        "Content-Disposition": f'attachment; filename="FloodGuard_SL_Report_{short_id}.pdf"'
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


# ── Model Registry ────────────────────────────────────────────────────

@app.get("/api/models", response_model=ModelInfoResponse, tags=["Model Management"])
async def model_info():
    meta = get_model_metadata()
    if "status" in meta:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    return ModelInfoResponse(**meta)


# ── District Reference ────────────────────────────────────────────────

@app.get("/api/districts", tags=["Data"])
async def list_districts():
    ref = get_district_reference()
    return {"districts": sorted(ref.keys())}


@app.get("/api/district/{district_name}", response_model=DistrictInfoResponse, tags=["Data"])
async def district_info(district_name: str):
    ref = get_district_reference()
    if district_name not in ref:
        raise HTTPException(status_code=404, detail=f"District '{district_name}' not found.")
    data = ref[district_name]
    return DistrictInfoResponse(
        district=district_name,
        center_lat=data.get("center_lat", data.get("latitude", 7.87)),
        center_lon=data.get("center_lon", data.get("longitude", 80.77)),
        defaults=data,
    )


from app.forecast import get_district_forecast, get_historical_forecast
from app.monitoring import get_all_today_forecasts as _get_all_today_forecasts


# ── Historical Simulation ────────────────────────────────────────────

@app.get("/api/simulate/historical", tags=["Simulation"])
async def simulate_historical(date: str):
    """
    Run the v703 model on ALL 25 districts using actual observed weather data
    from Open-Meteo's archive API for a specific past date.
    Returns district risk map for the chosen historical date.
    """
    from datetime import datetime as dt, date as d_type
    import re

    # Validate date format
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        raise HTTPException(status_code=422, detail="Invalid date format. Use YYYY-MM-DD.")

    try:
        target = dt.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid date: {date}")

    if target >= d_type.today():
        raise HTTPException(
            status_code=422,
            detail="Historical simulation requires a past date. Use the forecast API for future dates."
        )

    ref = get_district_reference()
    districts = sorted(ref.keys())
    results = {}
    errors  = []

    logger.info(f"[HistSim] Starting historical simulation for {date} across {len(districts)} districts...")

    for district in districts:
        try:
            result = get_historical_forecast(district, date)
            results[district] = result
        except Exception as e:
            logger.warning(f"[HistSim] {district} failed: {e}")
            errors.append({"district": district, "error": str(e)})

    logger.info(f"[HistSim] Completed: {len(results)}/{len(districts)} districts for {date}.")

    return {
        "date":      date,
        "districts": results,
        "ready":     len(results),
        "total":     len(districts),
        "errors":    errors,
    }


@app.get("/api/forecast/status", tags=["Forecast"])
async def forecast_precompute_status():
    """Returns background precompute progress: ready/total/complete."""
    return _precompute_status


@app.get("/api/forecasts/today", tags=["Forecast"])
async def all_today_district_forecasts():
    """
    Return today's (day 0) computed risk score for every district that
    has been processed so far. The frontend polls this to progressively
    render district risk prisms as the background thread computes them.
    """
    from datetime import date
    today = date.today().isoformat()
    ref = get_district_reference()
    data = _get_all_today_forecasts(today)
    return {
        "date":      today,
        "districts": data,
        "ready":     len(data),
        "total":     len(ref),
    }


@app.get("/api/forecast/{district_name}", tags=["Forecast"])
async def district_forecast(district_name: str):
    ref = get_district_reference()
    if district_name not in ref:
        raise HTTPException(status_code=404, detail=f"District '{district_name}' not found.")
    try:
        data = get_district_forecast(district_name)
        return {"district": district_name, "forecast": data}
    except Exception as e:
        logger.error(f"Failed to generate forecast for {district_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
