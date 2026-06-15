"""
FloodGuard SL — Shared Data Contracts (Pydantic Schemas)
All layers communicate through these schemas — no layer imports
another layer's internal types directly.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Prediction Request ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    district: str                   = Field(..., description="Sri Lankan district name")

    # Dynamic / User-Controlled Inputs
    rainfall_7d_mm: float           = Field(..., ge=0, le=2000, description="7-day accumulated rainfall (mm)")
    inundation_area_sqm: float      = Field(0.0, ge=0, description="Current inundation area (sqm)")
    flood_occurrence_current_event: str = Field("No", description="Active flooding: Yes/No")
    is_good_to_live: str            = Field("Yes", description="Survey: safe to live: Yes/No")
    reason_not_good_to_live: str    = Field("None", description="Reason location is unsafe")

    # Optional overrides (auto-filled from district_reference if absent)
    latitude:              Optional[float] = None
    longitude:             Optional[float] = None
    elevation_m:           Optional[float] = None
    distance_to_river_m:   Optional[float] = None
    monthly_rainfall_mm:   Optional[float] = None
    built_up_percent:      Optional[float] = None
    population_density_per_km2_log1p: Optional[float] = None
    infrastructure_score:  Optional[float] = None
    historical_flood_count: Optional[float] = None


class BatchPredictRequest(BaseModel):
    rows: List[PredictRequest]


# ── Prediction Response ───────────────────────────────────────────────

class PredictResponse(BaseModel):
    prediction_id:  str
    risk_score:     float           = Field(..., description="Flood risk score [0, 1]")
    risk_level:     str             = Field(..., description="LOW / MEDIUM / HIGH / EXTREME")
    district:       str
    rainfall_7d_mm: float
    latency_ms:     int
    warnings:       List[str]       = []
    briefing:       Optional[str]   = None


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total:   int


# ── Feedback ─────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    prediction_id: str
    feedback_type: str  = Field(..., description="accurate / inaccurate")


class FeedbackResponse(BaseModel):
    status:        str
    feedback_id:   str


# ── Metrics ──────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    total_predictions:   int
    avg_latency_ms:      float
    error_rate:          float
    feedback_accuracy:   Optional[float]
    risk_distribution:   Dict[str, float]
    predictions_last_24h: List[Dict[str, Any]]


# ── Model Info ───────────────────────────────────────────────────────

class ModelInfoResponse(BaseModel):
    version:            str
    base_pipeline:      str
    training_date:      str
    seed:               int
    n_folds:            int
    oof_mae:            float
    oof_rmse:           float
    oof_ev:             float
    est_lb_score:       float
    opt_lb_score:       float
    n_base_features:    int
    n_total_features:   int
    model_names:        List[str]


# ── District Reference ────────────────────────────────────────────────

class DistrictInfoResponse(BaseModel):
    district:    str
    center_lat:  float
    center_lon:  float
    defaults:    Dict[str, Any]
