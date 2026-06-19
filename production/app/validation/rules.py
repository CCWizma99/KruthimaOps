"""
Validation Layer — Rules Engine
Physical invariants + anomaly detection for incoming prediction requests.

Errors:   reject the request outright (impossible physics)
Warnings: flag inconsistency but allow inference to proceed

Returns (clean_data: dict, warnings: list[str])
"""
from __future__ import annotations
from typing import Tuple, List, Dict, Any

from app.schemas import PredictRequest


# ── Physical Bounds ────────────────────────────────────────────────────
HARD_RULES: List[Tuple[str, str, float, float]] = [
    # (field, label, min, max)
    ("rainfall_7d_mm",      "Rainfall",         0.0,    2000.0),
    ("inundation_area_sqm", "Inundation area",  0.0,    1_000_000.0),
    ("elevation_m",         "Elevation",        -20.0,  8850.0),
    ("distance_to_river_m", "River distance",   0.0,    500_000.0),
    ("built_up_percent",    "Built-up %",       0.0,    100.0),
    ("monthly_rainfall_mm", "Monthly rainfall", 0.0,    5000.0),
]

VALID_FLOOD_OCCURRENCE = {"yes", "no"}
VALID_IS_GOOD_TO_LIVE  = {"yes", "no"}


def validate(payload: PredictRequest) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate a PredictRequest.
    Returns cleaned data dict and a list of warning strings.
    Raises ValueError on hard constraint violations.
    """
    data     = payload.model_dump()
    warnings = []

    # ── Hard bounds on numeric optional fields ──────────────────────
    for field, label, lo, hi in HARD_RULES:
        val = data.get(field)
        if val is not None:
            if val < lo or val > hi:
                raise ValueError(
                    f"{label} value {val} is outside physically valid range [{lo}, {hi}]."
                )

    # ── Categorical normalisation ───────────────────────────────────
    flood_occ_norm = data["flood_occurrence_current_event"].strip().lower()
    is_good_norm   = data["is_good_to_live"].strip().lower()

    if flood_occ_norm not in VALID_FLOOD_OCCURRENCE:
        raise ValueError(
            f"flood_occurrence_current_event must be 'Yes' or 'No', got '{data['flood_occurrence_current_event']}'."
        )
    if is_good_norm not in VALID_IS_GOOD_TO_LIVE:
        raise ValueError(
            f"is_good_to_live must be 'Yes' or 'No', got '{data['is_good_to_live']}'."
        )

    # Normalise to Title Case for consistency with training data
    data["flood_occurrence_current_event"] = "Yes" if flood_occ_norm == "yes" else "No"
    data["is_good_to_live"]                = "Yes" if is_good_norm  == "yes" else "No"

    # ── Cross-field anomaly warnings & Hard Exceptions ──────────────
    if data["rainfall_7d_mm"] > 150 and data["inundation_area_sqm"] == 0:
        raise ValueError(
            "Physics Violation: Extreme rainfall (>150mm) mathematically contradicts 0 sqm inundation."
        )

    if data["inundation_area_sqm"] > 50000 and data["flood_occurrence_current_event"] == "No":
        raise ValueError(
            "Physics Violation: Massive inundation (>50000 sqm) contradicts 'No' active flooding."
        )

    if 10000 < data["inundation_area_sqm"] <= 50000 and data["flood_occurrence_current_event"] == "No":
        warnings.append(
            "⚠ High inundation area reported despite no active flood."
        )

    if data["flood_occurrence_current_event"] == "Yes" and data["is_good_to_live"] == "Yes":
        warnings.append(
            "⚠ Inconsistent survey: active flooding reported but location rated as safe to live."
        )

    built_up = data.get("built_up_percent")
    if built_up is not None and built_up > 90 and data.get("district", "").lower() in {
        "vavuniya", "mullaitivu", "mannar", "kilinochchi", "polonnaruwa"
    }:
        warnings.append(
            "⚠ Very high built-up percentage in a predominantly rural district — verify data."
        )

    if data["rainfall_7d_mm"] > 400:
        warnings.append(
            "⚠ Extreme rainfall value (>400mm/7d) — ensure this reflects actual observed data, not a forecast."
        )

    if data["inundation_area_sqm"] == 0 and data["flood_occurrence_current_event"] == "Yes":
        warnings.append(
            "⚠ Flooding reported as active but inundation area is 0 sqm — consider setting inundation > 0."
        )

    return data, warnings
