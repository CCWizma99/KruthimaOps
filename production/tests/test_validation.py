from app.validation.rules import validate
from app.schemas import PredictRequest
from pydantic import ValidationError
import pytest


def test_validate_valid_payload():
    payload = PredictRequest(
        district="Colombo",
        rainfall_7d_mm=50.0,
        inundation_area_sqm=100.0,
        flood_occurrence_current_event="Yes",
        is_good_to_live="No",
        reason_not_good_to_live="Flood Risk",
    )

    clean_data, warnings = validate(payload)

    assert clean_data["rainfall_7d_mm"] == 50.0
    assert len(warnings) == 0


def test_validate_hard_bounds_rainfall():
    with pytest.raises(ValidationError):
        PredictRequest(
            district="Colombo",
            rainfall_7d_mm=3000.0,  # Over max 2000
            inundation_area_sqm=100.0,
            flood_occurrence_current_event="No",
            is_good_to_live="Yes",
            reason_not_good_to_live="None",
        )


def test_validate_anomaly_warning():
    payload = PredictRequest(
        district="Colombo",
        rainfall_7d_mm=50.0,
        inundation_area_sqm=20000.0,
        flood_occurrence_current_event="No",  # Anomaly: high inundation but no flood
        is_good_to_live="Yes",
        reason_not_good_to_live="None",
    )

    clean_data, warnings = validate(payload)

    assert len(warnings) > 0
    assert any(
        "High inundation area reported despite no active flood" in warning
        for warning in warnings
    )