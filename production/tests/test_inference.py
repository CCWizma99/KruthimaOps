import pytest
from app.inference.v703_engine import infer, load_artifacts, get_model_metadata

@pytest.fixture(scope="module", autouse=True)
def setup_artifacts():
    load_artifacts()

def test_inference_returns_valid_score():
    features = {
        "district": "Colombo",
        "rainfall_7d_mm": 50.0,
        "inundation_area_sqm": 100.0,
        "flood_occurrence_current_event": "Yes",
        "is_good_to_live": "No",
        "reason_not_good_to_live": "Flood Risk"
    }
    score = infer(features)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_get_metadata():
    meta = get_model_metadata()
    assert "version" in meta
    assert meta["version"] == "prod_v1k.2"
    assert "oof_mae" in meta
