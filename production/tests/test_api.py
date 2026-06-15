import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_endpoint_valid():
    payload = {
        "district": "Colombo",
        "rainfall_7d_mm": 50.0,
        "inundation_area_sqm": 100.0,
        "flood_occurrence_current_event": "Yes",
        "is_good_to_live": "No",
        "reason_not_good_to_live": "Flood Risk"
    }
    # Wait until artifacts are loaded implicitly in tests?
    # TestClient doesn't run startup events automatically unless explicitly handled
    with TestClient(app) as client_started:
        response = client_started.post("/api/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "risk_score" in data
        assert "risk_level" in data
        assert "prediction_id" in data

def test_predict_endpoint_invalid_payload():
    payload = {
        "district": "Colombo",
        "rainfall_7d_mm": -10.0, # Invalid
        "inundation_area_sqm": 100.0,
        "flood_occurrence_current_event": "Yes",
        "is_good_to_live": "No",
        "reason_not_good_to_live": "Flood Risk"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422 # Pydantic validation error
