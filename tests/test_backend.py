# tests/test_backend.py
"""
Integration and API end-to-end routing verification tests.
"""

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_api_root():
    """Verifies that the root path returns successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Climate Platform Services Active"

def test_weather_endpoint():
    """Verifies that the weather API returns real-time telemetry details."""
    response = client.get("/api/v1/weather/current?latitude=22.9734&longitude=78.6569")
    assert response.status_code == 200
    data = response.json()
    assert "temperature" in data
    assert "relative_humidity" in data

def test_prediction_endpoint():
    """Verifies that the prediction endpoint returns regression forecasting values."""
    payload = {
        "temperature_2m_mean": 28.0,
        "relative_humidity_2m_mean": 80.0,
        "wind_speed_10m_max": 15.0,
        "pressure_msl_mean": 1008.0,
        "month": 6
    }
    response = client.post("/api/v1/predict/regression", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_precipitation" in data
    assert data["predicted_precipitation"] >= 0.0