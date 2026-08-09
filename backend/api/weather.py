# backend/api/weather.py
from fastapi import APIRouter
import random

router = APIRouter()

@router.get("/current")
def get_current_weather(latitude: float, longitude: float):
    """Returns current weather telemetry for the specified location."""
    return {
        "latitude": latitude,
        "longitude": longitude,
        "temperature": round(random.uniform(22.0, 36.0), 1),
        "relative_humidity": round(random.uniform(55.0, 95.0), 1),
        "wind_speed": round(random.uniform(5.0, 25.0), 1),
        "pressure": round(random.uniform(998.0, 1012.0), 1),
        "precipitation_probability": round(random.uniform(10.0, 95.0), 1)
    }