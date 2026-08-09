# backend/api/predict.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.ml_engine import MLEngine
from models.dl_engine import DLEngine
import torch

router = APIRouter()
ml_engine = MLEngine()

class PredictionInput(BaseModel):
    temperature_2m_mean: float
    relative_humidity_2m_mean: float
    wind_speed_10m_max: float
    pressure_msl_mean: float
    month: int

@router.post("/regression")
def predict_regression(data: PredictionInput):
    """Predicts precipitation using the trained Random Forest model."""
    try:
        features = data.model_dump()
        prediction = ml_engine.predict(features)
        return {"predicted_precipitation": prediction, "unit": "mm"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))