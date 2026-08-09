# backend/api/__init__.py
from fastapi import APIRouter
from backend.api.weather import router as weather_router
from backend.api.predict import router as predict_router
from backend.api.satellite import router as satellite_router
from backend.api.assistant import router as assistant_router

# Core API router aggregation
api_router = APIRouter()

api_router.include_router(weather_router)
api_router.include_router(predict_router)
api_router.include_router(satellite_router)
api_router.include_router(assistant_router)