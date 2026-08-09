# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import predict, weather

app = FastAPI(title="Weather Intelligence and Climate Decision Support Platform API")

# Allow Streamlit to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api/v1/predict", tags=["Prediction"])
app.include_router(weather.router, prefix="/api/v1/weather", tags=["Weather"])

@app.get("/")
def read_root():
    return {"message": "Climate Platform Services Active", "status": "Healthy"}