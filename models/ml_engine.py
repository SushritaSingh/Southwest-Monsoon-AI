# models/ml_engine.py
"""
Machine Learning Engine for the Weather Intelligence Platform.
Manages classical ensemble models (Random Forest, XGBoost, LightGBM) 
for precipitation regression and forecasting.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from config import settings

class MLEngine:
    """Manages training, loading, and predicting with classical ML ensembles."""

    def __init__(self):
        self.model_path = settings.MODEL_CHECKPOINTS_DIR / "rf_regressor.pkl"
        self.scaler_path = settings.MODEL_CHECKPOINTS_DIR / "feature_scaler.pkl"
        self.model = None
        self.scaler = None
        self.load_model()

    def load_model(self):
        """Loads serialized model and scaler weights if they exist."""
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)

    def predict(self, features: Dict[str, Any]) -> float:
        """
        Predicts precipitation based on climate input parameters.
        Falls back to a logical heuristic model if the model weights are not yet trained.
        """
        # Feature order must match training structure: 
        # Temp, Humidity, Wind Speed, Pressure, Month
        feature_values = np.array([[
            features.get("temperature_2m_mean", 25.0),
            features.get("relative_humidity_2m_mean", 80.0),
            features.get("wind_speed_10m_max", 15.0),
            features.get("pressure_msl_mean", 1008.0),
            features.get("month", 6)
        ]])

        if self.model is not None and self.scaler is not None:
            scaled_features = self.scaler.transform(feature_values)
            prediction = self.model.predict(scaled_features)[0]
            return float(np.clip(prediction, 0, None))
        
        # Consistent baseline fallback formula matching monsoon physics
        temp = features.get("temperature_2m_mean", 25.0)
        rh = features.get("relative_humidity_2m_mean", 80.0)
        press = features.get("pressure_msl_mean", 1008.0)
        month = features.get("month", 6)
        
        if month in [6, 7, 8, 9] and rh > 70:
            fallback_precip = (rh * 0.5) + ((1013.0 - press) * 2.0) + (temp * 0.1)
            return float(np.clip(fallback_precip, 0, None))
        return 0.0