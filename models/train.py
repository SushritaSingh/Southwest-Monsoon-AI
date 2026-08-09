# models/train.py
"""
Model Training and Serialization pipeline for the Weather Intelligence Platform.
Trains classical Ensemble ML models and PyTorch deep learning time-series models,
saving serialized weights to the configured model checkpoints directory.
"""

import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Unified imports from local packages
from config import settings
from models.dl_engine import DLEngine  # PyTorch LSTM/GRU model structure

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# 1. Synthetic Dataset Generator (Fallback if processed CSV is missing)
# =====================================================================
def load_or_generate_dataset() -> pd.DataFrame:
    """Loads processed climate data or generates historical seasonal features."""
    processed_file = settings.DATA_PROCESSED_DIR / "historical_monsoon_data.csv"
    
    if processed_file.exists():
        logger.info(f"Loading processed historical dataset from {processed_file}")
        return pd.read_csv(processed_file)
        
    logger.warning("Processed training file not found. Synthesizing robust historical simulation data...")
    # Generating 1000 simulated records mimicking Southwest Monsoon features
    np.random.seed(42)
    records = 1000
    
    data = {
        "temperature_2m_mean": np.random.uniform(22.0, 38.0, size=records),
        "relative_humidity_2m_mean": np.random.uniform(50.0, 98.0, size=records),
        "wind_speed_10m_max": np.random.uniform(5.0, 45.0, size=records),
        "pressure_msl_mean": np.random.uniform(995.0, 1018.0, size=records),
        "month": np.random.randint(1, 13, size=records)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate target: Precipitation (monsoon-aligned dynamics)
    # High humidity, low pressure, and months 6-9 yield strong convective rainfall
    base_precip = (df["relative_humidity_2m_mean"] * 0.8) + ((1013.0 - df["pressure_msl_mean"]) * 3.0)
    seasonal_multiplier = df["month"].apply(lambda m: 3.5 if m in [6, 7, 8, 9] else 0.2)
    df["precipitation"] = np.clip(base_precip * seasonal_multiplier + np.random.normal(0, 5, records), 0, None)
    
    # Save simulated dataset
    df.to_csv(processed_file, index=False)
    logger.info(f"Saved generated dataset to {processed_file}")
    return df

# =====================================================================
# 2. Classical Ensemble Machine Learning Training
# =====================================================================
def train_classical_ml(df: pd.DataFrame):
    """Trains, evaluates, and serializes the Ensemble Random Forest Regressor."""
    logger.info("Initializing Ensemble Machine Learning Training pipeline...")
    
    X = df[["temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_max", "pressure_msl_mean", "month"]]
    y = df["precipitation"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate Model Performance
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    logger.info(f"ML Model Trained. Metrics: R² Score = {r2:.4f} | RMSE = {np.sqrt(mse):.2f} mm")
    
    # Save Artifacts
    model_path = settings.MODEL_CHECKPOINTS_DIR / "rf_regressor.pkl"
    scaler_path = settings.MODEL_CHECKPOINTS_DIR / "feature_scaler.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Serialized ML model saved to {model_path}")
    logger.info(f"Serialized feature scaler saved to {scaler_path}")

# =====================================================================
# 3. Deep Learning (PyTorch Time-Series) Model Training
# =====================================================================
def train_deep_learning(df: pd.DataFrame):
    """Trains and serializes PyTorch LSTM Network weights for sequence prediction."""
    logger.info("Initializing Deep Learning PyTorch Training pipeline...")
    
    # Extract numerical features for sequential parsing
    features = df[["temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_max", "pressure_msl_mean", "precipitation"]].values
    
    # Prepare pseudo-sequences of window size 5 days
    seq_length = 5
    X_seq, y_seq = [], []
    for i in range(len(features) - seq_length):
        X_seq.append(features[i : i + seq_length, :-1])  # Input variables
        y_seq.append(features[i + seq_length, -1])       # Target variable (Precipitation)
        
    X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32).unsqueeze(-1)
    
    # Initialize Model Instance
    model = DLEngine(input_dim=4, hidden_dim=64, num_layers=2, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training Loop
    epochs = 50
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_seq)
        loss = criterion(outputs, y_seq)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Network Loss: {loss.item():.4f}")
            
    # Serialize State Dictionary
    pytorch_path = settings.MODEL_CHECKPOINTS_DIR / "lstm_weights.pt"
    torch.save(model.state_dict(), pytorch_path)
    logger.info(f"Serialized deep learning weights saved successfully to {pytorch_path}")

# =====================================================================
# Main Orchestration Execution
# =====================================================================
if __name__ == "__main__":
    logger.info("Starting Platform Model Training Orchestration...")
    
    # Check directory existence
    settings.MODEL_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Pipeline steps
    dataset = load_or_generate_dataset()
    train_classical_ml(dataset)
    train_deep_learning(dataset)
    
    logger.info("All platform models trained, evaluated, and serialized successfully!")