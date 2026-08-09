# models/__init__.py
"""
Models module initialization for the Weather Intelligence and Climate Decision Support Platform.
Exposes ML, DL, and CV prediction engines for precipitation and satellite analysis.
"""

# Import the model classes or functions from their respective modules
from models.ml_engine import MLEngine  # Class managing classical ML (XGBoost, Random Forest, etc.)
from models.dl_engine import DLEngine  # Class managing PyTorch LSTM/GRU network workflows
from models.cv_engine import SatelliteImageClassifier  # Class managing U-Net / CNN segmentation and detection

# Define the package's public API
__all__ = [
    "MLEngine",
    "DLEngine",
    "SatelliteImageClassifier"
]