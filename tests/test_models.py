# tests/test_models.py
"""
Unit and functional computational tests for indices and PyTorch model architectures.
"""

import numpy as np
from remote_sensing.processor import GeospatialProcessor
from models.dl_engine import DLEngine
import torch

def test_ndvi_equation():
    """Verifies the NDVI mathematical calculation output range."""
    processor = GeospatialProcessor()
    
    # Static arrays for Red and NIR values
    red = np.array([0.1, 0.2])
    nir = np.array([0.5, 0.6])
    
    ndvi = processor.calculate_ndvi(red, nir)
    
    # Expected: (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6 = 0.6667
    assert np.allclose(ndvi[0], 0.66666667)
    # NDVI values must always fall between -1.0 and 1.0
    assert np.all(ndvi >= -1.0)
    assert np.all(ndvi <= 1.0)

def test_deep_learning_shape():
    """Verifies the deep learning model output shape and gradient settings."""
    model = DLEngine(input_dim=4, hidden_dim=32, num_layers=1, output_dim=1)
    
    # Batch size: 2, sequence length: 5, features: 4
    sample_input = torch.randn(2, 5, 4)
    output = model(sample_input)
    
    assert output.shape == (2, 1)