# models/dl_engine.py
"""
Deep Learning Engine for the Weather Intelligence Platform.
Implements a PyTorch recurrent neural network architecture (LSTM/GRU)
for sequential time-series meteorological forecasting.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

class DLEngine(nn.Module):
    """
    A PyTorch-based sequential neural network model designed for handling 
    time-series climate sequences.
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1):
        super(DLEngine, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM core sequential architecture
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer to map to final continuous prediction target (e.g., rainfall mm)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes forward pass computational graph operations."""
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass through the sequence array
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output hidden state of the very last time step
        out = self.fc(out[:, -1, :])
        return out