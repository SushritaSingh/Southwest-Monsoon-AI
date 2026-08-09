# models/cv_engine.py
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger("monsoon_cv_engine")

class DoubleConv(nn.Module):
    """[Double Convolution -> Batch Normalization -> ReLU] block for U-Net architecture."""
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MonsoonUNet(nn.Module):
    """
    U-Net Semantic Segmentation model for processing high-resolution imagery 
    to output cloud cover segmentation masks or flood-inundation vectors.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super(MonsoonUNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down-sampling/contracting path
        features = [64, 128, 256, 512]
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up-sampling/expansive path
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(512, 1024)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            # Pad input features if dimension mismatches happen
            if x.shape != skip_connection.shape:
                diff_y = skip_connection.size()[2] - x.size()[2]
                diff_x = skip_connection.size()[3] - x.size()[3]
                x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
                
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return torch.sigmoid(self.final_conv(x))


class SatelliteImageClassifier:
    """Serves high-level vision pipeline inferences (e.g., Cyclone/Storm Detection Classification)."""
    def __init__(self):
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Simple binary representation: normal vs active-storm cloud profile
        self.labels = ["Clear / Non-Storm Cloud Layer", "Active Cyclonic Vorticity Found"]

    def infer(self, pil_image: Image.Image) -> Tuple[str, float]:
        """Runs image classifier preprocessing and feeds logits to score probabilities."""
        tensor_img = self.transform(pil_image).unsqueeze(0)
        
        # Simulating classification layer inference 
        with torch.no_grad():
            score = float(torch.sigmoid(tensor_img.sum() / 1e5).item())
            confidence = max(score, 1 - score)
            predicted_class = self.labels[1] if score > 0.5 else self.labels[0]
            
        return predicted_class, confidence

    def predict(self, pil_image: Image.Image) -> Tuple[str, float]:
        """Fallback alias method to prevent AttributeError during import cache states."""
        return self.infer(pil_image)