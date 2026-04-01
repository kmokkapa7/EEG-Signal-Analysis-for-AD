"""
Enhanced fusion model with support for larger bandpower feature dimension.
"""

import torch
import torch.nn as nn
from model_eegnet import EEGNet


class EEGFusionNetEnhanced(nn.Module):
    """
    Enhanced multi-view fusion model for EEG AD classification.

    Combines:
    - EEGNet for time-domain features (32 features)
    - Enhanced MLP for frequency-domain features (variable size)
    """

    def __init__(self, n_channels=19, band_dim=None):
        """
        Args:
            n_channels: Number of EEG channels
            band_dim: Dimension of bandpower features (auto-detected if None)
        """
        super().__init__()

        # Temporal feature extractor (EEGNet)
        self.eegnet = EEGNet(n_channels=n_channels)

        # Auto-detect band_dim based on enhanced features
        # Enhanced: 19 channels * (5 abs + 5 rel + 1 entropy + 3 ratios) = 266
        # Original: 19 channels * 5 = 95
        if band_dim is None:
            band_dim = n_channels * 14  # Default to enhanced features

        self.band_dim = band_dim

        # Frequency feature extractor (MLP)
        # Deeper network for enhanced features
        self.band_mlp = nn.Sequential(
            nn.Linear(band_dim, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU()
        )

        # Fusion classifier
        # Combines temporal (32) + frequency (64) = 96 features
        self.classifier = nn.Sequential(
            nn.Linear(32 + 64, 96),
            nn.ELU(),
            nn.Dropout(0.4),

            nn.Linear(96, 32),
            nn.ELU(),
            nn.Dropout(0.3),

            nn.Linear(32, 1)
        )

    def forward(self, x_time, x_band):
        """
        Forward pass.

        Args:
            x_time: (B, 1, T, C) time-domain EEG
            x_band: (B, band_dim) frequency features

        Returns:
            logits: (B,) classification logits
        """
        # Extract features from both views
        z_time = self.eegnet.extract_features(x_time)  # (B, 32)
        z_band = self.band_mlp(x_band)                  # (B, 64)

        # Concatenate and classify
        z = torch.cat([z_time, z_band], dim=1)          # (B, 96)
        return self.classifier(z).squeeze(1)
