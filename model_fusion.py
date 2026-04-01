import torch
import torch.nn as nn
from model_eegnet import EEGNet

class EEGFusionNet(nn.Module):
    def __init__(self, n_channels=19, band_dim=95):
        super().__init__()

        self.eegnet = EEGNet(n_channels=n_channels)

        self.band_mlp = nn.Sequential(
            nn.Linear(band_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 + 64, 64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x_time, x_band):
        z_time = self.eegnet.extract_features(x_time)   # (B, 32)
        z_band = self.band_mlp(x_band)                  # (B, 64)

        z = torch.cat([z_time, z_band], dim=1)          # (B, 96)
        return self.classifier(z).squeeze(1)
