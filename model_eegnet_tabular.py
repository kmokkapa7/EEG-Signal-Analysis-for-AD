import torch
import torch.nn as nn

class EEGNetTabular(nn.Module):
    def __init__(self, n_channels=16):
        super().__init__()

        self.spatial = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(16, 1)

    def forward(self, x):
        # x: (B, 1, C, 1)
        x = self.spatial(x)      # (B, 16, 1, 1)
        x = x.flatten(1)         # (B, 16)
        return self.classifier(x)
