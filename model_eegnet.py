import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, n_channels=19):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(64, 1), padding=(32, 0), bias=False),
            nn.BatchNorm2d(16)
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(
                16, 32,
                kernel_size=(1, n_channels),
                groups=16,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((4, 1)),
            nn.Dropout(0.5)
        )

        self.separable = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(16, 1), padding=(8, 0), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((4, 1)),
            nn.Dropout(0.5)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # classifier kept for non-fusion use
        self.classifier = nn.Linear(32, 1)

    def extract_features(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        x = self.adaptive_pool(x)
        return x.view(x.size(0), -1)  # (B, 32)

    def forward(self, x):
        feats = self.extract_features(x)
        return self.classifier(feats).squeeze(1)
