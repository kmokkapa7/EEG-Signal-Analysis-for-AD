import torch
import torch.nn as nn

class HMMSFeatureEncoder(nn.Module):
    def __init__(self, in_dim=40, out_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),

            nn.Linear(64, out_dim)
        )

        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z).squeeze(1)

    def extract_features(self, x):
        return self.encoder(x)
