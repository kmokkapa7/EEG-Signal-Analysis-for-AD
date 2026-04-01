"""
Deep neural network for Iraq EEG dataset (40 pre-extracted features).

Since the Iraq dataset provides preprocessed features (not raw signals),
we use a deep MLP instead of the multi-view fusion architecture.
"""

import torch
import torch.nn as nn


class IraqEEGNet(nn.Module):
    """
    Deep MLP for binary AD classification using Iraq dataset features.

    Architecture based on the benchmark paper's DNN (96.05% accuracy).
    """

    def __init__(self, input_dim=40, hidden_dims=[128, 256, 128, 64], dropout=0.4):
        """
        Args:
            input_dim: Number of input features (40 for Iraq dataset)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability for regularization
        """
        super().__init__()

        self.input_dim = input_dim

        # Build deep MLP
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(nn.ReLU())

            # Dropout (skip for last layer)
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(prev_dim, 1)  # Binary classification
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, 40) input features

        Returns:
            logits: (B,) classification logits
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features).squeeze(1)
        return logits


class IraqEEGNetLarge(nn.Module):
    """
    Larger variant for potentially higher accuracy.
    """

    def __init__(self, input_dim=40, dropout=0.3):
        super().__init__()

        self.model = nn.Sequential(
            # Layer 1: 40 → 256
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2: 256 → 512
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3: 512 → 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 4: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 5: 128 → 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Output: 64 → 1
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)


def get_iraq_model(variant='standard', input_dim=40):
    """
    Factory function to create Iraq EEG models.

    Args:
        variant: 'standard' or 'large'
        input_dim: Number of input features

    Returns:
        Model instance
    """
    if variant == 'standard':
        return IraqEEGNet(input_dim=input_dim)
    elif variant == 'large':
        return IraqEEGNetLarge(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    # Test models
    batch_size = 32
    x = torch.randn(batch_size, 40)

    print("Testing IraqEEGNet (standard):")
    model_std = IraqEEGNet()
    out = model_std(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_std.parameters()):,}")

    print("\nTesting IraqEEGNetLarge:")
    model_large = IraqEEGNetLarge()
    out = model_large(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_large.parameters()):,}")
