"""
Transfer Learning Architecture for EEG AD Classification.

This module implements a transfer learning approach that:
1. Pre-trains on Iraq dataset (40 features, 47K samples)
2. Fine-tunes on OpenNeuro dataset (266 features, 6.4K samples)
"""

import torch
import torch.nn as nn


class TransferLearningBase(nn.Module):
    """
    Base network trained on Iraq dataset.

    This network learns general AD patterns from the larger Iraq dataset.
    The learned representations will be transferred to OpenNeuro.
    """

    def __init__(self, input_dim=40, hidden_dims=[256, 512, 256, 128], dropout=0.3):
        """
        Args:
            input_dim: Number of input features (40 for Iraq)
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim

        # Feature extractor (will be transferred)
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Classification head (will be replaced for OpenNeuro)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(prev_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (B, input_dim) input features

        Returns:
            logits: (B,) classification logits
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features).squeeze(1)
        return logits

    def get_feature_dim(self):
        """Get the output dimension of feature extractor."""
        return self.classifier[1].in_features


class TransferLearningFinetuned(nn.Module):
    """
    Fine-tuned network for OpenNeuro dataset.

    This network adapts the pre-trained Iraq model to OpenNeuro's
    266 features by adding an adapter layer.
    """

    def __init__(self, pretrained_model, target_input_dim=266, freeze_base=True, dropout=0.3):
        """
        Args:
            pretrained_model: Pre-trained TransferLearningBase model
            target_input_dim: Input dimension for OpenNeuro (266)
            freeze_base: If True, freeze pretrained feature extractor
            dropout: Dropout probability
        """
        super().__init__()

        self.target_input_dim = target_input_dim
        self.freeze_base = freeze_base

        # Adapter: Projects OpenNeuro features (266) to Iraq features (40)
        self.adapter = nn.Sequential(
            nn.Linear(target_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, pretrained_model.input_dim)
        )

        # Transfer pretrained feature extractor
        self.feature_extractor = pretrained_model.feature_extractor

        # Freeze feature extractor if requested
        if freeze_base:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # New classification head for OpenNeuro
        feature_dim = pretrained_model.get_feature_dim()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 266) OpenNeuro features

        Returns:
            logits: (B,) classification logits
        """
        # Adapt OpenNeuro features to Iraq space
        x_adapted = self.adapter(x)

        # Extract features using pretrained network
        features = self.feature_extractor(x_adapted)

        # Classify
        logits = self.classifier(features).squeeze(1)
        return logits

    def unfreeze_base(self):
        """Unfreeze the pretrained feature extractor for full fine-tuning."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        self.freeze_base = False


class TransferLearningDirect(nn.Module):
    """
    Alternative approach: Direct feature alignment without adapter.

    This network learns to map OpenNeuro's 266 features directly to
    the pretrained feature space.
    """

    def __init__(self, pretrained_model, target_input_dim=266, freeze_base=True, dropout=0.3):
        """
        Args:
            pretrained_model: Pre-trained TransferLearningBase model
            target_input_dim: Input dimension for OpenNeuro (266)
            freeze_base: If True, freeze pretrained layers
            dropout: Dropout probability
        """
        super().__init__()

        self.target_input_dim = target_input_dim
        self.freeze_base = freeze_base

        # Feature projection layer
        self.projection = nn.Sequential(
            nn.Linear(target_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Transfer pretrained feature extractor (skip first layer)
        self.feature_extractor = nn.Sequential(*list(pretrained_model.feature_extractor.children())[4:])

        # Freeze feature extractor if requested
        if freeze_base:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # New classification head
        feature_dim = pretrained_model.get_feature_dim()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 266) OpenNeuro features

        Returns:
            logits: (B,) classification logits
        """
        # Project to feature space
        x_projected = self.projection(x)

        # Extract features using pretrained network
        features = self.feature_extractor(x_projected)

        # Classify
        logits = self.classifier(features).squeeze(1)
        return logits

    def unfreeze_base(self):
        """Unfreeze the pretrained feature extractor."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        self.freeze_base = False


def get_transfer_model(variant='finetuned', pretrained_model=None, target_input_dim=266,
                      freeze_base=True, dropout=0.3):
    """
    Factory function to create transfer learning models.

    Args:
        variant: 'base', 'finetuned', or 'direct'
        pretrained_model: Pre-trained base model (required for 'finetuned' and 'direct')
        target_input_dim: Input dimension for fine-tuning
        freeze_base: Whether to freeze pretrained layers
        dropout: Dropout probability

    Returns:
        Model instance
    """
    if variant == 'base':
        return TransferLearningBase(dropout=dropout)
    elif variant == 'finetuned':
        if pretrained_model is None:
            raise ValueError("pretrained_model is required for 'finetuned' variant")
        return TransferLearningFinetuned(pretrained_model, target_input_dim, freeze_base, dropout)
    elif variant == 'direct':
        if pretrained_model is None:
            raise ValueError("pretrained_model is required for 'direct' variant")
        return TransferLearningDirect(pretrained_model, target_input_dim, freeze_base, dropout)
    else:
        raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    # Test the models
    print("=" * 70)
    print("TESTING TRANSFER LEARNING MODELS")
    print("=" * 70)

    # Test base model (Iraq)
    print("\n[1/3] Testing TransferLearningBase (Iraq dataset):")
    base_model = TransferLearningBase(input_dim=40)
    x_iraq = torch.randn(32, 40)
    out = base_model(x_iraq)
    print(f"  Input shape: {x_iraq.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in base_model.parameters() if p.requires_grad):,}")

    # Test finetuned model (OpenNeuro)
    print("\n[2/3] Testing TransferLearningFinetuned (OpenNeuro dataset):")
    finetuned_model = TransferLearningFinetuned(base_model, target_input_dim=266, freeze_base=True)
    x_openneuro = torch.randn(32, 266)
    out = finetuned_model(x_openneuro)
    print(f"  Input shape: {x_openneuro.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Total parameters: {sum(p.numel() for p in finetuned_model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in finetuned_model.parameters() if p.requires_grad):,}")
    print(f"  Frozen parameters: {sum(p.numel() for p in finetuned_model.parameters() if not p.requires_grad):,}")

    # Test direct model (alternative)
    print("\n[3/3] Testing TransferLearningDirect (alternative approach):")
    direct_model = TransferLearningDirect(base_model, target_input_dim=266, freeze_base=True)
    out = direct_model(x_openneuro)
    print(f"  Input shape: {x_openneuro.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Total parameters: {sum(p.numel() for p in direct_model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in direct_model.parameters() if p.requires_grad):,}")
    print(f"  Frozen parameters: {sum(p.numel() for p in direct_model.parameters() if not p.requires_grad):,}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
