"""
Load and preprocess the EEG_AD_Iraq HMMS dataset.

This module loads the preprocessed HMMS.csv file and converts it to binary classification:
- Healthy → 0 (Control)
- Mild/Moderate/Sever → 1 (AD)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_iraq_dataset(csv_path="EEG_AD_Iraq/HMMS.csv", binary=True, balance_classes=True, random_state=42):
    """
    Load the Iraq EEG dataset from HMMS.csv.

    Args:
        csv_path: Path to HMMS.csv file
        binary: If True, convert to binary (Healthy vs AD). If False, keep 4 classes.
        balance_classes: If True, balance Healthy vs AD samples (recommended)
        random_state: Random seed for reproducibility

    Returns:
        X: Feature array (N, 40)
        y: Labels (N,) - 0=Healthy, 1=AD for binary
        sample_ids: Original indices from CSV
    """
    print(f"Loading Iraq dataset from {csv_path}...")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Extract features (columns 1-40, skip 'index' column)
    feature_cols = [f'feature_{i}' for i in range(1, 41)]
    X = df[feature_cols].values.astype(np.float32)

    # Extract labels
    labels = df['label'].values
    sample_ids = df['index'].values

    print(f"  Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"  Original label distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"    {label}: {count}")

    if binary:
        # Convert to binary classification
        y = np.zeros(len(labels), dtype=np.int64)
        y[labels != 'Healthy'] = 1  # All AD stages → 1

        print(f"\n  Binary label distribution:")
        print(f"    Control (Healthy): {(y == 0).sum()}")
        print(f"    AD (Mild/Moderate/Severe): {(y == 1).sum()}")

        if balance_classes:
            # Balance dataset: downsample AD to match Healthy
            healthy_idx = np.where(y == 0)[0]
            ad_idx = np.where(y == 1)[0]

            n_healthy = len(healthy_idx)
            n_ad = len(ad_idx)

            if n_ad > n_healthy:
                # Downsample AD to match Healthy
                np.random.seed(random_state)
                ad_idx_balanced = np.random.choice(ad_idx, size=n_healthy, replace=False)

                # Combine balanced indices
                balanced_idx = np.concatenate([healthy_idx, ad_idx_balanced])
                np.random.shuffle(balanced_idx)

                X = X[balanced_idx]
                y = y[balanced_idx]
                sample_ids = sample_ids[balanced_idx]

                print(f"\n  Balanced dataset:")
                print(f"    Control: {(y == 0).sum()}")
                print(f"    AD: {(y == 1).sum()}")
                print(f"    Total: {len(y)} samples")
    else:
        # Multi-class: encode as 0=Healthy, 1=Mild, 2=Moderate, 3=Severe
        label_map = {'Healthy': 0, 'Mild': 1, 'Moderate': 2, 'Sever': 3}
        y = np.array([label_map[label] for label in labels], dtype=np.int64)

    return X, y, sample_ids


def split_iraq_dataset(X, y, sample_ids, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split Iraq dataset into train/val/test sets.

    Note: This is sample-level split (not subject-level) since HMMS.csv
    doesn't contain subject identifiers.

    Args:
        X: Features (N, 40)
        y: Labels (N,)
        sample_ids: Sample indices (N,)
        test_size: Fraction for test set
        val_size: Fraction of remaining for validation
        random_state: Random seed

    Returns:
        Xtr, Xval, Xte, ytr, yval, yte
    """
    # First split: train+val vs test
    Xtr_val, Xte, ytr_val, yte, ids_tr_val, ids_te = train_test_split(
        X, y, sample_ids,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    Xtr, Xval, ytr, yval, ids_tr, ids_val = train_test_split(
        Xtr_val, ytr_val, ids_tr_val,
        test_size=val_size_adjusted,
        stratify=ytr_val,
        random_state=random_state
    )

    print(f"\n  Dataset split:")
    print(f"    Train: {len(ytr)} samples (Control: {(ytr==0).sum()}, AD: {(ytr==1).sum()})")
    print(f"    Val:   {len(yval)} samples (Control: {(yval==0).sum()}, AD: {(yval==1).sum()})")
    print(f"    Test:  {len(yte)} samples (Control: {(yte==0).sum()}, AD: {(yte==1).sum()})")

    return Xtr, Xval, Xte, ytr, yval, yte


if __name__ == "__main__":
    # Test the loader
    X, y, sample_ids = load_iraq_dataset(binary=True, balance_classes=True)
    print(f"\nFinal dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Feature range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Feature mean: {X.mean():.2f}, std: {X.std():.2f}")
