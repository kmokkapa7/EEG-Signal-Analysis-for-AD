import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class IraqEEGDataset(Dataset):
    """
    HMMS Iraq EEG dataset loader.
    - Expects 40 EEG feature columns: feature_1 ... feature_40
    - Label column: 'label' (string)
    - Supports binary or multiclass
    """

    LABEL_MAP = {
        "Healthy": 0,
        "Mild": 1,
        "Mid": 1,
        "Moderate": 2,
        "Severe": 3,
        "Sever": 3,     # typo in dataset
    }

    def __init__(self, csv_path, binary=True):
        df = pd.read_csv(csv_path)

        # ---------- LABEL ----------
        if "label" not in df.columns:
            raise ValueError("Expected column 'label' in HMMS.csv")

        y_raw = df["label"].astype(str).str.strip()

        y = []
        keep_idx = []

        for i, v in enumerate(y_raw):
            if v in self.LABEL_MAP:
                y.append(self.LABEL_MAP[v])
                keep_idx.append(i)

        y = np.array(y, dtype=np.int64)

        if binary:
            y = (y > 0).astype(np.float32)   # AD vs Healthy

        # ---------- FEATURES ----------
        feature_cols = [c for c in df.columns if c.startswith("feature_")]

        if len(feature_cols) != 40:
            raise ValueError(
                f"Expected 40 feature columns, found {len(feature_cols)}"
            )

        X = df.loc[keep_idx, feature_cols].values.astype(np.float32)

        # ---------- NORMALIZATION ----------
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

        # ---------- TORCH ----------
        self.X = torch.tensor(X)        # (N, 40)
        self.y = torch.tensor(y)        # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
