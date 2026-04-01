import numpy as np
import torch
from torch.utils.data import Dataset

class IntegratedEEGDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)

        X = data["X_raw"].astype(np.float32)   # (N, 128, 19)
        y = data["y_labels"]

        # ---- Binary label mapping ----
        # AD = 1, Control = 0
        ad_labels = {"1", "1.0", "2"}   # AD-related
        ctrl_labels = {"0", "0.0"}

        y_bin = []
        for row in y:
            label = row[0]
            if label in ad_labels:
                y_bin.append(1)
            elif label in ctrl_labels:
                y_bin.append(0)
            else:
                y_bin.append(None)

        X = X[np.array([v is not None for v in y_bin])]
        y = np.array([v for v in y_bin if v is not None], dtype=np.float32)

        # Z-score per channel
        mean = X.mean(axis=(0, 1), keepdims=True)
        std = X.std(axis=(0, 1), keepdims=True) + 1e-6
        X = (X - mean) / std

        self.X = torch.tensor(X).unsqueeze(1)  # (B, 1, 128, 19)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
