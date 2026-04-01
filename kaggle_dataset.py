import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class KaggleEEGDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        X = df.drop(columns=["status"]).values.astype(np.float32)
        y = df["status"].values.astype(np.float32)

        # Z-score per channel (global)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

        self.X = torch.tensor(X).unsqueeze(-1).unsqueeze(1)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
