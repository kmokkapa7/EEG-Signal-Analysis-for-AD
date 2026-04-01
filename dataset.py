import torch
from torch.utils.data import Dataset

class EEGMultiViewDataset(Dataset):
    def __init__(self, X_time, X_band, y):
        self.X_time = torch.tensor(X_time, dtype=torch.float32).unsqueeze(1)
        self.X_band = torch.tensor(X_band, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_time[idx], self.X_band[idx], self.y[idx]
