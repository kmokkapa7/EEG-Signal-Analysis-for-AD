import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

def train_pretrain_model(
    model,
    dataset,
    epochs=20,
    batch_size=256,
    lr=1e-3
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    y_bin = (dataset.y > 0).float()

    idx = np.arange(len(dataset))
    tr, te = train_test_split(
        idx, test_size=0.2, stratify=y_bin.numpy(), random_state=42
    )

    train_ds = torch.utils.data.Subset(dataset, tr)
    val_ds   = torch.utils.data.Subset(dataset, te)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = (y > 0).float().to(device)

            logits = model(X)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = (y > 0).float().to(device)

                preds = (torch.sigmoid(model(X)) > 0.5)
                correct += (preds == y).sum().item()
                total += y.numel()

        print(f"Epoch {epoch+1}: Val Acc = {correct/total:.4f}")
