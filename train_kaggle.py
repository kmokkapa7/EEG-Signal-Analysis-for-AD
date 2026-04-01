import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

def train_kaggle_model(
    model,
    dataset,
    epochs=10,
    batch_size=1024,
    lr=1e-3
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=dataset.y.numpy(),
        random_state=42
    )

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds   = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            logits = model(X).view(-1)
            y = y.view(-1)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = (torch.sigmoid(model(X)) > 0.5).view(-1)
                correct += (preds == y).sum().item()
                total += y.numel()

        acc = correct / total
        print(f"Epoch {epoch+1}: Val Accuracy = {acc:.4f}")
