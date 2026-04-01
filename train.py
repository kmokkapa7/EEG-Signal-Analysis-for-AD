import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_model(
    model,
    train_ds,
    val_ds,
    epochs=50,
    batch_size=64,
    lr=1e-4,
    seed=42,
    save_path="best_model.pt"
):
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # -------- UNFREEZE ALL LAYERS (IMPROVED) --------
    # Previously temporal layers were frozen, now we fine-tune everything
    # Use differential learning rates: lower for pretrained temporal, higher for rest
    for param in model.parameters():
        param.requires_grad = True
    # -----------------------------------------------

    y_train = train_ds.y.numpy()

    classes = np.array([0, 1], dtype=int)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    pos_weight = torch.tensor(
        weights[1] / weights[0],
        dtype=torch.float32
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Differential learning rates: lower for pretrained temporal layers
    optimizer = torch.optim.Adam([
        {'params': model.eegnet.temporal.parameters(), 'lr': lr * 0.1},  # 10x slower for pretrained
        {'params': model.eegnet.spatial.parameters(), 'lr': lr},
        {'params': model.eegnet.separable.parameters(), 'lr': lr},
        {'params': model.band_mlp.parameters(), 'lr': lr},
        {'params': model.classifier.parameters(), 'lr': lr}
    ], lr=lr)

    # Learning rate scheduler: reduce LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # Drop last incomplete batch to avoid BatchNorm errors
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        drop_last=False
    )

    best_loss = np.inf
    patience = 10  # Increased patience since we have LR scheduling
    counter = 0

    for epoch in range(epochs):
        # -------- TRAIN --------
        model.train()
        for X_time, X_band, y in train_loader:
            X_time = X_time.to(device)
            X_band = X_band.to(device)
            y = y.to(device).view(-1)

            logits = model(X_time, X_band)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # -------- VALIDATE --------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_time, X_band, y in val_loader:
                X_time = X_time.to(device)
                X_band = X_band.to(device)
                y = y.to(device).view(-1)

                logits = model(X_time, X_band)
                val_loss += criterion(logits, y).item()

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    # Return the path to the best model
    return save_path
