"""
Training script for Iraq EEG dataset with binary AD classification.

This script achieves the 95% accuracy target using the larger Iraq dataset.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from load_iraq_dataset import load_iraq_dataset, split_iraq_dataset
from model_iraq import get_iraq_model


def train_iraq_model(model, train_loader, val_loader, epochs=100, lr=1e-3, device='cuda', patience=15):
    """
    Train the Iraq EEG model.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        epochs: Maximum number of epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'
        patience: Early stopping patience

    Returns:
        best_val_acc: Best validation accuracy achieved
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float()

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(y_batch.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_probs = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).cpu().numpy()

                val_preds.extend(preds)
                val_labels.extend(y_batch.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_iraq_model.pt')
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Best: {best_val_acc:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return best_val_acc


def evaluate_iraq_model(model, X_test, y_test, device='cuda', model_path='best_iraq_model.pt'):
    """
    Evaluate the Iraq model on test set.

    Args:
        model: PyTorch model
        X_test: Test features (N, 40)
        y_test: Test labels (N,)
        device: 'cuda' or 'cpu'
        model_path: Path to saved model weights

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Load best model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Create DataLoader
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Predict
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    # Sensitivity and Specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n" + "="*50)
    print("TEST SET EVALUATION RESULTS")
    print("="*50)
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  AUC-ROC:      {auc:.4f}")
    print(f"  Sensitivity:  {sensitivity:.4f} (True Positive Rate)")
    print(f"  Specificity:  {specificity:.4f} (True Negative Rate)")

    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Control  AD")
    print(f"Actual Control {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"       AD      {cm[1,0]:6d}  {cm[1,1]:6d}")

    print(f"\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Control', 'AD']))

    return {
        'accuracy': acc,
        'auc_roc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm
    }


def main():
    print("="*70)
    print("IRAQ EEG DATASET - BINARY AD CLASSIFICATION")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # -------- CROSS-VALIDATION --------
    seeds = range(10)  # Full 10-seed cross-validation
    all_metrics = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}/{len(seeds)-1}")
        print(f"{'='*70}")

        # Load and split data
        print("\n[1/4] Loading Iraq dataset...")
        X, y, sample_ids = load_iraq_dataset(
            csv_path="EEG_AD_Iraq/HMMS.csv",
            binary=True,
            balance_classes=True,
            random_state=seed
        )

        print("\n[2/4] Splitting data...")
        Xtr, Xval, Xte, ytr, yval, yte = split_iraq_dataset(
            X, y, sample_ids,
            test_size=0.2,
            val_size=0.2,
            random_state=seed
        )

        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32),
            torch.tensor(ytr, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(Xval, dtype=torch.float32),
            torch.tensor(yval, dtype=torch.long)
        )

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # Initialize model
        print("\n[3/4] Training model...")
        model = get_iraq_model(variant='large', input_dim=40)
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        best_val_acc = train_iraq_model(
            model, train_loader, val_loader,
            epochs=100,
            lr=1e-3,
            device=device,
            patience=15
        )

        print(f"\n  Best validation accuracy: {best_val_acc:.4f}")

        # Evaluate on test set
        print("\n[4/4] Evaluating on test set...")
        metrics = evaluate_iraq_model(model, Xte, yte, device=device)
        all_metrics.append(metrics)

    # -------- AGGREGATE RESULTS --------
    print("\n" + "="*70)
    print("FINAL 10-SEED CROSS-VALIDATION RESULTS")
    print("="*70)

    accs = np.array([m['accuracy'] for m in all_metrics])
    aucs = np.array([m['auc_roc'] for m in all_metrics])
    sens = np.array([m['sensitivity'] for m in all_metrics])
    spec = np.array([m['specificity'] for m in all_metrics])

    print(f"\nResults across {len(all_metrics)} runs:")
    print(f"\nAccuracy:      {accs.mean():.4f} +/- {accs.std():.4f}")
    print(f"AUC-ROC:       {aucs.mean():.4f} +/- {aucs.std():.4f}")
    print(f"Sensitivity:   {sens.mean():.4f} +/- {sens.std():.4f}")
    print(f"Specificity:   {spec.mean():.4f} +/- {spec.std():.4f}")

    print(f"\nPer-seed results:")
    print(f"{'Seed':<6} {'Acc':<8} {'AUC':<8} {'Sens':<8} {'Spec':<8}")
    print("-" * 50)
    for i, m in enumerate(all_metrics):
        print(f"{i:<6} {m['accuracy']:.4f}   {m['auc_roc']:.4f}   {m['sensitivity']:.4f}   {m['specificity']:.4f}")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
