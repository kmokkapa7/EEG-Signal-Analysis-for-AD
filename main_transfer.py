"""
Transfer Learning Training Script.

This script implements the full transfer learning pipeline:
1. Pre-train on Iraq dataset (40 features, 47K samples)
2. Fine-tune on OpenNeuro dataset (266 features, 6.4K samples)

Expected improvement: +2-5% over standalone models
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from load_iraq_dataset import load_iraq_dataset, split_iraq_dataset
from load_openneuro import load_openneuro_dataset
from features_bandpower_enhanced import EnhancedBandpowerExtractor
from preprocess import subject_channel_zscore
from model_transfer import get_transfer_model


def train_base_model(model, train_loader, val_loader, epochs=100, lr=1e-3,
                     device='cuda', patience=15, save_path='base_model.pt'):
    """
    Train the base model on Iraq dataset.

    Args:
        model: TransferLearningBase model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        epochs: Maximum number of epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'
        patience: Early stopping patience
        save_path: Path to save best model

    Returns:
        best_val_acc: Best validation accuracy
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n  Training base model on Iraq dataset...")

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
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            print(f"    Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Best: {best_val_acc:.4f}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    return best_val_acc


def finetune_model(model, train_loader, val_loader, epochs=50, lr=1e-4,
                  device='cuda', patience=10, save_path='finetuned_model.pt',
                  unfreeze_after=None):
    """
    Fine-tune the model on OpenNeuro dataset.

    Args:
        model: TransferLearningFinetuned model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        epochs: Maximum number of epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'
        patience: Early stopping patience
        save_path: Path to save best model
        unfreeze_after: Unfreeze base model after this many epochs (None = keep frozen)

    Returns:
        best_val_acc: Best validation accuracy
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n  Fine-tuning model on OpenNeuro dataset...")

    for epoch in range(epochs):
        # Unfreeze base model if requested
        if unfreeze_after is not None and epoch == unfreeze_after:
            print(f"    [INFO] Unfreezing base model at epoch {epoch}")
            model.unfreeze_base()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr*0.1, weight_decay=1e-4)

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
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 5 == 0 or patience_counter == 0:
            print(f"    Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Best: {best_val_acc:.4f}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    return best_val_acc


def evaluate_model(model, X_test, y_test, subjects_test, device='cuda', model_path=None):
    """
    Evaluate model on test set with subject-level aggregation.

    Args:
        model: PyTorch model
        X_test: Test features
        y_test: Test labels
        subjects_test: Test subject IDs
        device: 'cuda' or 'cpu'
        model_path: Path to saved model (optional)

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    model.eval()

    # Create DataLoader
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Window-level predictions
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_preds.extend(preds)
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Subject-level aggregation (majority voting)
    unique_subjects = np.unique(subjects_test)
    subject_preds = []
    subject_labels = []

    for subject in unique_subjects:
        subject_mask = subjects_test == subject
        subject_prob = all_probs[subject_mask].mean()
        subject_pred = int(subject_prob > 0.5)
        subject_label = int(y_test[subject_mask][0])

        subject_preds.append(subject_pred)
        subject_labels.append(subject_label)

    subject_preds = np.array(subject_preds)
    subject_labels = np.array(subject_labels)

    # Calculate metrics
    acc = accuracy_score(subject_labels, subject_preds)
    auc = roc_auc_score(subject_labels, [all_probs[subjects_test == s].mean() for s in unique_subjects])
    cm = confusion_matrix(subject_labels, subject_preds)

    # Sensitivity and Specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n" + "="*50)
    print("SUBJECT-LEVEL EVALUATION RESULTS")
    print("="*50)
    print(f"Number of subjects: {len(unique_subjects)}")
    print(f"  - Class 0 (Control): {(subject_labels == 0).sum()}")
    print(f"  - Class 1 (AD): {(subject_labels == 1).sum()}")

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
    print(classification_report(subject_labels, subject_preds, target_names=['Control', 'AD']))

    return {
        'accuracy': acc,
        'auc_roc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm
    }


def main():
    print("="*70)
    print("TRANSFER LEARNING: IRAQ -> OPENNEURO")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # -------- CROSS-VALIDATION --------
    seeds = range(3)  # Start with 3-seed for faster testing
    all_metrics = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}/{len(seeds)-1}")
        print(f"{'='*70}")

        # ========================================
        # STAGE 1: PRE-TRAIN ON IRAQ DATASET
        # ========================================
        print("\n[STAGE 1/2] PRE-TRAINING ON IRAQ DATASET")
        print(f"{'='*70}")

        print("\n  [A] Loading Iraq dataset...")
        X_iraq, y_iraq, sample_ids_iraq = load_iraq_dataset(
            csv_path="EEG_AD_Iraq/HMMS.csv",
            binary=True,
            balance_classes=True,
            random_state=seed
        )

        print("\n  [B] Splitting Iraq dataset...")
        Xtr_iraq, Xval_iraq, Xte_iraq, ytr_iraq, yval_iraq, yte_iraq = split_iraq_dataset(
            X_iraq, y_iraq, sample_ids_iraq,
            test_size=0.2,
            val_size=0.2,
            random_state=seed
        )

        # Create DataLoaders
        train_dataset_iraq = TensorDataset(
            torch.tensor(Xtr_iraq, dtype=torch.float32),
            torch.tensor(ytr_iraq, dtype=torch.long)
        )
        val_dataset_iraq = TensorDataset(
            torch.tensor(Xval_iraq, dtype=torch.float32),
            torch.tensor(yval_iraq, dtype=torch.long)
        )

        train_loader_iraq = DataLoader(train_dataset_iraq, batch_size=128, shuffle=True)
        val_loader_iraq = DataLoader(val_dataset_iraq, batch_size=256, shuffle=False)

        print("\n  [C] Training base model on Iraq dataset...")
        base_model = get_transfer_model(variant='base', dropout=0.3)
        print(f"    Base model parameters: {sum(p.numel() for p in base_model.parameters()):,}")

        best_val_acc_iraq = train_base_model(
            base_model, train_loader_iraq, val_loader_iraq,
            epochs=100,
            lr=1e-3,
            device=device,
            patience=15,
            save_path=f'base_model_seed{seed}.pt'
        )

        print(f"\n  [STAGE 1 COMPLETE] Best Iraq validation accuracy: {best_val_acc_iraq:.4f}")

        # Load best base model
        base_model.load_state_dict(torch.load(f'base_model_seed{seed}.pt', map_location=device))

        # ========================================
        # STAGE 2: FINE-TUNE ON OPENNEURO DATASET
        # ========================================
        print(f"\n{'='*70}")
        print("[STAGE 2/2] FINE-TUNING ON OPENNEURO DATASET")
        print(f"{'='*70}")

        print("\n  [A] Loading OpenNeuro dataset...")
        X_time, y_openneuro, subjects_openneuro = load_openneuro_dataset(
            root_dir="dataset"
        )
        print(f"    Loaded {len(X_time)} windows from {len(np.unique(subjects_openneuro))} subjects")

        # Set random seed for reproducibility
        np.random.seed(seed)

        print("\n  [B] Normalizing time-domain signals...")
        X_time = subject_channel_zscore(X_time, subjects_openneuro)

        print("\n  [C] Splitting OpenNeuro dataset (subject-level)...")
        unique_subjects = np.unique(subjects_openneuro)
        np.random.seed(seed)
        np.random.shuffle(unique_subjects)

        n_subjects = len(unique_subjects)
        n_train = int(0.6 * n_subjects)
        n_val = int(0.2 * n_subjects)

        train_subjects = unique_subjects[:n_train]
        val_subjects = unique_subjects[n_train:n_train+n_val]
        test_subjects = unique_subjects[n_train+n_val:]

        train_mask = np.isin(subjects_openneuro, train_subjects)
        val_mask = np.isin(subjects_openneuro, val_subjects)
        test_mask = np.isin(subjects_openneuro, test_subjects)

        Xtr_time = X_time[train_mask]
        Xval_time = X_time[val_mask]
        Xte_time = X_time[test_mask]

        ytr_openneuro = y_openneuro[train_mask]
        yval_openneuro = y_openneuro[val_mask]
        yte_openneuro = y_openneuro[test_mask]

        subjects_tr = subjects_openneuro[train_mask]
        subjects_val = subjects_openneuro[val_mask]
        subjects_te = subjects_openneuro[test_mask]

        print(f"    Train: {len(train_subjects)} subjects, {len(Xtr_time)} windows")
        print(f"    Val:   {len(val_subjects)} subjects, {len(Xval_time)} windows")
        print(f"    Test:  {len(test_subjects)} subjects, {len(Xte_time)} windows")

        print("\n  [D] Extracting enhanced bandpower features...")
        extractor = EnhancedBandpowerExtractor(include_coherence=True)
        Xtr_band = extractor.fit_transform(Xtr_time)
        Xval_band = extractor.transform(Xval_time)
        Xte_band = extractor.transform(Xte_time)
        print(f"    Feature dimension: {Xtr_band.shape[1]}")

        # Create DataLoaders for OpenNeuro
        train_dataset_openneuro = TensorDataset(
            torch.tensor(Xtr_band, dtype=torch.float32),
            torch.tensor(ytr_openneuro, dtype=torch.long)
        )
        val_dataset_openneuro = TensorDataset(
            torch.tensor(Xval_band, dtype=torch.float32),
            torch.tensor(yval_openneuro, dtype=torch.long)
        )

        train_loader_openneuro = DataLoader(train_dataset_openneuro, batch_size=64, shuffle=True)
        val_loader_openneuro = DataLoader(val_dataset_openneuro, batch_size=128, shuffle=False)

        print("\n  [E] Creating fine-tuned model...")
        finetuned_model = get_transfer_model(
            variant='finetuned',
            pretrained_model=base_model,
            target_input_dim=Xtr_band.shape[1],
            freeze_base=True,
            dropout=0.3
        )
        total_params = sum(p.numel() for p in finetuned_model.parameters())
        trainable_params = sum(p.numel() for p in finetuned_model.parameters() if p.requires_grad)
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")
        print(f"    Frozen parameters: {total_params - trainable_params:,}")

        print("\n  [F] Fine-tuning on OpenNeuro dataset...")
        best_val_acc_openneuro = finetune_model(
            finetuned_model, train_loader_openneuro, val_loader_openneuro,
            epochs=50,
            lr=1e-4,
            device=device,
            patience=10,
            save_path=f'finetuned_model_seed{seed}.pt',
            unfreeze_after=None  # Keep base frozen
        )

        print(f"\n  [STAGE 2 COMPLETE] Best OpenNeuro validation accuracy: {best_val_acc_openneuro:.4f}")

        # ========================================
        # EVALUATION ON TEST SET
        # ========================================
        print(f"\n{'='*70}")
        print("[EVALUATION] TESTING ON HELD-OUT OPENNEURO TEST SET")
        print(f"{'='*70}")

        metrics = evaluate_model(
            finetuned_model, Xte_band, yte_openneuro, subjects_te,
            device=device,
            model_path=f'finetuned_model_seed{seed}.pt'
        )
        all_metrics.append(metrics)

    # ========================================
    # AGGREGATE RESULTS
    # ========================================
    print("\n" + "="*70)
    print(f"FINAL {len(seeds)}-SEED TRANSFER LEARNING RESULTS")
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

    # Comparison with baseline
    print("\n" + "="*70)
    print("COMPARISON WITH BASELINE")
    print("="*70)
    print(f"OpenNeuro Enhanced (baseline): 88.5% ± 7.1%")
    print(f"Transfer Learning (this run):  {accs.mean()*100:.1f}% ± {accs.std()*100:.1f}%")
    improvement = (accs.mean() - 0.885) * 100
    print(f"Improvement: {improvement:+.1f}%")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
