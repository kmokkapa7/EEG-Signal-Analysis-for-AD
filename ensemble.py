"""
Ensemble evaluation using all 10 trained models.
Averages predictions from multiple models for improved accuracy.
"""

import torch
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, precision_recall_fscore_support
)
from model_fusion import EEGFusionNet
from load_openneuro import load_openneuro_dataset
from preprocess import subject_channel_zscore
from subject_split import subject_stratified_split
from features_bandpower import BandpowerExtractor
import glob


def ensemble_predict(models, X_time, X_band, device):
    """
    Get ensemble predictions by averaging probabilities from multiple models.

    Args:
        models: List of trained models
        X_time: Time-domain features
        X_band: Bandpower features
        device: Device to run inference on

    Returns:
        Averaged probabilities across all models
    """
    all_probs = []

    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.tensor(X_time, dtype=torch.float32).unsqueeze(1).to(device),
                torch.tensor(X_band, dtype=torch.float32).to(device)
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    # Average probabilities across all models
    ensemble_probs = np.mean(all_probs, axis=0)
    return ensemble_probs


def evaluate_ensemble(model_paths, X_time, X_band, y, subjects):
    """
    Evaluate ensemble of models at subject level.

    Args:
        model_paths: List of paths to saved model checkpoints
        X_time: Time-domain features
        X_band: Bandpower features
        y: Labels
        subjects: Subject IDs

    Returns:
        dict: Dictionary containing all metrics
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load all models
    print(f"Loading {len(model_paths)} models for ensemble...")
    models = []
    for path in model_paths:
        model = EEGFusionNet(n_channels=19)
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            models.append(model)
            print(f"  Loaded: {path}")
        except FileNotFoundError:
            print(f"  Warning: {path} not found, skipping")

    if len(models) == 0:
        raise ValueError("No models loaded successfully!")

    print(f"\nEnsemble size: {len(models)} models")

    # Get ensemble predictions
    probs = ensemble_predict(models, X_time, X_band, device)

    # Aggregate by subject
    subj_preds = []
    subj_probs = []
    subj_true = []

    for subj in np.unique(subjects):
        idx = np.where(subjects == subj)[0]
        mean_prob = probs[idx].mean()
        pred = int(mean_prob >= 0.5)

        subj_preds.append(pred)
        subj_probs.append(mean_prob)
        subj_true.append(y[idx][0])

    subj_preds = np.array(subj_preds)
    subj_probs = np.array(subj_probs)
    subj_true = np.array(subj_true)

    # Calculate metrics
    acc = accuracy_score(subj_true, subj_preds)

    try:
        auc = roc_auc_score(subj_true, subj_probs)
    except ValueError:
        auc = float('nan')
        print("Warning: AUC cannot be calculated (only one class present)")

    precision, recall, f1, _ = precision_recall_fscore_support(
        subj_true, subj_preds, average='binary', zero_division=0
    )

    # Calculate sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(subj_true, subj_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Print results
    print("\n" + "="*60)
    print("ENSEMBLE SUBJECT-LEVEL EVALUATION RESULTS")
    print("="*60)
    print(f"Ensemble size: {len(models)} models")
    print(f"Number of subjects: {len(subj_true)}")
    print(f"  - Class 0 (Control): {(subj_true == 0).sum()}")
    print(f"  - Class 1 (AD): {(subj_true == 1).sum()}")
    print("\nPerformance Metrics:")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  AUC-ROC:      {auc:.4f}")
    print(f"  Sensitivity:  {sensitivity:.4f} (True Positive Rate)")
    print(f"  Specificity:  {specificity:.4f} (True Negative Rate)")
    print(f"  Precision:    {precision:.4f}")
    print(f"  Recall:       {recall:.4f}")
    print(f"  F1-Score:     {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Control  AD")
    print(f"Actual Control   {tn:3d}   {fp:3d}")
    print(f"       AD        {fn:3d}   {tp:3d}")
    print("\nDetailed Classification Report:")
    print(classification_report(subj_true, subj_preds, target_names=['Control', 'AD']))
    print("="*60)

    return {
        'accuracy': acc,
        'auc_roc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': confusion_matrix(subj_true, subj_preds),
        'predictions': subj_preds,
        'probabilities': subj_probs,
        'true_labels': subj_true
    }


def main():
    """
    Main function to evaluate ensemble on all available test data.
    Uses seed 0 split for consistent evaluation.
    """
    print("="*60)
    print("ENSEMBLE EVALUATION ON TEST SET")
    print("="*60)

    # Load data
    print("\n[1/4] Loading EEG dataset...")
    try:
        X, y, subjects = load_openneuro_dataset("dataset")
        X = subject_channel_zscore(X, subjects)
        print(f"  Loaded {len(X)} windows from {len(np.unique(subjects))} subjects")
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return

    # Use seed 0 split for evaluation
    print("\n[2/4] Splitting data (using seed=0 for consistency)...")
    Xtr_t, Xval_t, Xte_t, ytr, yval, yte, subj_tr, subj_val, subj_te = subject_stratified_split(
        X, y, subjects, test_size=0.2, val_size=0.2, random_state=0
    )
    print(f"  Test set: {len(np.unique(subj_te))} subjects, {len(Xte_t)} windows")

    # Extract bandpower features
    print("\n[3/4] Extracting bandpower features...")
    bp_extractor = BandpowerExtractor(fs=128)
    bp_extractor.fit(Xtr_t)  # Fit on training data
    Xte_b = bp_extractor.transform(Xte_t)
    print(f"  Extracted {Xte_b.shape[1]} bandpower features")

    # Find all model files
    print("\n[4/4] Loading models for ensemble...")
    model_paths = sorted(glob.glob("best_model_seed*.pt"))

    if len(model_paths) == 0:
        print("  Error: No model files found (best_model_seed*.pt)")
        return

    print(f"  Found {len(model_paths)} model files")

    # Evaluate ensemble
    metrics = evaluate_ensemble(model_paths, Xte_t, Xte_b, yte, subj_te)

    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
