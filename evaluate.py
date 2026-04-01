import numpy as np
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, precision_recall_fscore_support,
    roc_curve
)

def evaluate_subject_level(model, X_time, X_band, y, subjects, model_path=None):
    """
    Evaluate model at subject level with comprehensive metrics.

    Args:
        model: Model to evaluate
        X_time: Time-domain features
        X_band: Bandpower features
        y: Labels
        subjects: Subject IDs
        model_path: Path to saved model weights (if None, uses current model state)

    Returns:
        dict: Dictionary containing all metrics
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load best model if path provided
    if model_path is not None:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Using current model state.")

    model.eval()

    # Get predictions
    with torch.no_grad():
        logits = model(
            torch.tensor(X_time, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(X_band, dtype=torch.float32).to(device)
        )
        probs = torch.sigmoid(logits).cpu().numpy()

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

    # Handle case where only one class is present
    try:
        auc = roc_auc_score(subj_true, subj_probs)
    except ValueError:
        auc = float('nan')
        print("Warning: AUC cannot be calculated (only one class present)")

    precision, recall, f1, _ = precision_recall_fscore_support(
        subj_true, subj_preds, average='binary', zero_division=0
    )

    # Calculate sensitivity (recall for class 1) and specificity
    tn, fp, fn, tp = confusion_matrix(subj_true, subj_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Print results
    print("\n" + "="*50)
    print("SUBJECT-LEVEL EVALUATION RESULTS")
    print("="*50)
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
    print("="*50)

    # Return metrics dictionary
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
