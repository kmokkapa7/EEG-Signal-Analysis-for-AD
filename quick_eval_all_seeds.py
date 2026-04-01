"""
Quick evaluation of all 10 trained models on their respective test sets.
Loads dataset once and evaluates each seed.
"""

import torch
import numpy as np
from model_fusion import EEGFusionNet
from load_openneuro import load_openneuro_dataset
from preprocess import subject_channel_zscore
from subject_split import subject_stratified_split
from features_bandpower import BandpowerExtractor
from evaluate import evaluate_subject_level

def main():
    print("="*60)
    print("EVALUATING ALL 10 SEEDS")
    print("="*60)

    # Load dataset once
    print("\nLoading dataset...")
    X, y, subjects = load_openneuro_dataset('dataset')
    X = subject_channel_zscore(X, subjects)
    print(f"Loaded {len(X)} windows from {len(np.unique(subjects))} subjects")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results = []

    for seed in range(10):
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        # Split data for this seed
        Xtr_t, Xval_t, Xte_t, ytr, yval, yte, subj_tr, subj_val, subj_te = subject_stratified_split(
            X, y, subjects, test_size=0.2, val_size=0.2, random_state=seed
        )

        print(f"Test set: {len(np.unique(subj_te))} subjects, {len(Xte_t)} windows")

        # Extract bandpower features (fit on train, transform test)
        bp = BandpowerExtractor(fs=128)
        bp.fit(Xtr_t)
        Xte_b = bp.transform(Xte_t)

        # Load model for this seed
        model_path = f'best_model_seed{seed}.pt'
        model = EEGFusionNet(n_channels=19)

        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)

            # Evaluate
            metrics = evaluate_subject_level(model, Xte_t, Xte_b, yte, subj_te, model_path=model_path)
            results.append(metrics)

            # Print summary for this seed
            print(f"\nResults:")
            print(f"  Accuracy:     {metrics['accuracy']:.1%}")
            print(f"  AUC-ROC:      {metrics['auc_roc']:.1%}")
            print(f"  Sensitivity:  {metrics['sensitivity']:.1%}")
            print(f"  Specificity:  {metrics['specificity']:.1%}")
            print(f"  F1-Score:     {metrics['f1_score']:.1%}")

        except FileNotFoundError:
            print(f"ERROR: Model file not found: {model_path}")
            continue
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Summary across all seeds
    print("\n" + "="*60)
    print("SUMMARY ACROSS ALL SEEDS")
    print("="*60)

    if len(results) == 0:
        print("No successful evaluations!")
        return

    accs = np.array([m['accuracy'] for m in results])
    aucs = np.array([m['auc_roc'] for m in results if not np.isnan(m['auc_roc'])])
    sens = np.array([m['sensitivity'] for m in results])
    spec = np.array([m['specificity'] for m in results])
    f1s = np.array([m['f1_score'] for m in results])

    print(f"\nSuccessful evaluations: {len(results)}/10")
    print(f"\nAccuracy:      {accs.mean():.1%} +/- {accs.std():.1%}")
    if len(aucs) > 0:
        print(f"AUC-ROC:       {aucs.mean():.1%} +/- {aucs.std():.1%}")
    print(f"Sensitivity:   {sens.mean():.1%} +/- {sens.std():.1%}")
    print(f"Specificity:   {spec.mean():.1%} +/- {spec.std():.1%}")
    print(f"F1-Score:      {f1s.mean():.1%} +/- {f1s.std():.1%}")

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
