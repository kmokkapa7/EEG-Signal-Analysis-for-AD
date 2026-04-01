"""
Enhanced training script with improved features and data augmentation.

Improvements over main.py:
1. Enhanced bandpower features (theta/alpha ratio, spectral entropy, etc.)
2. Data augmentation (amplitude scaling, noise injection)
3. Deeper MLP for frequency features
"""

from load_openneuro import load_openneuro_dataset
from preprocess import subject_channel_zscore
from subject_split import subject_stratified_split
from dataset import EEGMultiViewDataset
from features_bandpower_enhanced import EnhancedBandpowerExtractor
from model_fusion_enhanced import EEGFusionNetEnhanced
from train import train_model
from evaluate import evaluate_subject_level
from augmentation import augment_dataset

import numpy as np
import torch
import os


def main():
    print("="*70)
    print("ENHANCED EEG ALZHEIMER'S DISEASE CLASSIFICATION")
    print("="*70)
    print("\nImprovements:")
    print("  - Enhanced bandpower features (theta/alpha ratio, spectral entropy)")
    print("  - Data augmentation (amplitude scaling, Gaussian noise)")
    print("  - Deeper frequency MLP architecture")

    # -------- LOAD EEG DATA --------
    print("\n[1/6] Loading EEG dataset...")
    try:
        X, y, subjects = load_openneuro_dataset(
            root_dir="dataset",
            window_sec=8.0
        )
        print(f"  [OK] Loaded {len(X)} windows from {len(np.unique(subjects))} subjects")
        print(f"  [OK] Shape: {X.shape} (samples, time, channels)")
        print(f"  [OK] Class distribution: {(y==0).sum()} Control, {(y==1).sum()} AD")
    except Exception as e:
        print(f"  [ERROR] Error loading dataset: {e}")
        return

    # -------- NORMALIZE TIME DOMAIN --------
    print("\n[2/6] Normalizing time-domain signals...")
    X = subject_channel_zscore(X, subjects)
    print("  [OK] Applied per-subject, per-channel z-score normalization")

    # -------- RUN CROSS-VALIDATION --------
    print("\n[3/6] Starting enhanced cross-validation...")
    seeds = range(10)  # Full 10-seed cross-validation
    all_metrics = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}/{len(seeds)-1}")
        print(f"{'='*70}")

        # -------- SPLIT DATA --------
        print("\n  [A] Splitting data into train/val/test sets...")
        try:
            Xtr_t, Xval_t, Xte_t, ytr, yval, yte, subj_tr, subj_val, subj_te = subject_stratified_split(
                X, y, subjects,
                test_size=0.2,
                val_size=0.2,
                random_state=seed
            )
            print(f"    Train: {len(np.unique(subj_tr))} subjects, {len(ytr)} windows")
            print(f"    Val:   {len(np.unique(subj_val))} subjects, {len(yval)} windows")
            print(f"    Test:  {len(np.unique(subj_te))} subjects, {len(yte)} windows")
        except Exception as e:
            print(f"    [ERROR] Error during split: {e}")
            continue

        # -------- DATA AUGMENTATION --------
        print("\n  [B] Applying data augmentation to training set...")
        try:
            Xtr_t_aug, ytr_aug, subj_tr_aug = augment_dataset(
                Xtr_t, ytr, subj_tr,
                augmentation_factor=1,  # 1x augmentation (doubles data)
                seed=seed
            )
            print(f"    [OK] Training set size: {len(ytr)} -> {len(ytr_aug)}")
        except Exception as e:
            print(f"    [ERROR] Error during augmentation: {e}")
            continue

        # -------- EXTRACT ENHANCED BANDPOWER --------
        print("\n  [C] Extracting enhanced bandpower features...")
        try:
            bp_extractor = EnhancedBandpowerExtractor(
                fs=128,
                include_coherence=False  # Set True for even more features (slower)
            )

            # Fit ONLY on original training data (not augmented)
            bp_extractor.fit(Xtr_t)

            # Transform all sets
            Xtr_b_aug = bp_extractor.transform(Xtr_t_aug)
            Xval_b = bp_extractor.transform(Xval_t)
            Xte_b = bp_extractor.transform(Xte_t)

            print(f"    [OK] Feature dimension: {Xtr_b_aug.shape[1]}")
            print("    [OK] Normalization fit on original training set only")
        except Exception as e:
            print(f"    [ERROR] Error extracting features: {e}")
            continue

        # -------- CREATE DATASETS --------
        train_ds = EEGMultiViewDataset(Xtr_t_aug, Xtr_b_aug, ytr_aug)
        val_ds   = EEGMultiViewDataset(Xval_t, Xval_b, yval)

        # -------- INITIALIZE ENHANCED MODEL --------
        print("\n  [D] Initializing enhanced model...")
        model = EEGFusionNetEnhanced(
            n_channels=19,
            band_dim=Xtr_b_aug.shape[1]  # Auto-detect from features
        )

        # Load pretrained weights if available
        pretrained_path = "eegnet_pretrained.pt"
        if os.path.exists(pretrained_path):
            try:
                state_dict = torch.load(pretrained_path, map_location="cpu")
                model.eegnet.load_state_dict(state_dict, strict=False)
                print(f"    [OK] Loaded pretrained EEGNet from {pretrained_path}")
            except Exception as e:
                print(f"    [WARNING] Could not load pretrained weights: {e}")
                print("    -> Training from scratch")
        else:
            print(f"    [WARNING] Pretrained model not found at {pretrained_path}")
            print("    -> Training from scratch")

        # -------- TRAIN MODEL --------
        print("\n  [E] Training enhanced model...")
        model_path = f"best_model_enhanced_seed{seed}.pt"
        try:
            best_model_path = train_model(
                model,
                train_ds,
                val_ds,
                epochs=50,
                batch_size=64,
                lr=1e-4,
                seed=seed,
                save_path=model_path
            )
            print(f"    [OK] Training completed. Best model saved to {best_model_path}")
        except Exception as e:
            print(f"    [ERROR] Error during training: {e}")
            continue

        # -------- EVALUATE ON TEST SET --------
        print("\n  [F] Evaluating on held-out test set...")
        try:
            metrics = evaluate_subject_level(
                model,
                Xte_t, Xte_b, yte, subj_te,
                model_path=best_model_path
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"    [ERROR] Error during evaluation: {e}")
            continue

    # -------- AGGREGATE RESULTS --------
    print("\n" + "="*70)
    print("FINAL ENHANCED CROSS-VALIDATION RESULTS")
    print("="*70)

    if len(all_metrics) == 0:
        print("No successful runs completed.")
        return

    # Extract metrics
    accs = np.array([m['accuracy'] for m in all_metrics])
    aucs = np.array([m['auc_roc'] for m in all_metrics if not np.isnan(m['auc_roc'])])
    sens = np.array([m['sensitivity'] for m in all_metrics])
    spec = np.array([m['specificity'] for m in all_metrics])
    f1s = np.array([m['f1_score'] for m in all_metrics])

    print(f"\nResults across {len(all_metrics)} runs:")
    print(f"\nAccuracy:      {accs.mean():.4f} +/- {accs.std():.4f}")
    if len(aucs) > 0:
        print(f"AUC-ROC:       {aucs.mean():.4f} +/- {aucs.std():.4f}")
    print(f"Sensitivity:   {sens.mean():.4f} +/- {sens.std():.4f}")
    print(f"Specificity:   {spec.mean():.4f} +/- {spec.std():.4f}")
    print(f"F1-Score:      {f1s.mean():.4f} +/- {f1s.std():.4f}")

    print("\nPer-seed results:")
    print(f"{'Seed':<6} {'Acc':<8} {'AUC':<8} {'Sens':<8} {'Spec':<8} {'F1':<8}")
    print("-" * 50)
    for i, m in enumerate(all_metrics):
        auc_str = f"{m['auc_roc']:.4f}" if not np.isnan(m['auc_roc']) else "N/A"
        print(f"{i:<6} {m['accuracy']:.4f}   {auc_str:<8} {m['sensitivity']:.4f}   {m['specificity']:.4f}   {m['f1_score']:.4f}")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
