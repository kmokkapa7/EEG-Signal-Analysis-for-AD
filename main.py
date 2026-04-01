from load_openneuro import load_openneuro_dataset
from preprocess import subject_channel_zscore
from subject_split import subject_stratified_split
from dataset import EEGMultiViewDataset
from features_bandpower import BandpowerExtractor
from model_fusion import EEGFusionNet
from train import train_model
from evaluate import evaluate_subject_level

import numpy as np
import torch
import os

def main():
    print("="*70)
    print("EEG ALZHEIMER'S DISEASE CLASSIFICATION")
    print("="*70)

    # -------- LOAD EEG DATA --------
    print("\n[1/5] Loading EEG dataset...")
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
    print("\n[2/5] Normalizing time-domain signals...")
    X = subject_channel_zscore(X, subjects)
    print("  [OK] Applied per-subject, per-channel z-score normalization")

    # -------- RUN CROSS-VALIDATION --------
    print("\n[3/5] Starting cross-validation...")
    seeds = range(10)
    all_metrics = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}/9")
        print(f"{'='*70}")

        # -------- SPLIT DATA (with seed for reproducibility) --------
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

        # -------- EXTRACT BANDPOWER (FIX DATA LEAKAGE) --------
        print("\n  [B] Extracting bandpower features (without data leakage)...")
        try:
            bp_extractor = BandpowerExtractor(fs=128)

            # Fit ONLY on training data
            Xtr_b = bp_extractor.fit_transform(Xtr_t)

            # Transform validation and test using training statistics
            Xval_b = bp_extractor.transform(Xval_t)
            Xte_b = bp_extractor.transform(Xte_t)

            print(f"    [OK] Extracted {Xtr_b.shape[1]} bandpower features per window")
            print("    [OK] Normalization fit on training set only (no leakage)")
        except Exception as e:
            print(f"    [ERROR] Error extracting features: {e}")
            continue

        # -------- CREATE DATASETS --------
        train_ds = EEGMultiViewDataset(Xtr_t, Xtr_b, ytr)
        val_ds   = EEGMultiViewDataset(Xval_t, Xval_b, yval)

        # -------- INITIALIZE MODEL --------
        print("\n  [C] Initializing model...")
        model = EEGFusionNet(n_channels=19)

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
        print("\n  [D] Training model...")
        model_path = f"best_model_seed{seed}.pt"
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
        print("\n  [E] Evaluating on held-out test set...")
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
    print("FINAL CROSS-VALIDATION RESULTS")
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
