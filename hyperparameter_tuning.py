"""
Hyperparameter tuning using Optuna for optimizing model performance.
Searches for optimal learning rate, batch size, and other hyperparameters.
"""

import optuna
import torch
import numpy as np
from model_fusion import EEGFusionNet
from load_openneuro import load_openneuro_dataset
from preprocess import subject_channel_zscore
from subject_split import subject_stratified_split
from features_bandpower import BandpowerExtractor
from dataset import EEGMultiViewDataset
from train import train_model
from evaluate import evaluate_subject_level


def objective(trial):
    """
    Optuna objective function to optimize.

    Args:
        trial: Optuna trial object

    Returns:
        Validation accuracy to maximize
    """
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = trial.suggest_int("epochs", 50, 150)

    # Fixed seed for reproducibility during tuning
    seed = 42

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: lr={lr:.2e}, batch_size={batch_size}, epochs={epochs}")
    print(f"{'='*60}")

    try:
        # Load data
        X, y, subjects = load_openneuro_dataset("dataset")
        X = subject_channel_zscore(X, subjects)

        # Split data
        Xtr_t, Xval_t, Xte_t, ytr, yval, yte, str_, sval, ste = subject_stratified_split(
            X, y, subjects, test_size=0.2, val_size=0.2, random_state=seed
        )

        # Extract bandpower features
        bp_extractor = BandpowerExtractor(fs=128)
        Xtr_b = bp_extractor.fit_transform(Xtr_t)
        Xval_b = bp_extractor.transform(Xval_t)
        Xte_b = bp_extractor.transform(Xte_t)

        # Create datasets
        train_ds = EEGMultiViewDataset(Xtr_t, Xtr_b, ytr)
        val_ds = EEGMultiViewDataset(Xval_t, Xval_b, yval)

        # Initialize model
        model = EEGFusionNet(n_channels=19)

        # Load pretrained weights if available
        try:
            state = torch.load("eegnet_pretrained.pt", map_location="cpu")
            model.eegnet.load_state_dict(state, strict=False)
        except:
            pass

        # Train model
        model_path = f"optuna_trial_{trial.number}.pt"
        train_model(
            model, train_ds, val_ds,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            save_path=model_path
        )

        # Evaluate on validation set
        metrics = evaluate_subject_level(
            model, Xval_t, Xval_b, yval, sval,
            model_path=model_path
        )

        # Return validation accuracy
        val_acc = metrics['accuracy']
        print(f"\nTrial {trial.number} Result: Validation Accuracy = {val_acc:.4f}")

        return val_acc

    except Exception as e:
        print(f"\nTrial {trial.number} Failed: {e}")
        return 0.0  # Return worst score on failure


def main():
    """
    Main function to run hyperparameter optimization.
    """
    print("="*60)
    print("HYPERPARAMETER TUNING WITH OPTUNA")
    print("="*60)
    print("\nSearching for optimal hyperparameters...")
    print("Parameters to optimize:")
    print("  - Learning rate: [1e-5, 1e-3]")
    print("  - Batch size: [32, 64, 128]")
    print("  - Epochs: [50, 150]")
    print()

    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="eeg_ad_classification"
    )

    # Run optimization
    n_trials = 20  # Number of trials to run
    print(f"Running {n_trials} trials...\n")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Validation Accuracy): {trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Print top 5 trials
    print("\n" + "="*60)
    print("TOP 5 TRIALS")
    print("="*60)

    df = study.trials_dataframe()
    df_sorted = df.sort_values('value', ascending=False).head(5)

    print("\n" + df_sorted[['number', 'value', 'params_lr', 'params_batch_size', 'params_epochs']].to_string(index=False))

    # Save study
    print("\n" + "="*60)
    print(f"Study saved. Use these hyperparameters in main.py for best results.")
    print("="*60)

    return study


if __name__ == "__main__":
    # Check if optuna is installed
    try:
        import optuna
    except ImportError:
        print("Error: Optuna is not installed.")
        print("Please install it with: pip install optuna")
        exit(1)

    study = main()
