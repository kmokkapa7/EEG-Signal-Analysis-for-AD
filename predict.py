"""
Standalone inference script — no original datasets required.

Usage
-----
From Python:
    from predict import predict_eeg
    result = predict_eeg(eeg_array)   # eeg_array shape: (T, 19) or (N, T, 19)
    print(result)

From the command line:
    python predict.py --file my_eeg.npy
    python predict.py --file my_eeg.npy --model best_model_seed0.pt  # single model

Input
-----
- NumPy array of shape (T, C) for a single continuous recording, or
  (N, T, C) for pre-segmented windows.
- C must be 19 (EEG channels).
- T should be at least 256 samples (2 s at 128 Hz). Longer recordings are
  automatically segmented into non-overlapping 256-sample windows.
- Sampling rate assumed to be 128 Hz (matches training data).

Output
------
{
  "prediction":   "AD" or "Control",
  "probability":  float  (probability of AD, 0–1),
  "confidence":   "High" / "Medium" / "Low",
  "n_windows":    int,
  "models_used":  int
}
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import welch

# ── Model definitions (copied here so this file is fully self-contained) ──────

class EEGNet(nn.Module):
    def __init__(self, n_channels=19):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(64, 1), padding=(32, 0), bias=False),
            nn.BatchNorm2d(16)
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, n_channels), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((4, 1)),
            nn.Dropout(0.5)
        )
        self.separable = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(16, 1), padding=(8, 0), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((4, 1)),
            nn.Dropout(0.5)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, 1)

    def extract_features(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        x = self.adaptive_pool(x)
        return x.view(x.size(0), -1)

    def forward(self, x):
        return self.classifier(self.extract_features(x)).squeeze(1)


class EEGFusionNet(nn.Module):
    def __init__(self, n_channels=19, band_dim=95):
        super().__init__()
        self.eegnet = EEGNet(n_channels=n_channels)
        self.band_mlp = nn.Sequential(
            nn.Linear(band_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 + 64, 64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x_time, x_band):
        z_time = self.eegnet.extract_features(x_time)
        z_band = self.band_mlp(x_band)
        return self.classifier(torch.cat([z_time, z_band], dim=1)).squeeze(1)


# ── Preprocessing ──────────────────────────────────────────────────────────────

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
FS = 128          # sampling rate (Hz)
WINDOW = 256      # samples per window (2 s)


def _segment(X, window=WINDOW):
    """(T, C) → (N, window, C)  — drops the last incomplete window."""
    T, C = X.shape
    n = T // window
    return X[:n * window].reshape(n, window, C).astype(np.float32)


def _zscore(X):
    """Per-recording z-score: (N, T, C) → (N, T, C)."""
    mean = X.mean(axis=(0, 1), keepdims=True)
    std  = X.std(axis=(0, 1),  keepdims=True) + 1e-6
    return (X - mean) / std


def _bandpower(X, fs=FS):
    """(N, T, C) → (N, C*5) bandpower features, self-normalised."""
    N, T, C = X.shape
    feats = []
    for i in range(N):
        row = []
        for ch in range(C):
            freqs, psd = welch(X[i, :, ch], fs=fs, nperseg=fs * 2)
            for lo, hi in BANDS.values():
                row.append(psd[(freqs >= lo) & (freqs <= hi)].mean())
        feats.append(row)
    feats = np.array(feats, dtype=np.float32)
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-6)
    return feats


# ── Inference ─────────────────────────────────────────────────────────────────

DEFAULT_MODELS = [f"best_model_seed{i}.pt" for i in range(10)]


def predict_eeg(eeg, model_paths=None, device=None):
    """
    Parameters
    ----------
    eeg : np.ndarray
        Shape (T, 19) for a continuous recording, or (N, T, 19) for
        pre-segmented windows.
    model_paths : list[str] | None
        Paths to .pt weight files.  Defaults to best_model_seed0..9.pt in
        the current directory.  Missing files are skipped silently.
    device : str | None
        'cpu', 'cuda', or None (auto-detect).

    Returns
    -------
    dict with keys: prediction, probability, confidence, n_windows, models_used
    """
    if model_paths is None:
        model_paths = DEFAULT_MODELS

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    eeg = np.array(eeg, dtype=np.float32)

    # Accept (T, C) or (N, T, C)
    if eeg.ndim == 2:
        eeg = _segment(eeg)
    elif eeg.ndim != 3:
        raise ValueError(f"Expected 2-D or 3-D array, got shape {eeg.shape}")

    N, T, C = eeg.shape
    if C != 19:
        raise ValueError(f"Expected 19 channels, got {C}. "
                         "Transpose your array if channels are on axis 0.")
    if N == 0:
        raise ValueError("Recording too short — need at least 256 samples.")

    # Preprocess
    X_time = _zscore(eeg)                          # (N, T, C)
    X_band = _bandpower(X_time)                    # (N, 95)

    # Tensors
    t_time = torch.tensor(X_time, dtype=torch.float32).unsqueeze(1).to(device)
    t_band = torch.tensor(X_band, dtype=torch.float32).to(device)

    # Ensemble over all available model files
    all_probs = []
    models_used = 0

    for path in model_paths:
        if not os.path.isfile(path):
            continue
        try:
            model = EEGFusionNet(n_channels=19).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            with torch.no_grad():
                logits = model(t_time, t_band)
                probs  = torch.sigmoid(logits).cpu().numpy()   # (N,)
            all_probs.append(probs.mean())   # aggregate windows → scalar
            models_used += 1
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")

    if models_used == 0:
        raise RuntimeError("No model files found. Make sure the .pt files are "
                           "in the same directory as this script.")

    final_prob = float(np.mean(all_probs))
    prediction = "AD" if final_prob >= 0.5 else "Control"

    # Confidence: distance from the 0.5 decision boundary
    margin = abs(final_prob - 0.5)
    if margin >= 0.3:
        confidence = "High"
    elif margin >= 0.15:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "prediction":  prediction,
        "probability": round(final_prob, 4),
        "confidence":  confidence,
        "n_windows":   N,
        "models_used": models_used,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AD prediction on an EEG recording.")
    parser.add_argument("--file",  required=True,
                        help="Path to a .npy file with shape (T, 19) or (N, T, 19)")
    parser.add_argument("--model", default=None, nargs="+",
                        help="One or more .pt model files (default: all 10 seeds)")
    parser.add_argument("--device", default=None,
                        help="'cpu' or 'cuda' (default: auto)")
    args = parser.parse_args()

    eeg = np.load(args.file)
    print(f"Loaded EEG array with shape {eeg.shape}")

    result = predict_eeg(eeg, model_paths=args.model, device=args.device)

    print("\n── Result ──────────────────────────────")
    print(f"  Prediction : {result['prediction']}")
    print(f"  Probability: {result['probability']:.1%} AD")
    print(f"  Confidence : {result['confidence']}")
    print(f"  Windows    : {result['n_windows']}")
    print(f"  Models used: {result['models_used']}/10")
    print("────────────────────────────────────────")
