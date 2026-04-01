# EEG Signal Analysis for Alzheimer's Disease Detection

A deep learning pipeline for binary classification of Alzheimer's Disease (AD) vs. healthy controls using resting-state EEG signals. Achieves **94.9% accuracy** on a 101,916-sample integrated dataset using a multi-view fusion network.

---

## Results

| Model | Dataset | Accuracy | Architecture |
|---|---|---|---|
| Integrated Fusion | 101,916 windows | **94.9%** | EEGNet + Bandpower MLP |
| OpenNeuro Baseline | ~5,000 windows | 85.4% ± 8.7% | EEGNet + Bandpower MLP |
| Iraq Dataset | Iraq EEG | See `best_iraq_model.pt` | Custom CNN |

Validated with **subject-stratified 10-seed cross-validation** (no data leakage between subjects).

---

## Architecture

The model (`EEGFusionNet`) fuses two parallel pathways:

```
EEG Window (19 channels, 128 Hz)
        |
        ├─── EEGNet (temporal CNN) ───► 32-dim
        |
        └─── Bandpower MLP ───────────► 64-dim
                                              |
                                        Concatenate (96-dim)
                                              |
                                        Classifier → AD / Control
```

- **Temporal path (EEGNet):** Three convolutional layers learning temporal and spatial patterns from raw EEG
- **Frequency path (MLP):** Processes 95 bandpower features (5 bands × 19 channels: δ, θ, α, β, γ)
- **Total parameters:** ~17,200 — lightweight enough for edge deployment

See [ARCHITECTURE_SUMMARY_94.9_PERCENT.md](ARCHITECTURE_SUMMARY_94.9_PERCENT.md) for full details.

---

## Quick Start — Inference (No Dataset Required)

The trained model weights are included. You can run predictions on any new EEG data immediately.

### Install dependencies

```bash
pip install torch numpy scipy
```

### Run from Python

```python
from predict import predict_eeg
import numpy as np

eeg = np.load("my_eeg.npy")   # shape: (T, 19) or (N, T, 19)
result = predict_eeg(eeg)
print(result)
# {'prediction': 'AD', 'probability': 0.823, 'confidence': 'High', 'n_windows': 12, 'models_used': 10}
```

### Run from command line

```bash
python predict.py --file my_eeg.npy
```

**Input format:**
- NumPy array of shape `(T, 19)` — continuous recording, or `(N, T, 19)` — pre-segmented windows
- 19 EEG channels in the 10-20 international system
- Sampling rate: 128 Hz (recordings are automatically segmented into 2-second windows)

The script runs an ensemble over all 10 trained seeds and returns a majority probability.

---

## Project Structure

```
├── predict.py                     # Standalone inference — no dataset needed
│
├── Model weights
│   ├── best_model_seed{0-9}.pt    # 10-seed ensemble (main model)
│   ├── best_model_enhanced_seed{0-9}.pt
│   ├── best_iraq_model.pt         # Iraq dataset model
│   ├── base_model_seed{0-2}.pt
│   └── finetuned_model_seed{0-2}.pt
│
├── Architecture
│   ├── model_fusion.py            # EEGFusionNet (main model)
│   ├── model_eegnet.py            # EEGNet backbone
│   ├── model_iraq.py              # Iraq-specific model
│   ├── model_transfer.py          # Transfer learning model
│   └── model_fusion_enhanced.py
│
├── Training
│   ├── main.py                    # Main training script
│   ├── main_enhanced.py           # Enhanced training
│   ├── main_iraq.py               # Iraq dataset training
│   ├── main_transfer.py           # Transfer learning
│   ├── train.py / train_kaggle.py / train_pretrain.py
│   └── hyperparameter_tuning.py
│
├── Data & Features
│   ├── dataset.py                 # OpenNeuro dataset loader
│   ├── integrated_dataset.py      # Integrated dataset loader
│   ├── iraq_dataset.py            # Iraq EEG loader
│   ├── preprocess.py              # Z-score normalisation
│   ├── features_bandpower.py      # Bandpower extraction
│   ├── augmentation.py            # Training augmentation
│   └── subject_split.py           # Subject-stratified splits
│
├── Evaluation
│   ├── evaluate.py                # Subject-level metrics
│   └── quick_eval_all_seeds.py    # Evaluate all 10 seeds
│
└── Docs
    ├── ARCHITECTURE_SUMMARY_94.9_PERCENT.md
    ├── EXCLUDED_FILES.md          # Large files not in this repo
    ├── AUDIT_REPORT.md
    └── IMPROVEMENTS_SUMMARY.md
```

---

## Training From Scratch

You will need to download the datasets first (see [EXCLUDED_FILES.md](EXCLUDED_FILES.md)).

```bash
# Train main model (10 seeds)
python main.py

# Train on Iraq dataset
python main_iraq.py

# Evaluate all trained seeds
python quick_eval_all_seeds.py
```

### Datasets

| Dataset | Source |
|---|---|
| OpenNeuro ds004504 | [openneuro.org](https://openneuro.org) — search accession `ds004504` |
| Iraq EEG Dataset | Mendeley Data — Ieracitano et al. |

After downloading, place them in `dataset/` and `EEG_AD_Iraq/` respectively. The preprocessed files (`integrated_eeg_dataset.npz`, `X_raw_preprocessed.npy`) can be regenerated by running `preprocess.py`.

---

## EEG Biomarkers

The model captures known neurophysiological signatures of AD:

| Band | Frequency | AD Signature |
|---|---|---|
| Delta | 0.5–4 Hz | Increased power |
| Theta | 4–8 Hz | Increased power |
| Alpha | 8–13 Hz | Decreased power |
| Beta | 13–30 Hz | Decreased power |
| Gamma | 30–45 Hz | Decreased power |

The theta/alpha ratio — a well-established AD biomarker — is implicitly captured by the bandpower feature pathway.

---

## References

- Lawhern et al. (2018). *EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.* Journal of Neural Engineering.
- Ieracitano et al. (2023). *Multi-Modal Data of Alzheimer's Disease, Frontotemporal Dementia and Healthy Controls.* Data, 8(6):95.
- Babiloni et al. (2020). *What electrophysiology tells us about Alzheimer's disease.* Neurobiology of Aging.
