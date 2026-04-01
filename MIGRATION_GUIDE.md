# Migration Guide: Updating Existing Code

If you have other scripts that use the old API, here's how to update them.

## Quick Reference

### 1. Bandpower Feature Extraction

**OLD (causes data leakage):**
```python
from features_bandpower import bandpower_features

X_band = bandpower_features(X, fs=128)
# Split after extraction - WRONG!
Xtr_b = X_band[train_idx]
Xte_b = X_band[test_idx]
```

**NEW (no leakage):**
```python
from features_bandpower import BandpowerExtractor

# Create extractor
bp_extractor = BandpowerExtractor(fs=128)

# Fit on training data ONLY
Xtr_b = bp_extractor.fit_transform(Xtr_t)

# Transform test data using training statistics
Xte_b = bp_extractor.transform(Xte_t)
```

---

### 2. Data Splitting

**OLD (no seed, 2-way split):**
```python
from subject_split import subject_stratified_split

Xtr, Xte, ytr, yte, subj_tr, subj_te = subject_stratified_split(
    X, y, subjects
)
```

**NEW (with seed, 3-way split):**
```python
from subject_split import subject_stratified_split

# Three-way split with seed
Xtr, Xval, Xte, ytr, yval, yte, str, sval, ste = subject_stratified_split(
    X, y, subjects,
    test_size=0.2,
    val_size=0.2,
    random_state=42  # For reproducibility
)

# OR two-way split (backward compatible)
Xtr, Xte, ytr, yte, str, ste = subject_stratified_split(
    X, y, subjects,
    test_size=0.2,
    random_state=42
)
```

---

### 3. Model Training

**OLD (no save path returned):**
```python
from train import train_model

train_model(model, train_ds, val_ds, seed=42)
```

**NEW (returns best model path):**
```python
from train import train_model

best_model_path = train_model(
    model,
    train_ds,
    val_ds,
    seed=42,
    save_path="my_model.pt"  # Optional, defaults to "best_model.pt"
)
```

---

### 4. Model Evaluation

**OLD (uses current model state, returns only accuracy):**
```python
from evaluate import evaluate_subject_level

acc = evaluate_subject_level(model, Xte_t, Xte_b, yte, subj_te)
print(f"Accuracy: {acc:.4f}")
```

**NEW (loads best model, returns comprehensive metrics):**
```python
from evaluate import evaluate_subject_level

metrics = evaluate_subject_level(
    model,
    Xte_t, Xte_b, yte, subj_te,
    model_path="best_model.pt"  # Loads best checkpoint
)

# Access all metrics
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
print(f"Sensitivity: {metrics['sensitivity']:.4f}")
print(f"Specificity: {metrics['specificity']:.4f}")
```

---

## Complete Example: Full Pipeline

**OLD CODE:**
```python
# Load data
X, y, subjects = load_openneuro_dataset("dataset")
X = subject_channel_zscore(X, subjects)

# Extract bandpower (LEAKAGE!)
X_band = bandpower_features(X, fs=128)

# Split (no seed)
Xtr_t, Xte_t, ytr, yte, str, ste = subject_stratified_split(X, y, subjects)
Xtr_b = X_band[np.isin(subjects, str)]
Xte_b = X_band[np.isin(subjects, ste)]

# Train
model = EEGFusionNet(n_channels=19)
train_model(model, train_ds, val_ds, seed=42)

# Evaluate (no best model, only accuracy)
acc = evaluate_subject_level(model, Xte_t, Xte_b, yte, ste)
```

**NEW CODE:**
```python
# Load data
X, y, subjects = load_openneuro_dataset("dataset")
X = subject_channel_zscore(X, subjects)

# Split FIRST (with seed and validation set)
Xtr_t, Xval_t, Xte_t, ytr, yval, yte, str, sval, ste = subject_stratified_split(
    X, y, subjects,
    test_size=0.2,
    val_size=0.2,
    random_state=42
)

# Extract bandpower (NO LEAKAGE)
bp_extractor = BandpowerExtractor(fs=128)
Xtr_b = bp_extractor.fit_transform(Xtr_t)      # Fit on train
Xval_b = bp_extractor.transform(Xval_t)        # Transform val
Xte_b = bp_extractor.transform(Xte_t)          # Transform test

# Create datasets
train_ds = EEGMultiViewDataset(Xtr_t, Xtr_b, ytr)
val_ds = EEGMultiViewDataset(Xval_t, Xval_b, yval)

# Train
model = EEGFusionNet(n_channels=19)
best_path = train_model(
    model, train_ds, val_ds,
    seed=42,
    save_path="best_model_seed42.pt"
)

# Evaluate (loads best model, comprehensive metrics)
metrics = evaluate_subject_level(
    model, Xte_t, Xte_b, yte, ste,
    model_path=best_path
)

# Access all metrics
print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
print(f"Sensitivity: {metrics['sensitivity']:.4f}")
print(f"Specificity: {metrics['specificity']:.4f}")
```

---

## Checklist for Updating Your Code

- [ ] Replace `bandpower_features()` with `BandpowerExtractor`
- [ ] Extract features AFTER splitting (not before)
- [ ] Fit extractor only on training data
- [ ] Add `random_state` parameter to splits
- [ ] Use 3-way split (train/val/test)
- [ ] Capture return value from `train_model()`
- [ ] Pass `model_path` to `evaluate_subject_level()`
- [ ] Use returned metrics dictionary instead of just accuracy
- [ ] Add error handling around file operations
- [ ] Use unique save paths for different experiments

---

## Testing Your Updated Code

Run this quick test to verify the changes:

```python
# Test 1: BandpowerExtractor
bp = BandpowerExtractor(fs=128)
Xtr_b = bp.fit_transform(Xtr)
Xte_b = bp.transform(Xte)
assert bp.mean_ is not None, "Extractor not fitted!"
print("✓ BandpowerExtractor working")

# Test 2: Seeded splits produce different results
split1 = subject_stratified_split(X, y, subjects, random_state=1)
split2 = subject_stratified_split(X, y, subjects, random_state=2)
assert not np.array_equal(split1[0], split2[0]), "Seeds not working!"
print("✓ Random seeds working")

# Test 3: Evaluation returns dict
metrics = evaluate_subject_level(model, Xte_t, Xte_b, yte, ste)
assert isinstance(metrics, dict), "Should return dict!"
assert 'auc_roc' in metrics, "Missing AUC-ROC!"
print("✓ Evaluation returns comprehensive metrics")
```

---

## Common Errors After Migration

### Error: "Extractor not fitted"
**Cause**: Calling `transform()` before `fit()`
**Fix**: Use `fit_transform()` on training data first

### Error: Wrong number of return values from split
**Cause**: Old code expects 6 values, new code returns 9 (if using val_size)
**Fix**: Either use 2-way split or update unpacking

### Error: evaluate_subject_level() got unexpected keyword argument 'model_path'
**Cause**: Using old version of evaluate.py
**Fix**: Update evaluate.py to the new version

---

## Need Help?

If you encounter issues during migration:
1. Check that you've updated all affected files
2. Verify you're using the new function signatures
3. Test each component individually
4. Refer to `main.py` for a complete working example
