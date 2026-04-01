from sklearn.model_selection import GroupShuffleSplit
import numpy as np

def subject_stratified_split(X, y, subjects, test_size=0.2, random_state=None, val_size=None):
    """
    Split data by subjects with optional stratification and validation set.

    Args:
        X: Feature array (N, ...)
        y: Labels (N,)
        subjects: Subject IDs (N,)
        test_size: Proportion of subjects for test set (default: 0.2)
        random_state: Random seed for reproducibility
        val_size: If provided, creates train/val/test split. Should be proportion of original data.

    Returns:
        If val_size is None: (X_train, X_test, y_train, y_test, subj_train, subj_test)
        If val_size provided: (X_train, X_val, X_test, y_train, y_val, y_test, subj_train, subj_val, subj_test)
    """
    # Get unique subjects and their labels
    unique_subjects = np.unique(subjects)
    subject_labels = np.array([y[subjects == subj][0] for subj in unique_subjects])

    # First split: separate test set
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # Create dummy data for split (we only care about groups)
    dummy_X = np.zeros(len(subjects))
    train_val_idx, test_idx = next(gss_test.split(dummy_X, y, groups=subjects))

    if val_size is None:
        # Two-way split: train/test
        return (
            X[train_val_idx], X[test_idx],
            y[train_val_idx], y[test_idx],
            subjects[train_val_idx], subjects[test_idx]
        )
    else:
        # Three-way split: train/val/test
        # Calculate validation proportion relative to remaining data
        val_size_adjusted = val_size / (1 - test_size)

        X_train_val, y_train_val, subj_train_val = X[train_val_idx], y[train_val_idx], subjects[train_val_idx]

        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_state)
        dummy_train_val = np.zeros(len(subj_train_val))
        train_idx, val_idx = next(gss_val.split(dummy_train_val, y_train_val, groups=subj_train_val))

        return (
            X_train_val[train_idx], X_train_val[val_idx], X[test_idx],
            y_train_val[train_idx], y_train_val[val_idx], y[test_idx],
            subj_train_val[train_idx], subj_train_val[val_idx], subjects[test_idx]
        )
