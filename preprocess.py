import numpy as np

def subject_channel_zscore(X, subjects):
    Xn = np.zeros_like(X)

    for subj in np.unique(subjects):
        idx = np.where(subjects == subj)[0]
        subj_data = X[idx]

        mean = subj_data.mean(axis=(0, 1), keepdims=True)
        std = subj_data.std(axis=(0, 1), keepdims=True) + 1e-6

        Xn[idx] = (subj_data - mean) / std

    return Xn
