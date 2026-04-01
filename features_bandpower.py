import numpy as np
from scipy.signal import welch

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45)
}

class BandpowerExtractor:
    """
    Bandpower feature extractor with proper train/test normalization.
    Prevents data leakage by fitting normalization parameters only on training data.
    """
    def __init__(self, fs=128):
        self.fs = fs
        self.mean_ = None
        self.std_ = None

    def _extract_raw_features(self, X):
        """
        Extract raw bandpower features without normalization.
        X: (N, T, C)
        returns: (N, C * 5)
        """
        N, T, C = X.shape
        feats = []

        for i in range(N):
            fvec = []
            for ch in range(C):
                freqs, psd = welch(X[i, :, ch], fs=self.fs, nperseg=self.fs*2)
                for lo, hi in BANDS.values():
                    idx = np.logical_and(freqs >= lo, freqs <= hi)
                    fvec.append(psd[idx].mean())
            feats.append(fvec)

        return np.array(feats, dtype=np.float32)

    def fit(self, X):
        """
        Fit normalization parameters on training data.
        X: (N, T, C)
        """
        feats = self._extract_raw_features(X)
        self.mean_ = feats.mean(axis=0)
        self.std_ = feats.std(axis=0) + 1e-6
        return self

    def transform(self, X):
        """
        Transform data using fitted normalization parameters.
        X: (N, T, C)
        returns: (N, C * 5)
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Extractor not fitted. Call fit() first.")

        feats = self._extract_raw_features(X)
        return (feats - self.mean_) / self.std_

    def fit_transform(self, X):
        """
        Fit and transform in one step.
        X: (N, T, C)
        returns: (N, C * 5)
        """
        return self.fit(X).transform(X)


def bandpower_features(X, fs=128):
    """
    DEPRECATED: Use BandpowerExtractor for proper train/test handling.
    This function performs global normalization and causes data leakage.

    X: (N, T, C)
    returns: (N, C * 5)
    """
    N, T, C = X.shape
    feats = []

    for i in range(N):
        fvec = []
        for ch in range(C):
            freqs, psd = welch(X[i, :, ch], fs=fs, nperseg=fs*2)
            for lo, hi in BANDS.values():
                idx = np.logical_and(freqs >= lo, freqs <= hi)
                fvec.append(psd[idx].mean())
        feats.append(fvec)

    feats = np.array(feats, dtype=np.float32)
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-6)
    return feats
