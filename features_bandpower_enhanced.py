"""
Enhanced bandpower features with AD-specific biomarkers.

Key additions:
1. Band ratios (theta/alpha, theta/beta) - Known AD biomarkers
2. Spectral entropy - Measures signal complexity/randomness
3. Relative band powers - Normalized by total power
4. Channel-pair coherence - Functional connectivity measures
"""

import numpy as np
from scipy.signal import welch, coherence
from scipy.stats import entropy

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45)
}


class EnhancedBandpowerExtractor:
    """
    Enhanced bandpower feature extractor with AD-specific biomarkers.
    Prevents data leakage by fitting normalization only on training data.
    """
    def __init__(self, fs=128, include_coherence=False):
        """
        Args:
            fs: Sampling frequency
            include_coherence: If True, add channel coherence features (slower but more informative)
        """
        self.fs = fs
        self.include_coherence = include_coherence
        self.mean_ = None
        self.std_ = None

    def _extract_raw_features(self, X):
        """
        Extract enhanced features from EEG data.

        Args:
            X: (N, T, C) - N samples, T timepoints, C channels

        Returns:
            features: (N, feature_dim) array
        """
        N, T, C = X.shape
        all_features = []

        for i in range(N):
            features = []

            # Per-channel features
            band_powers = {}  # Store for ratio computation

            for ch in range(C):
                signal = X[i, :, ch]
                freqs, psd = welch(signal, fs=self.fs, nperseg=min(self.fs*2, T))

                # 1. Absolute band powers
                channel_band_powers = []
                for band_name, (lo, hi) in BANDS.items():
                    idx = np.logical_and(freqs >= lo, freqs <= hi)
                    power = psd[idx].mean()
                    channel_band_powers.append(power)

                    # Store for ratio computation
                    if ch not in band_powers:
                        band_powers[ch] = {}
                    band_powers[ch][band_name] = power

                features.extend(channel_band_powers)

                # 2. Relative band powers (normalized by total power)
                total_power = sum(channel_band_powers) + 1e-10
                relative_powers = [p / total_power for p in channel_band_powers]
                features.extend(relative_powers)

                # 3. Spectral entropy (measures signal complexity)
                # Higher entropy = more random/complex
                # AD patients often show reduced entropy
                psd_norm = psd / (psd.sum() + 1e-10)
                spectral_entropy = entropy(psd_norm + 1e-10)
                features.append(spectral_entropy)

            # 4. Band ratios (clinically relevant for AD)
            # Theta/Alpha ratio: Increases in AD (slowing of EEG)
            # Theta/Beta ratio: Also increases in AD
            for ch in range(C):
                theta_power = band_powers[ch]["theta"]
                alpha_power = band_powers[ch]["alpha"]
                beta_power = band_powers[ch]["beta"]

                theta_alpha_ratio = theta_power / (alpha_power + 1e-10)
                theta_beta_ratio = theta_power / (beta_power + 1e-10)
                alpha_beta_ratio = alpha_power / (beta_power + 1e-10)

                features.extend([theta_alpha_ratio, theta_beta_ratio, alpha_beta_ratio])

            # 5. Optional: Channel coherence (functional connectivity)
            # Measures synchronization between channel pairs
            # AD shows altered connectivity patterns
            if self.include_coherence:
                # Compute coherence for key channel pairs
                # Use subset to avoid O(C^2) explosion
                key_pairs = [
                    (0, 1),   # Frontal
                    (9, 10),  # Central
                    (18, 17), # Posterior
                ]

                for ch1, ch2 in key_pairs:
                    if ch1 < C and ch2 < C:
                        freqs_coh, coh = coherence(
                            X[i, :, ch1], X[i, :, ch2],
                            fs=self.fs, nperseg=min(self.fs*2, T)
                        )

                        # Average coherence in each band
                        for lo, hi in BANDS.values():
                            idx = np.logical_and(freqs_coh >= lo, freqs_coh <= hi)
                            if idx.any():
                                features.append(coh[idx].mean())
                            else:
                                features.append(0.0)

            all_features.append(features)

        return np.array(all_features, dtype=np.float32)

    def fit(self, X):
        """
        Fit normalization parameters on training data.

        Args:
            X: (N, T, C) training data
        """
        print(f"  Extracting enhanced features from {len(X)} training samples...")
        feats = self._extract_raw_features(X)
        self.mean_ = feats.mean(axis=0)
        self.std_ = feats.std(axis=0) + 1e-6
        print(f"  Feature dimension: {feats.shape[1]} (enhanced from 95)")
        return self

    def transform(self, X):
        """
        Transform data using fitted normalization parameters.

        Args:
            X: (N, T, C) data to transform

        Returns:
            Normalized features: (N, feature_dim)
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Extractor not fitted. Call fit() first.")

        feats = self._extract_raw_features(X)
        return (feats - self.mean_) / self.std_

    def fit_transform(self, X):
        """
        Fit and transform in one step.

        Args:
            X: (N, T, C) training data

        Returns:
            Normalized features: (N, feature_dim)
        """
        return self.fit(X).transform(X)


def get_feature_names(n_channels=19, include_coherence=False):
    """
    Get feature names for interpretation.

    Returns:
        List of feature names
    """
    names = []

    # Per-channel features
    for ch in range(n_channels):
        # Absolute powers
        for band in BANDS.keys():
            names.append(f"ch{ch}_{band}_abs")

        # Relative powers
        for band in BANDS.keys():
            names.append(f"ch{ch}_{band}_rel")

        # Spectral entropy
        names.append(f"ch{ch}_entropy")

    # Band ratios
    for ch in range(n_channels):
        names.append(f"ch{ch}_theta_alpha_ratio")
        names.append(f"ch{ch}_theta_beta_ratio")
        names.append(f"ch{ch}_alpha_beta_ratio")

    # Coherence (if included)
    if include_coherence:
        key_pairs = [(0, 1), (9, 10), (18, 17)]
        for ch1, ch2 in key_pairs:
            for band in BANDS.keys():
                names.append(f"coherence_ch{ch1}_ch{ch2}_{band}")

    return names
