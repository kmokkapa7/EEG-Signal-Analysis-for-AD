"""
Data augmentation for EEG signals.

Techniques:
1. Time jittering - Add small time shifts
2. Amplitude scaling - Scale signal amplitude
3. Gaussian noise - Add small random noise
4. Time warping - Stretch/compress time axis slightly

All augmentations preserve the general characteristics of EEG signals
while increasing training data diversity.
"""

import numpy as np


class EEGAugmenter:
    """
    Data augmentation for EEG time series.
    """

    def __init__(self, seed=None):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)

    def time_jitter(self, X, max_shift=10):
        """
        Apply small random time shifts to each sample.

        Args:
            X: (N, T, C) EEG data
            max_shift: Maximum shift in samples

        Returns:
            Augmented data: (N, T, C)
        """
        N, T, C = X.shape
        X_aug = np.zeros_like(X)

        for i in range(N):
            shift = self.rng.randint(-max_shift, max_shift + 1)

            if shift > 0:
                X_aug[i, shift:, :] = X[i, :-shift, :]
                X_aug[i, :shift, :] = X[i, 0:1, :]  # Pad with first value
            elif shift < 0:
                X_aug[i, :shift, :] = X[i, -shift:, :]
                X_aug[i, shift:, :] = X[i, -1:, :]  # Pad with last value
            else:
                X_aug[i] = X[i]

        return X_aug

    def amplitude_scale(self, X, scale_range=(0.9, 1.1)):
        """
        Scale signal amplitude by random factor.

        Args:
            X: (N, T, C) EEG data
            scale_range: (min, max) scaling factors

        Returns:
            Augmented data: (N, T, C)
        """
        N, T, C = X.shape
        X_aug = np.zeros_like(X)

        for i in range(N):
            scale = self.rng.uniform(scale_range[0], scale_range[1])
            X_aug[i] = X[i] * scale

        return X_aug

    def add_gaussian_noise(self, X, noise_std=0.01):
        """
        Add Gaussian noise to signal.

        Args:
            X: (N, T, C) EEG data
            noise_std: Standard deviation of noise relative to signal std

        Returns:
            Augmented data: (N, T, C)
        """
        N, T, C = X.shape

        # Compute per-channel std
        signal_std = X.std(axis=1, keepdims=True)  # (N, 1, C)

        # Generate noise scaled by signal std
        noise = self.rng.randn(N, T, C) * signal_std * noise_std

        return X + noise

    def time_warp(self, X, warp_range=(0.95, 1.05)):
        """
        Slightly stretch or compress the time axis.

        Args:
            X: (N, T, C) EEG data
            warp_range: (min, max) warping factors

        Returns:
            Augmented data: (N, T, C)
        """
        N, T, C = X.shape
        X_aug = np.zeros_like(X)

        for i in range(N):
            warp_factor = self.rng.uniform(warp_range[0], warp_range[1])
            new_T = int(T * warp_factor)

            # Resample each channel
            for c in range(C):
                # Create time indices
                old_indices = np.linspace(0, T - 1, new_T)
                resampled = np.interp(old_indices, np.arange(T), X[i, :, c])

                # Crop or pad to original length
                if new_T >= T:
                    X_aug[i, :, c] = resampled[:T]
                else:
                    X_aug[i, :new_T, c] = resampled
                    X_aug[i, new_T:, c] = resampled[-1]  # Pad with last value

        return X_aug

    def augment(self, X, methods=['amplitude_scale', 'add_gaussian_noise'], **kwargs):
        """
        Apply multiple augmentation methods.

        Args:
            X: (N, T, C) EEG data
            methods: List of augmentation method names
            **kwargs: Arguments for specific methods

        Returns:
            Augmented data: (N, T, C)
        """
        X_aug = X.copy()

        for method in methods:
            if method == 'time_jitter':
                max_shift = kwargs.get('max_shift', 10)
                X_aug = self.time_jitter(X_aug, max_shift=max_shift)

            elif method == 'amplitude_scale':
                scale_range = kwargs.get('scale_range', (0.9, 1.1))
                X_aug = self.amplitude_scale(X_aug, scale_range=scale_range)

            elif method == 'add_gaussian_noise':
                noise_std = kwargs.get('noise_std', 0.01)
                X_aug = self.add_gaussian_noise(X_aug, noise_std=noise_std)

            elif method == 'time_warp':
                warp_range = kwargs.get('warp_range', (0.95, 1.05))
                X_aug = self.time_warp(X_aug, warp_range=warp_range)

        return X_aug


def augment_dataset(X, y, subjects, augmentation_factor=2, seed=42):
    """
    Create augmented dataset by applying random augmentations.

    Args:
        X: (N, T, C) EEG data
        y: (N,) labels
        subjects: (N,) subject IDs
        augmentation_factor: How many augmented copies to create per sample
        seed: Random seed

    Returns:
        X_combined: (N * (1 + augmentation_factor), T, C)
        y_combined: (N * (1 + augmentation_factor),)
        subjects_combined: (N * (1 + augmentation_factor),)
    """
    augmenter = EEGAugmenter(seed=seed)

    X_list = [X]  # Original data
    y_list = [y]
    subjects_list = [subjects]

    # Create augmented copies
    for i in range(augmentation_factor):
        print(f"  Creating augmented copy {i+1}/{augmentation_factor}...")

        # Randomly select augmentation methods
        X_aug = augmenter.augment(
            X,
            methods=['amplitude_scale', 'add_gaussian_noise'],
            scale_range=(0.9, 1.1),
            noise_std=0.01
        )

        X_list.append(X_aug)
        y_list.append(y)
        subjects_list.append(subjects)

    # Combine
    X_combined = np.concatenate(X_list, axis=0)
    y_combined = np.concatenate(y_list, axis=0)
    subjects_combined = np.concatenate(subjects_list, axis=0)

    print(f"  Original dataset: {len(X)} samples")
    print(f"  Augmented dataset: {len(X_combined)} samples ({augmentation_factor}x increase)")

    return X_combined, y_combined, subjects_combined
