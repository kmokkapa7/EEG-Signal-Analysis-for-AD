# features.py
import numpy as np
from scipy.signal import welch, coherence
import matplotlib.pyplot as plt

# -----------------------------------
# Global EEG band definitions
# -----------------------------------
bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
}

ALPHA_BAND = (8, 13)


# -----------------------------------
# Basic utilities
# -----------------------------------

def bandpower(signal, fs, band, nperseg=128):
    """
    Compute band power in a given frequency band using Welch PSD.
    signal: 1D array (time,)
    fs: sampling frequency
    band: (fmin, fmax)
    """
    fmin, fmax = band
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    if not np.any(idx):
        return 0.0
    bp = np.trapz(psd[idx], freqs[idx])
    return bp


def spectral_entropy(signal, fs, nperseg=128):
    """
    Spectral entropy based on Welch PSD.
    """
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    psd_sum = np.sum(psd)
    if psd_sum == 0:
        return 0.0
    psd_norm = psd / psd_sum
    psd_norm = psd_norm + 1e-12  # avoid log(0)
    se = -np.sum(psd_norm * np.log2(psd_norm))
    return se


def hjorth_params(signal):
    """
    Compute Hjorth activity, mobility and complexity for 1D signal.
    """
    x = np.asarray(signal)
    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)

    activity = float(var_x)

    if var_x > 0:
        mobility = float(np.sqrt(var_dx / var_x))
    else:
        mobility = 0.0

    if var_dx > 0 and mobility > 0:
        complexity = float(np.sqrt(var_ddx / var_dx) / mobility)
    else:
        complexity = 0.0

    return activity, mobility, complexity


def peak_alpha_frequency(signal, fs, alpha_band=ALPHA_BAND, nperseg=128):
    """
    Peak alpha frequency within alpha band.
    Returns NaN if no valid alpha bins.
    """
    fmin, fmax = alpha_band
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    if not np.any(idx):
        return np.nan
    sub_freqs = freqs[idx]
    sub_psd = psd[idx]
    if np.all(sub_psd == 0):
        return np.nan
    peak_idx = np.argmax(sub_psd)
    return float(sub_freqs[peak_idx])


def alpha_band_coherence(sig_ref, sig_ch, fs, alpha_band=ALPHA_BAND):
    """
    Mean alpha-band coherence between two signals.
    """
    f, Cxy = coherence(sig_ref, sig_ch, fs=fs, nperseg=128)
    idx = np.logical_and(f >= alpha_band[0], f <= alpha_band[1])
    if not np.any(idx):
        return 0.0
    return float(np.mean(Cxy[idx]))


# -----------------------------------
# OLD: Bandpower / Entropy / Coherence plots
# -----------------------------------

def plot_bandpower_for_sample(X, fs, sample_idx=0):
    """
    Plot band power per channel for one sample.
    X: (n_samples, n_timepoints, n_channels) preprocessed EEG
    """
    sample = X[sample_idx]          # (time, channels)
    n_times, n_ch = sample.shape

    band_powers = {name: [] for name in bands.keys()}

    for ch in range(n_ch):
        sig = sample[:, ch]
        for name, band in bands.items():
            bp = bandpower(sig, fs, band)
            band_powers[name].append(bp)

    for name in band_powers:
        band_powers[name] = np.array(band_powers[name])

    channels = np.arange(1, n_ch + 1)

    plt.figure(figsize=(10, 6))
    for name, vals in band_powers.items():
        plt.plot(channels, vals, marker="o", label=name)

    plt.xlabel("Channel")
    plt.ylabel("Band Power")
    plt.title(f"Band Power per Channel (Sample {sample_idx})")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_coherence_for_sample(X, fs, sample_idx=0, ref_ch=0, max_pairs=5):
    """
    Plot full-spectrum coherence between a reference channel and a few others.
    """
    sample = X[sample_idx]          # (time, channels)
    n_times, n_ch = sample.shape
    sig_ref = sample[:, ref_ch]

    plt.figure(figsize=(12, 8))

    for ch in range(1, min(max_pairs, n_ch)):
        sig_ch = sample[:, ch]
        f, Cxy = coherence(sig_ref, sig_ch, fs=fs, nperseg=128)
        plt.plot(f, Cxy, label=f"Ch {ref_ch+1}–Ch {ch+1}")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")
    plt.title(f"Coherence with Channel {ref_ch+1} (Sample {sample_idx})")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_entropy_for_sample(X, fs, sample_idx=0):
    """
    Plot spectral entropy per channel for one sample.
    """
    sample = X[sample_idx]          # (time, channels)
    n_times, n_ch = sample.shape

    entropies = []
    for ch in range(n_ch):
        sig = sample[:, ch]
        se = spectral_entropy(sig, fs)
        entropies.append(se)

    entropies = np.array(entropies)
    channels = np.arange(1, n_ch + 1)

    plt.figure(figsize=(8, 5))
    plt.bar(channels, entropies)
    plt.xlabel("Channel")
    plt.ylabel("Spectral Entropy")
    plt.title(f"Spectral Entropy per Channel (Sample {sample_idx})")
    plt.grid(True)
    plt.show()


# -----------------------------------
# NEW: Relative bandpower & band ratios
# -----------------------------------

def plot_relative_bandpower_for_sample(X, fs, sample_idx=0):
    """
    Plot relative bandpower (band power / sum of all bands) per channel.
    """
    sample = X[sample_idx]          # (time, channels)
    n_times, n_ch = sample.shape

    abs_bp = {name: [] for name in bands.keys()}

    for ch in range(n_ch):
        sig = sample[:, ch]
        for name, band in bands.items():
            bp = bandpower(sig, fs, band)
            abs_bp[name].append(bp)

    for name in abs_bp:
        abs_bp[name] = np.array(abs_bp[name])

    total_power = np.zeros(n_ch)
    for name in abs_bp:
        total_power += abs_bp[name]

    rel_bp = {}
    for name in abs_bp:
        rel_bp[name] = np.divide(abs_bp[name], total_power + 1e-12)

    channels = np.arange(1, n_ch + 1)

    plt.figure(figsize=(10, 6))
    for name, vals in rel_bp.items():
        plt.plot(channels, vals, marker="o", label=name)

    plt.xlabel("Channel")
    plt.ylabel("Relative Band Power")
    plt.title(f"Relative Band Power per Channel (Sample {sample_idx})")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_band_ratios_for_sample(X, fs, sample_idx=0):
    """
    Plot theta/alpha and delta/alpha ratios per channel.
    """
    sample = X[sample_idx]
    n_times, n_ch = sample.shape

    theta_alpha = []
    delta_alpha = []

    for ch in range(n_ch):
        sig = sample[:, ch]
        bp_delta = bandpower(sig, fs, bands["delta"])
        bp_theta = bandpower(sig, fs, bands["theta"])
        bp_alpha = bandpower(sig, fs, bands["alpha"])

        if bp_alpha == 0:
            theta_alpha.append(0.0)
            delta_alpha.append(0.0)
        else:
            theta_alpha.append(bp_theta / bp_alpha)
            delta_alpha.append(bp_delta / bp_alpha)

    theta_alpha = np.array(theta_alpha)
    delta_alpha = np.array(delta_alpha)
    channels = np.arange(1, n_ch + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(channels, theta_alpha, marker="o", label="Theta/Alpha")
    plt.plot(channels, delta_alpha, marker="s", label="Delta/Alpha")

    plt.xlabel("Channel")
    plt.ylabel("Band Ratio")
    plt.title(f"Band Ratios per Channel (Sample {sample_idx})")
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------------------
# NEW: Hjorth parameters
# -----------------------------------

def plot_hjorth_for_sample(X, fs, sample_idx=0):
    """
    Plot Hjorth activity, mobility and complexity per channel.
    (fs is not used here; kept for consistency.)
    """
    sample = X[sample_idx]
    n_times, n_ch = sample.shape

    activity = []
    mobility = []
    complexity = []

    for ch in range(n_ch):
        sig = sample[:, ch]
        a, m, c = hjorth_params(sig)
        activity.append(a)
        mobility.append(m)
        complexity.append(c)

    activity = np.array(activity)
    mobility = np.array(mobility)
    complexity = np.array(complexity)
    channels = np.arange(1, n_ch + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(channels, activity, marker="o", label="Activity")
    plt.plot(channels, mobility, marker="s", label="Mobility")
    plt.plot(channels, complexity, marker="^", label="Complexity")

    plt.xlabel("Channel")
    plt.ylabel("Hjorth value")
    plt.title(f"Hjorth Parameters per Channel (Sample {sample_idx})")
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------------------
# NEW: Peak alpha frequency
# -----------------------------------

def plot_paf_for_sample(X, fs, sample_idx=0):
    """
    Plot peak alpha frequency (PAF) per channel.
    """
    sample = X[sample_idx]
    n_times, n_ch = sample.shape

    pafs = []
    for ch in range(n_ch):
        sig = sample[:, ch]
        paf = peak_alpha_frequency(sig, fs, alpha_band=ALPHA_BAND)
        pafs.append(paf)

    pafs = np.array(pafs)
    channels = np.arange(1, n_ch + 1)

    plt.figure(figsize=(10, 6))
    plt.bar(channels, pafs)
    plt.xlabel("Channel")
    plt.ylabel("Peak Alpha Frequency (Hz)")
    plt.title(f"Peak Alpha Frequency per Channel (Sample {sample_idx})")
    plt.grid(True)
    plt.show()


# -----------------------------------
# NEW: Alpha-band coherence summary
# -----------------------------------

def plot_alpha_coherence_for_sample(X, fs, sample_idx=0, ref_ch=0):
    """
    Plot mean alpha-band coherence between a reference channel and others.
    """
    sample = X[sample_idx]
    n_times, n_ch = sample.shape
    sig_ref = sample[:, ref_ch]

    alpha_coh = []
    channels = []

    for ch in range(n_ch):
        if ch == ref_ch:
            continue
        sig_ch = sample[:, ch]
        coh_val = alpha_band_coherence(sig_ref, sig_ch, fs)
        alpha_coh.append(coh_val)
        channels.append(ch + 1)

    alpha_coh = np.array(alpha_coh)

    plt.figure(figsize=(10, 6))
    plt.bar(channels, alpha_coh)
    plt.xlabel("Channel")
    plt.ylabel("Mean Alpha Coherence")
    plt.title(
        f"Alpha-band Coherence with Channel {ref_ch + 1} (Sample {sample_idx})"
    )
    plt.grid(True)
    plt.show()


# -----------------------------------
# NEW: extract_features_all
# -----------------------------------

def extract_features_all(X, fs, ref_ch=0):
    """
    Extract a rich set of handcrafted features for all samples.

    X:  (n_samples, n_timepoints, n_channels) preprocessed EEG
    fs: sampling frequency
    ref_ch: reference channel for alpha-band coherence

    Features per sample:
      - Absolute band power:          n_bands * n_ch
      - Relative band power:          n_bands * n_ch
      - Band ratios (theta/alpha,
        delta/alpha):                 2 * n_ch
      - Spectral entropy:             1 * n_ch
      - Hjorth (activity, mobility,
        complexity):                  3 * n_ch
      - Peak alpha frequency (PAF):   1 * n_ch
      - Alpha-band coherence (ref_ch
        to others):                   (n_ch - 1)

    Total = (2*n_bands + 7)*n_ch + (n_ch - 1)
    """
    n_samples, n_times, n_ch = X.shape
    n_bands = len(bands)

    n_features = (2 * n_bands + 7) * n_ch + (n_ch - 1)
    F = np.zeros((n_samples, n_features), dtype=np.float32)

    band_names = list(bands.keys())

    for i in range(n_samples):
        sample = X[i]  # (time, channels)
        feat_vec = []

        # ----- 1. Absolute band power per band per channel -----
        abs_bp = {name: [] for name in band_names}
        for ch in range(n_ch):
            sig = sample[:, ch]
            for name in band_names:
                bp = bandpower(sig, fs, bands[name])
                abs_bp[name].append(bp)

        for name in abs_bp:
            abs_bp[name] = np.array(abs_bp[name])

        # store absolute band powers
        for name in band_names:
            feat_vec.extend(abs_bp[name].tolist())

        # ----- 2. Relative band power per band per channel -----
        total_power = np.zeros(n_ch)
        for name in band_names:
            total_power += abs_bp[name]

        for name in band_names:
            rel = np.divide(abs_bp[name], total_power + 1e-12)
            feat_vec.extend(rel.tolist())

        # ----- 3. Band ratios: theta/alpha, delta/alpha -----
        theta_alpha = []
        delta_alpha = []
        theta_bp = abs_bp["theta"]
        delta_bp = abs_bp["delta"]
        alpha_bp = abs_bp["alpha"]

        for ch in range(n_ch):
            a = alpha_bp[ch]
            if a == 0:
                theta_alpha.append(0.0)
                delta_alpha.append(0.0)
            else:
                theta_alpha.append(theta_bp[ch] / a)
                delta_alpha.append(delta_bp[ch] / a)

        feat_vec.extend(theta_alpha)
        feat_vec.extend(delta_alpha)

        # ----- 4. Spectral entropy per channel -----
        se_list = []
        for ch in range(n_ch):
            sig = sample[:, ch]
            se_val = spectral_entropy(sig, fs)
            se_list.append(se_val)
        feat_vec.extend(se_list)

        # ----- 5. Hjorth parameters per channel -----
        act_list = []
        mob_list = []
        comp_list = []
        for ch in range(n_ch):
            sig = sample[:, ch]
            a, m, c = hjorth_params(sig)
            act_list.append(a)
            mob_list.append(m)
            comp_list.append(c)

        feat_vec.extend(act_list)
        feat_vec.extend(mob_list)
        feat_vec.extend(comp_list)

        # ----- 6. Peak alpha frequency per channel -----
        paf_list = []
        for ch in range(n_ch):
            sig = sample[:, ch]
            paf_val = peak_alpha_frequency(sig, fs, alpha_band=ALPHA_BAND)
            paf_list.append(0.0 if np.isnan(paf_val) else paf_val)

        feat_vec.extend(paf_list)

        # ----- 7. Alpha-band coherence with reference channel -----
        sig_ref = sample[:, ref_ch]
        alpha_coh = []
        for ch in range(n_ch):
            if ch == ref_ch:
                continue
            sig_ch = sample[:, ch]
            coh_val = alpha_band_coherence(sig_ref, sig_ch, fs)
            alpha_coh.append(coh_val)

        feat_vec.extend(alpha_coh)

        # assign to feature matrix
        F[i, :] = np.array(feat_vec, dtype=np.float32)

    return F

def get_feature_names(n_channels):
    names = []

    bands = ["delta", "theta", "alpha", "beta"]

    # Absolute band power
    for b in bands:
        for ch in range(n_channels):
            names.append(f"{b}_abs_ch{ch+1}")

    # Relative band power
    for b in bands:
        for ch in range(n_channels):
            names.append(f"{b}_rel_ch{ch+1}")

    # Ratios
    for ch in range(n_channels):
        names.append(f"theta_alpha_ch{ch+1}")
    for ch in range(n_channels):
        names.append(f"delta_alpha_ch{ch+1}")

    # Entropy
    for ch in range(n_channels):
        names.append(f"entropy_ch{ch+1}")

    # Hjorth
    for ch in range(n_channels):
        names.append(f"hjorth_activity_ch{ch+1}")
    for ch in range(n_channels):
        names.append(f"hjorth_mobility_ch{ch+1}")
    for ch in range(n_channels):
        names.append(f"hjorth_complexity_ch{ch+1}")

    # PAF
    for ch in range(n_channels):
        names.append(f"paf_ch{ch+1}")

    # Alpha coherence (ref ch removed)
    for ch in range(1, n_channels):
        names.append(f"alpha_coh_ch{ch+1}")

    return names
