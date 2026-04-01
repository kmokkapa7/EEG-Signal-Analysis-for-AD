import os
import numpy as np
import pandas as pd
import mne


def load_openneuro_dataset(
    root_dir,
    window_sec=4.0,
    target_fs=128
):
    """
    Load OpenNeuro ds004504 EEG dataset (derivatives only),
    with boundary-aware windowing.

    Returns:
        X        : (N, T, C) float32
        y        : (N,) int64
        subjects : (N,) subject IDs
    """

    participants = pd.read_csv(
        os.path.join(root_dir, "participants.tsv"),
        sep="\t"
    )

    # Dataset-specific diagnosis column
    diag_col = "Group"

    # Dataset-specific label encoding
    label_map = {
        "A": 1,  # Alzheimer's disease
        "C": 0   # Cognitively normal
        # "F" (FTD) intentionally excluded
    }

    X_all, y_all, subj_all = [], [], []

    for _, row in participants.iterrows():
        subj_id = row["participant_id"]
        group = row[diag_col]

        if group not in label_map:
            continue  # exclude FTD

        eeg_path = os.path.join(
            root_dir,
            "derivatives",
            subj_id,
            "eeg",
            f"{subj_id}_task-eyesclosed_eeg.set"
        )

        if not os.path.exists(eeg_path):
            continue

        # ---- Load EEG ----
        raw = mne.io.read_raw_eeglab(
            eeg_path,
            preload=True,
            verbose=False
        )

        # Keep EEG channels only (modern API)
        raw.pick_types(eeg=True)

        # Resample to target frequency
        raw.resample(target_fs)

        # ---- Handle boundary events ----
        # EEGLAB inserts "boundary" annotations where data was cut
        events, event_id = mne.events_from_annotations(
            raw, verbose=False
        )

        boundary_ids = [
            eid for name, eid in event_id.items()
            if "boundary" in name.lower()
        ]

        if boundary_ids:
            boundary_samples = events[
                np.isin(events[:, 2], boundary_ids), 0
            ]
        else:
            boundary_samples = []

        # ---- Windowing without crossing boundaries ----
        data = raw.get_data().T  # (time, channels)
        win_len = int(window_sec * target_fs)

        # Treat boundaries as segment separators
        boundaries = list(boundary_samples) + [len(data)]
        start = 0

        for end in boundaries:
            segment = data[start:end]

            n_windows = segment.shape[0] // win_len
            for i in range(n_windows):
                seg = segment[i * win_len:(i + 1) * win_len]

                X_all.append(seg)
                y_all.append(label_map[group])
                subj_all.append(subj_id)

            start = end

    # ---- Final arrays ----
    X = np.stack(X_all).astype(np.float32)
    y = np.array(y_all, dtype=np.int64)
    subjects = np.array(subj_all)

    return X, y, subjects
