# Excluded Files (Too Large for GitHub)

The following files/folders were excluded from this repository because they exceed GitHub's 100MB file size limit or are large binary datasets. They are listed in `.gitignore`.

## Raw Dataset Files

| Path | Approximate Size | Reason |
|------|-----------------|--------|
| `AD_all_patients.csv` | ~116 MB | Exceeds GitHub 100MB limit |
| `integrated_eeg_dataset.npz` | ~2.0 GB | Exceeds GitHub 100MB limit |
| `X_raw_preprocessed.npy` | ~1.9 GB | Exceeds GitHub 100MB limit |

## EEG Iraq Dataset (`EEG_AD_Iraq/` folder)

| Path | Approximate Size |
|------|-----------------|
| `EEG_AD_Iraq/HMMS.csv` | ~295 MB |
| `EEG_AD_Iraq/Mild/Mild1.csv` | ~428 MB |
| `EEG_AD_Iraq/Mild/Mild2.csv` | ~418 MB |
| `EEG_AD_Iraq/Mild/Mild3.csv` | ~487 MB |
| `EEG_AD_Iraq/Mild/Mild4.csv` | ~432 MB |
| `EEG_AD_Iraq/Mild/Mild5.csv` | ~440 MB |
| `EEG_AD_Iraq/Mild/Mild6.csv` | ~441 MB |
| `EEG_AD_Iraq/Mild/Mild7.csv` | ~441 MB |
| `EEG_AD_Iraq/Mild/Mild8.csv` | ~98 MB |
| `EEG_AD_Iraq/Moderate/Mod1.csv` – `Mod12.csv` | ~194 MB – ~643 MB each |
| `EEG_AD_Iraq/Severe/Sv1.csv` – `Sv10.csv` | ~57 MB – ~424 MB each |
| `EEG_AD_Iraq/Healthy/CN1.csv` – `CN23.csv` | Various sizes |

## OpenNeuro BIDS Dataset (`dataset/` folder)

Contains 357 files including raw and preprocessed EEG recordings in BIDS format (`.set`, `.fdt`, `.json`, `.tsv`, `.eeg`, `.vhdr` files). Total size is several GB.

## How to Obtain the Data

- **Iraq EEG dataset**: Available on [Mendeley Data](https://data.mendeley.com/) or the original publication source.
- **OpenNeuro dataset**: Available at [OpenNeuro](https://openneuro.org/) (search for the dataset accession number in `dataset/dataset_description.json`).
- **Preprocessed files** (`integrated_eeg_dataset.npz`, `X_raw_preprocessed.npy`, `AD_all_patients.csv`): Regenerate by running the preprocessing scripts in this repo.
