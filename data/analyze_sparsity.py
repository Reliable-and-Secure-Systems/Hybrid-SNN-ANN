from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path.home() / "ATM-data/ATM-main"

manifest = pd.read_csv(ROOT / "paired_trial_manifest.csv")
eeg = np.load(ROOT / "paired_eeg_matrices.npy")
fnirs = np.load(ROOT / "paired_fnirs_matrices.npy")

manifest["eeg_nonzero_count"] = np.count_nonzero(
    eeg,
    axis=(1, 2)
)

manifest["fnirs_nonzero_count"] = np.count_nonzero(
    fnirs,
    axis=(1, 2)
)

manifest["eeg_is_zero"] = (
    manifest["eeg_nonzero_count"] == 0
)

manifest["fnirs_is_zero"] = (
    manifest["fnirs_nonzero_count"] == 0
)

print("\n========== OVERALL ==========")

print(
    "EEG zero percentage:",
    round(manifest["eeg_is_zero"].mean() * 100, 2),
    "%"
)

print(
    "fNIRS zero percentage:",
    round(manifest["fnirs_is_zero"].mean() * 100, 2),
    "%"
)

print("\n========== BY SUBJECT ==========")

by_subject = manifest.groupby("subject").agg(
    trials=("subject", "size"),
    eeg_zero_percentage=("eeg_is_zero", lambda x: x.mean() * 100),
    fnirs_zero_percentage=("fnirs_is_zero", lambda x: x.mean() * 100),
    eeg_average_nonzero=("eeg_nonzero_count", "mean"),
    fnirs_average_nonzero=("fnirs_nonzero_count", "mean"),
)

print(by_subject.round(2).to_string())

print("\n========== BY GROUP ==========")

by_group = manifest.groupby("group").agg(
    trials=("group", "size"),
    eeg_zero_percentage=("eeg_is_zero", lambda x: x.mean() * 100),
    fnirs_zero_percentage=("fnirs_is_zero", lambda x: x.mean() * 100),
    eeg_average_nonzero=("eeg_nonzero_count", "mean"),
    fnirs_average_nonzero=("fnirs_nonzero_count", "mean"),
)

print(by_group.round(2).to_string())

print("\n========== BY TASK ==========")

by_task = manifest.groupby("task").agg(
    trials=("task", "size"),
    eeg_zero_percentage=("eeg_is_zero", lambda x: x.mean() * 100),
    fnirs_zero_percentage=("fnirs_is_zero", lambda x: x.mean() * 100),
    eeg_average_nonzero=("eeg_nonzero_count", "mean"),
    fnirs_average_nonzero=("fnirs_nonzero_count", "mean"),
)

print(by_task.round(2).to_string())

print("\n========== BY HAND ==========")

by_hand = manifest.groupby("hand").agg(
    trials=("hand", "size"),
    eeg_zero_percentage=("eeg_is_zero", lambda x: x.mean() * 100),
    fnirs_zero_percentage=("fnirs_is_zero", lambda x: x.mean() * 100),
    eeg_average_nonzero=("eeg_nonzero_count", "mean"),
    fnirs_average_nonzero=("fnirs_nonzero_count", "mean"),
)

print(by_hand.round(2).to_string())

print("\n========== NONZERO DISTRIBUTION ==========")

print("\nEEG:")
print(manifest["eeg_nonzero_count"].describe())

print("\nfNIRS:")
print(manifest["fnirs_nonzero_count"].describe())