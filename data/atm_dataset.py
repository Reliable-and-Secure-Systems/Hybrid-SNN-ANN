import glob
import os

import numpy as np
import pandas as pd


# Extracted ATM dataset
ATM_ROOT = os.path.expanduser(
    "~/ATM-data/ATM-main"
)


def load_atm_dataset():

    pattern = os.path.join(
        ATM_ROOT,
        "*",
        "ATM_EEG",
        "source_space",
        "full_window",
        "dSPM",
        "broadband",
        "csv",
        "MOTOR_IMAGERY-*_trial_*_atm.csv"
    )

    files = glob.glob(pattern)

    left_files = [
        f for f in files
        if "MOTOR_IMAGERY-LEFT" in os.path.basename(f)
    ]

    right_files = [
        f for f in files
        if "MOTOR_IMAGERY-RIGHT" in os.path.basename(f)
    ]

    left_files.sort()
    right_files.sort()

    print("LEFT files found :", len(left_files))
    print("RIGHT files found:", len(right_files))

    data = []
    labels = []

    files_with_labels = (
        [(f, 0) for f in left_files]
        + [(f, 1) for f in right_files]
    )

    files_with_labels.sort()

    for index, (file, label) in enumerate(files_with_labels):

        df = pd.read_csv(file)

        # Remove the first column containing parcel names.
        # Remaining values form the 68 x 68 ATM matrix.
        values = df.iloc[:, 1:].values.astype(np.float32)

        data.append(values)
        labels.append(label)

    X = np.array(data, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    return X, y


if __name__ == "__main__":

    X, y = load_atm_dataset()

    print("=" * 50)
    print("ATM DATASET")
    print("=" * 50)

    print("Data shape   :", X.shape)
    print("Labels shape :", y.shape)

    print("LEFT samples :", np.sum(y == 0))
    print("RIGHT samples:", np.sum(y == 1))

    print("Minimum      :", X.min())
    print("Maximum      :", X.max())
    print("Mean         :", X.mean())
    print("Std          :", X.std())

    print("=" * 50)
