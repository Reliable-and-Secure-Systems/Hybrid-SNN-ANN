import glob
import os

import numpy as np
import pandas as pd


# Location of the cloned ATM repository
ATM_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "ATM"
)

DATA_PATH = os.path.join(
    ATM_ROOT,
    "S01_Sess02",
    "atm_eeg",
    "dSPM",
    "full_window",
    "broadband",
    "csv"
)


def load_atm_dataset():

    left_files = glob.glob(
        os.path.join(
            DATA_PATH,
            "MOTOR_IMAGERY-LEFT_trial_*_atm.csv"
        )
    )

    right_files = glob.glob(
        os.path.join(
            DATA_PATH,
            "MOTOR_IMAGERY-RIGHT_trial_*_atm.csv"
        )
    )

    files = left_files + right_files

    files.sort()

    data = []
    labels = []

    for file in files:

        df = pd.read_csv(file)

        # Remove the first column containing parcel names
        values = df.iloc[:, 1:].values.astype(np.float32)

        data.append(values)

        if "MOTOR_IMAGERY-LEFT" in file:
            labels.append(0)

        elif "MOTOR_IMAGERY-RIGHT" in file:
            labels.append(1)

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