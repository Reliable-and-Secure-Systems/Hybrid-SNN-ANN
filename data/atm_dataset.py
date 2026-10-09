import glob
import os

import numpy as np
import pandas as pd


DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "atm_diagnostic_eeg"
)

GROUP_LABELS = {
    "HC": 0,
    "MCI": 1,
    "AD": 2,
}


def load_atm_diagnostic_dataset():
    data = []
    labels = []
    subjects = []
    files_used = []

    for group, label in GROUP_LABELS.items():
        group_path = os.path.join(DATA_PATH, group)

        files = glob.glob(
            os.path.join(group_path, "*", "*_trial_*.csv")
        )
        files.sort()

        print(f"{group}: {len(files)} trial files")

        for file in files:
            df = pd.read_csv(file, header=None)

            # Remove the row/column containing region names.
            values = df.iloc[1:, 1:].values.astype(np.float32)

            if values.shape != (68, 68):
                raise ValueError(
                    f"Unexpected shape {values.shape} in {file}"
                )

            data.append(values)
            labels.append(label)

            subject_session = os.path.basename(
                os.path.dirname(file)
            )
            subject = subject_session.split("_")[0]

            subjects.append(subject)
            files_used.append(file)

    return (
        np.array(data, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(subjects),
        files_used,
    )