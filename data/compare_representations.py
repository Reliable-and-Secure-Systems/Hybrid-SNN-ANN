from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.home() / "ATM-data/ATM-main/S01_Sess01"

files = {
    "EEG sensor beta hub_csv": ROOT / "ATM_EEG/sensor_space/full_window/beta/hub_csv/MOTOR_IMAGERY-LEFT_trial_24_atm.csv",
    "EEG sensor beta csv": ROOT / "ATM_EEG/sensor_space/full_window/beta/csv/MOTOR_IMAGERY-LEFT_trial_24_atm.csv",
    "EEG source broadband hub_csv": ROOT / "ATM_EEG/source_space/full_window/dSPM/broadband/hub_csv/MOTOR_IMAGERY-LEFT_trial_24_atm.csv",
    "EEG source broadband csv": ROOT / "ATM_EEG/source_space/full_window/dSPM/broadband/csv/MOTOR_IMAGERY-LEFT_trial_24_atm.csv",
    "EEG connectivity beta": ROOT / "ATM_EEG/sensor_space/full_window/beta/connectivity/atm_eeg_MOTOR_IMAGERY-LEFT.csv",
    "fNIRS parcel broadband hub_csv": ROOT / "ATM_fNIRS/parcel_space/hrf_window/nearest_vertex/broadband/hub_csv/MOTOR_IMAGERY-LEFT_trial_24_atm.csv",
    "fNIRS parcel broadband csv": ROOT / "ATM_fNIRS/parcel_space/hrf_window/nearest_vertex/broadband/csv/MOTOR_IMAGERY-LEFT_trial_24_atm.csv",
    "fNIRS parcel task_hrf hub_csv": ROOT / "ATM_fNIRS/parcel_space/hrf_window/nearest_vertex/task_hrf/hub_csv/MOTOR_IMAGERY-LEFT_trial_24_atm.csv",
    "fNIRS parcel task_hrf csv": ROOT / "ATM_fNIRS/parcel_space/hrf_window/nearest_vertex/task_hrf/csv/MOTOR_IMAGERY-LEFT_trial_24_atm.csv",
}

for name, path in files.items():
    print("\n" + "=" * 80)
    print(name)
    print("Path:", path)
    print("Exists:", path.exists())

    if not path.exists():
        continue

    try:
        df = pd.read_csv(path)
        print("Shape:", df.shape)
        print("Columns:", list(df.columns)[:10])
        print(df.head(3).to_string())

        numeric = df.select_dtypes(include=[np.number])

        if numeric.shape[1] > 0:
            values = numeric.to_numpy(dtype=float)
            print("Numeric shape:", values.shape)
            print("Min:", values.min())
            print("Max:", values.max())
            print("Mean:", values.mean())
            print("Std:", values.std())
            print("Unique values, first 20:", np.unique(values)[:20])
            print("Non-zero count:", np.count_nonzero(values))
        else:
            print("No numeric columns detected.")

    except Exception as e:
        print("Could not read:", repr(e))
