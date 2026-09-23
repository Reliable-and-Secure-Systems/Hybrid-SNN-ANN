from pathlib import Path
import re
import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_ROOT = Path.home() / "ATM-data/ATM-main"

METADATA_FILE = DATA_ROOT / "Participants_metadata_anonymized.csv"

EEG_RELATIVE_PATH = (
    "ATM_EEG/source_space/full_window/dSPM/broadband/hub_csv"
)

FNIRS_RELATIVE_PATH = (
    "ATM_fNIRS/parcel_space/hrf_window/nearest_vertex/broadband/hub_csv"
)


# --------------------------------------------------
# Load participant metadata
# --------------------------------------------------

metadata = pd.read_csv(
    METADATA_FILE,
    encoding="latin1"
)

metadata["Subject"] = metadata["Subject"].astype(str).str.strip()
metadata["Sessions"] = metadata["Sessions"].astype(str).str.strip()
metadata["Group"] = metadata["Group"].astype(str).str.strip()


def build_session_labels(metadata_df):
    """
    Creates a mapping:

        (subject, session) -> clinical group

    Example:
        ('S01', 'Sess01') -> 'HC'
    """

    labels = {}

    for _, row in metadata_df.iterrows():
        subject = row["Subject"]
        sessions = row["Sessions"]
        group = row["Group"]

        if pd.isna(group) or group in ("nan", ""):
            continue

        # Sessions may be written as:
        # Sess01
        # Sess01 & Sess02
        # Sess01, Sess02
        session_list = re.split(r"\s*[&,]\s*", sessions)

        for session in session_list:
            session = session.strip()

            if session:
                labels[(subject, session)] = group

    return labels


session_labels = build_session_labels(metadata)


# --------------------------------------------------
# Parse trial filename
# --------------------------------------------------

def parse_trial_filename(filename):
    """
    Example filename:

        MOTOR_IMAGERY-LEFT_trial_24_atm.csv

    Returns:

        task = MOTOR_IMAGERY
        hand = LEFT
        trial = 24
    """

    pattern = r"^(MOTOR_IMAGERY|MOTOR_EXECUTION)-(LEFT|RIGHT)_trial_(\d+)_atm\.csv$"

    match = re.match(pattern, filename)

    if match is None:
        return None

    task, hand, trial_number = match.groups()

    return {
        "task": task,
        "hand": hand,
        "trial": int(trial_number),
    }


# --------------------------------------------------
# Load connectivity matrix
# --------------------------------------------------

def load_matrix(file_path):
    """
    Loads a CSV connectivity matrix.

    The first column contains region names.
    Remaining columns contain numeric matrix values.
    """

    df = pd.read_csv(file_path)

    matrix = df.iloc[:, 1:].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(dtype=np.float32)

    if np.isnan(matrix).any():
        raise ValueError(
            f"NaN values found in matrix: {file_path}"
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Matrix is not square: {file_path}, shape={matrix.shape}"
        )

    return matrix


# --------------------------------------------------
# Build paired dataset
# --------------------------------------------------

records = []
missing_fnirs = []
missing_eeg = []
shape_mismatches = []

session_directories = sorted(
    DATA_ROOT.glob("S*_Sess*")
)

print(f"Found {len(session_directories)} session directories")


for session_dir in session_directories:

    session_name = session_dir.name

    match = re.match(r"^(S\d+)_(Sess\d+)$", session_name)

    if match is None:
        continue

    subject, session = match.groups()

    group = session_labels.get((subject, session))

    if group is None:
        print(
            f"Skipping {subject}/{session}: "
            f"no clinical group in metadata"
        )
        continue

    eeg_dir = session_dir / EEG_RELATIVE_PATH
    fnirs_dir = session_dir / FNIRS_RELATIVE_PATH

    if not eeg_dir.exists():
        print(f"Missing EEG directory: {eeg_dir}")
        continue

    if not fnirs_dir.exists():
        print(f"Missing fNIRS directory: {fnirs_dir}")
        continue

    eeg_files = sorted(
        eeg_dir.glob("MOTOR_*.csv")
    )

    for eeg_file in eeg_files:

        parsed = parse_trial_filename(eeg_file.name)

        if parsed is None:
            continue

        fnirs_file = fnirs_dir / eeg_file.name

        if not fnirs_file.exists():
            missing_fnirs.append(str(fnirs_file))
            continue

        try:
            eeg_matrix = load_matrix(eeg_file)
            fnirs_matrix = load_matrix(fnirs_file)
        except Exception as error:
            print(f"Error loading {eeg_file}: {error}")
            continue

        if eeg_matrix.shape != fnirs_matrix.shape:
            shape_mismatches.append({
                "eeg_file": str(eeg_file),
                "fnirs_file": str(fnirs_file),
                "eeg_shape": eeg_matrix.shape,
                "fnirs_shape": fnirs_matrix.shape,
            })
            continue

        records.append({
            "subject": subject,
            "session": session,
            "group": group,
            "task": parsed["task"],
            "hand": parsed["hand"],
            "trial": parsed["trial"],
            "eeg_file": str(eeg_file),
            "fnirs_file": str(fnirs_file),
            "eeg_matrix": eeg_matrix,
            "fnirs_matrix": fnirs_matrix,
        })


# --------------------------------------------------
# Print summary
# --------------------------------------------------

dataset = pd.DataFrame(records)

print("\n========== DATASET SUMMARY ==========")
print(f"Total paired trials: {len(dataset)}")

if len(dataset) > 0:
    print("\nGroups:")
    print(dataset["group"].value_counts())

    print("\nTasks:")
    print(dataset["task"].value_counts())

    print("\nHands:")
    print(dataset["hand"].value_counts())

    print("\nSubjects:")
    print(dataset["subject"].nunique())

    print("\nMatrix shapes:")
    print(
        dataset["eeg_matrix"]
        .apply(lambda x: x.shape)
        .value_counts()
    )

print("\nMissing fNIRS files:", len(missing_fnirs))
print("Shape mismatches:", len(shape_mismatches))
print("\nFirst 20 missing fNIRS files:")
for file in missing_fnirs[:20]:
    print(file)

# --------------------------------------------------
# Save metadata manifest
# --------------------------------------------------

if len(dataset) > 0:

    manifest = dataset.drop(
        columns=["eeg_matrix", "fnirs_matrix"]
    )

    manifest.to_csv(
        DATA_ROOT / "paired_trial_manifest.csv",
        index=False
    )

    np.save(
        DATA_ROOT / "paired_eeg_matrices.npy",
        np.stack(dataset["eeg_matrix"].to_numpy())
    )

    np.save(
        DATA_ROOT / "paired_fnirs_matrices.npy",
        np.stack(dataset["fnirs_matrix"].to_numpy())
    )

    print("\nSaved:")
    print(DATA_ROOT / "paired_trial_manifest.csv")
    print(DATA_ROOT / "paired_eeg_matrices.npy")
    print(DATA_ROOT / "paired_fnirs_matrices.npy")