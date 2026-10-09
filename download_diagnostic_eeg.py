import os
import re
import requests
from bs4 import BeautifulSoup

OUTPUT_DIR = "data/atm_diagnostic_eeg"

groups = {
    "HC": [
        "S01_Sess01", "S01_Sess02",
        "S06_Sess01",
        "S07_Sess01", "S07_Sess02",
        "S17_Sess01", "S17_Sess02",
        "S20_Sess01", "S20_Sess02",
    ],
    "MCI": [
        "S03_Sess01",
        "S09_Sess01", "S09_Sess02",
        "S11_Sess01",
        "S13_Sess01",
        "S14_Sess01",
        "S15_Sess01",
    ],
    "AD": [
        "S16_Sess01", "S16_Sess02",
        "S18_Sess01",
        "S19_Sess01", "S19_Sess02",
        "S21_Sess01",
        "S27_Sess01",
        "S30_Sess01",
        "S41_Sess01", "S41_Sess02",
        "S43_Sess01", "S43_Sess02",
        "S47_Sess01", "S47_Sess02",
    ],
}


def get_trial_files(session):
    path = (
        f"S09_Sess01/ATM_EEG/source_space/full_window/"
        f"dSPM/broadband/csv"
    )

    # Replace the hard-coded session above
    path = (
        f"{session}/ATM_EEG/source_space/full_window/"
        f"dSPM/broadband/csv"
    )

    url = f"https://github.com/QinXinlan/ATM/tree/main/{path}"

    response = requests.get(url)
    response.raise_for_status()

    # Find only actual trial CSVs.
    pattern = r'MOTOR_IMAGERY-(?:LEFT|RIGHT)_trial_\d+_atm\.csv'

    filenames = sorted(set(re.findall(pattern, response.text)))

    return filenames


total_downloaded = 0

for group, sessions in groups.items():

    for session in sessions:

        print(f"\nChecking {group} / {session}")

        try:
            filenames = get_trial_files(session)
        except requests.RequestException as e:
            print(f"Could not access {session}: {e}")
            continue

        print(f"Found {len(filenames)} trials")

        session_dir = os.path.join(
            OUTPUT_DIR,
            group,
            session
        )

        os.makedirs(session_dir, exist_ok=True)

        downloaded = 0

        for filename in filenames:

            output_file = os.path.join(
                session_dir,
                filename
            )

            # Already downloaded → skip.
            if os.path.exists(output_file):
                continue

            raw_url = (
                "https://raw.githubusercontent.com/"
                "QinXinlan/ATM/main/"
                f"{session}/ATM_EEG/source_space/full_window/"
                f"dSPM/broadband/csv/{filename}"
            )

            try:
                response = requests.get(raw_url)
                response.raise_for_status()

                with open(output_file, "wb") as f:
                    f.write(response.content)

                downloaded += 1
                total_downloaded += 1

                print(f"Downloaded: {filename}")

            except requests.RequestException as e:
                print(f"Failed: {filename}")
                print(e)

        print(f"New files downloaded: {downloaded}")


print("\n==============================")
print("Download process complete")
print(f"New files downloaded: {total_downloaded}")
print("==============================")