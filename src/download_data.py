"""
Downloads the IEEE-CIS Fraud Detection dataset via kagglehub and copies
the relevant CSVs into data/ where the rest of the pipeline expects them.

This is a Kaggle COMPETITION dataset, not a plain dataset — you must
first join the "IEEE-CIS Fraud Detection" competition on kaggle.com
(one click, "Late Submission" access) before the API will let you
download it. If you skip that step, kagglehub will raise a 403.

Requires Kaggle API credentials configured
(~/.kaggle/kaggle.json, or KAGGLE_USERNAME/KAGGLE_KEY env vars).

Run once, from the project root:
    python src/download_data.py
"""
import os
import shutil
import kagglehub

TARGET_FILES = ["train_transaction.csv", "train_identity.csv"]


def download():
    print("Downloading IEEE-CIS Fraud Detection dataset via kagglehub...")
    print("(If this fails with a 403, join the competition first at:")
    print(" https://www.kaggle.com/competitions/ieee-fraud-detection )")

    path = kagglehub.competition_download("ieee-fraud-detection")
    print(f"Downloaded to: {path}")

    os.makedirs("data", exist_ok=True)
    found = []
    for root, _, files in os.walk(path):
        for fname in files:
            if fname in TARGET_FILES:
                src = os.path.join(root, fname)
                dst = os.path.join("data", fname)
                shutil.copy(src, dst)
                found.append(fname)
                print(f"Copied {fname} -> {dst}")

    missing = set(TARGET_FILES) - set(found)
    if missing:
        raise FileNotFoundError(
            f"Could not find {missing} in the downloaded dataset at {path}. "
            f"Check the competition download contents manually."
        )
    print("\nDone. train_transaction.csv and train_identity.csv are in data/.")


if __name__ == "__main__":
    download()
