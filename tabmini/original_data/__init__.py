import os
import glob
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # tabmini/original_data

def get_data_dir():
    OFFLINE_DATA_DIR = dict()
    for path in glob.glob(f"{str(ROOT)}/*/*"):
        if os.path.basename(path) not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            OFFLINE_DATA_DIR[os.path.basename(path)] = path

    return OFFLINE_DATA_DIR

OFFLINE_DATA_DIR = get_data_dir()

if __name__ == "__main__":
    print(OFFLINE_DATA_DIR)