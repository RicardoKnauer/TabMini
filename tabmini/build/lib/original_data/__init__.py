import os
import glob

def get_data_dir():
    OFFLINE_DATA_DIR = dict()
    for path in glob.glob("tabmini/original_data/*/*"):
        if os.path.basename(path) not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            OFFLINE_DATA_DIR[os.path.basename(path)] = path

    return OFFLINE_DATA_DIR

OFFLINE_DATA_DIR = get_data_dir()

if __name__ == "__main__":
    print(OFFLINE_DATA_DIR)