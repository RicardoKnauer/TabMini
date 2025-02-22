import os
import glob
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  

def get_data_dir():
    OFFLINE_DATA_DIR = dict()
    for path in glob.glob("tabmini/original_data/*/*"):
        if os.path.basename(path) not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            OFFLINE_DATA_DIR[os.path.basename(path)] = path

    return OFFLINE_DATA_DIR

OFFLINE_DATA_DIR = get_data_dir()

if __name__ == "__main__":
    print(OFFLINE_DATA_DIR)