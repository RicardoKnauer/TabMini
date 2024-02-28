from pathlib import Path
from typing import Tuple

import pandas as pd
from pmlb import fetch_data

from automl.data import data_info


def load_dataset(reduced: bool = False) -> dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load the dataset for AutoML. The datasets are loaded from the PMLB library.
    :param reduced: Whether to exclude the datasets that have been used to train TabPFN. Default is False.
    :return: A dictionary containing the loaded dataset. The key is the dataset name and the value is a tuple containing
    the input features and the target variable.
    """
    yielded = False
    # Load the dataset
    for idx, _datasets in enumerate(data_info.files):
        datasets = _datasets if not reduced else [file for file in _datasets if data_info.is_not_excluded(file)]

        for dataset_name in datasets:
            fetched_data = fetch_data(dataset_name)
            if not isinstance(fetched_data, pd.DataFrame):
                print(f"Dataset {dataset_name} is not an instance of DataFrame. Skipping...")
                continue

            data = fetched_data.sample(frac=1, random_state=42).reset_index(drop=True)

            yielded = True  # At least one dataset was loaded
            yield dataset_name, (data.drop(columns=["target"]), data["target"])

    if not yielded:
        raise ValueError("No datasets were loaded.")


def load_dummy_dataset() -> dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load a smaller subset of the dataset for AutoML. The datasets are loaded from the PMLB library.
    This is for testing purposes only.
    :return: A dictionary containing the loaded dataset. The key is the dataset name and the value is a tuple containing
    the input features and the target variable.
    """
    print("YOU ARE USING THE DUMMY DATASET LOADER. THIS IS FOR TESTING PURPOSES ONLY.")
    # We want to load the first ten rows of every dataset
    for idx, _datasets in enumerate(data_info.files[0:2]):
        for dataset_name in _datasets[0:2]:
            fetched_data = fetch_data(dataset_name)
            if not isinstance(fetched_data, pd.DataFrame):
                print(f"Dataset {dataset_name} is not an instance of DataFrame. Skipping...")
                continue

            data = fetched_data.sample(frac=1, random_state=42).reset_index(drop=True)

            yield dataset_name, (data.drop(columns=["target"]).head(20), data["target"].head(20))


def save_dataset(dataset: dict[str, Tuple[pd.DataFrame, pd.DataFrame]], path: Path = Path("datasets")) -> None:
    """
    Save the dataset for AutoML. The datasets are loaded from the PMLB library.
    :param dataset: A dictionary containing the loaded dataset. The key is the dataset name and the value is a tuple
    containing the input features and the target variable.
    :param path: The path to save the datasets.
    """
    for idx, (dataset_name, (X, y)) in enumerate(dataset.items()):
        output_dir = path / f"{idx + 1}" / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        X.to_csv(output_dir, "X.csv", index=False)
        y.to_csv(output_dir, "y.csv", index=False)

        print(f"Saved {dataset_name} to {output_dir}")
