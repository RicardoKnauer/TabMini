import pandas as pd
from pmlb import fetch_data

from tabmini.data import data_info
from tabmini.types import TabminiDataset


def load_dataset(reduced: bool = False) -> TabminiDataset:
    """
    Load the dataset for AutoML. The datasets are loaded from the PMLB library.
    :param reduced: Whether to exclude the datasets that have been used to train TabPFN. Default is False.
    :return: A dictionary containing the loaded dataset. The key is the dataset name and the value is a tuple containing
    the input features and the target variable.
    """
    dataset = {}

    print("Loading dataset...")
    for idx, _datasets in enumerate(data_info.files):
        datasets = _datasets if not reduced else [file for file in _datasets if data_info.is_not_excluded(file)]

        for dataset_name in datasets:
            fetched_data = fetch_data(dataset_name)
            if not isinstance(fetched_data, pd.DataFrame):
                print(f"Dataset {dataset_name} is not an instance of DataFrame. Skipping...")
                continue

            data = fetched_data.sample(frac=1, random_state=42).reset_index(drop=True)

            dataset[dataset_name] = (data.drop(columns=["target"]), data["target"])

    # Print on the same line
    print("Dataset loaded.")

    return dataset


def load_dummy_dataset() -> TabminiDataset:
    """
    Load a smaller subset of the dataset for AutoML. The datasets are loaded from the PMLB library.
    This is for testing purposes only.
    :return: A dictionary containing the loaded dataset. The key is the dataset name and the value is a tuple containing
    the input features and the target variable.
    """
    print("YOU ARE USING THE DUMMY DATASET LOADER. THIS IS FOR TESTING PURPOSES ONLY.")

    dataset = {}

    # We want to load the first ten rows of every dataset
    for idx, _datasets in enumerate(data_info.files[0:2]):
        for dataset_name in _datasets[0:2]:
            fetched_data = fetch_data(dataset_name)
            if not isinstance(fetched_data, pd.DataFrame):
                print(f"Dataset {dataset_name} is not an instance of DataFrame. Skipping...")
                continue

            data = fetched_data.sample(frac=1, random_state=42).reset_index(drop=True)

            dataset[dataset_name] = (data.drop(columns=["target"]).head(20), data["target"].head(20))

    return dataset
