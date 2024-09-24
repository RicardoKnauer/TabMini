from pathlib import Path
from pydoc import doc
from typing import Tuple, Literal

import pandas as pd
from sklearn.base import BaseEstimator

from tabmini import analysis, estimators
from tabmini.analysis import meta_feature, scorer
from tabmini.data import data_handler
from .types import TabminiDataset


def load_dataset(reduced: bool = False) -> TabminiDataset:
    """
    Load the dataset for AutoML.

    Args:
        reduced (bool): Whether to exclude the datasets that have been used to train TabPFN. Default is False.

    Returns:
        dict[str, Tuple[pd.DataFrame, pd.DataFrame]]: A dictionary containing the loaded dataset.

    """
    return data_handler.load_dataset(reduced)


def load_dummy_dataset() -> TabminiDataset:
    """
    Load the dummy dataset for AutoML.

    Returns:
        dict[str, Tuple[pd.DataFrame, pd.DataFrame]]: A dictionary containing the loaded dataset.

    """
    return data_handler.load_dummy_dataset()



def compare(
        method_name: str,
        estimator: BaseEstimator,
        dataset: TabminiDataset,
        working_directory: Path,
        scoring_method: str = "roc_auc",
        cv: int = 3,
        time_limit: int = 3600,
        methods: set[str] = estimators.get_available_methods(),
        device: str = "cpu",
        n_jobs: int = -1,
        kwargs_per_classifier: dict[str, dict] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare the performance of the estimator on the given dataset against all predefined estimators.

    There are many side effects to this function.
    One of them is that: If the model you are trying to compare does any sort of hyperparameter
    optimization, the results will be saved in the working directory.

    Args:
        method_name: The name of the estimator to be compared.
        estimator: The estimator to be compared.
        dataset: The dataset to be used for comparison.
        working_directory: The working directory to save the results.
        scoring_method: The scoring method to be used. Default is "roc_auc".
        cv: The cross-validation strategy to be used. Default is 3.
        time_limit: The time limit for the comparison in seconds. Default is 3600.
        methods: The methods to be compared. Default is all available methods.
        device: The device to be used. Default is "cpu".
        n_jobs: The number of jobs to be used. Default is -1. If -1, all CPUs are used.
        kwargs_per_classifier: The keyword arguments for each classifier.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - A DataFrame containing the test scores.
            - A DataFrame containing the train scores.
    """
    print(f"Evaluating {method_name} against {methods}...")
    # The result of scorer.compare is a dictionary shaped like
    # dataset_name: {method_name: (test_score, train_score)}
    compare_results: dict[str, dict[str, tuple[float, float]]] = scorer.compare(
        method_name,
        estimator,
        dataset,
        working_directory,
        scoring_method=scoring_method,
        cv=cv,
        time_limit=time_limit,
        methods=methods,
        device=device,
        n_jobs=n_jobs,
        kwargs_per_classifier=kwargs_per_classifier,
    )

    def extract_from(results, index: Literal[0, 1]) -> dict[str, dict[str, float]]:
        return {
            dataset_name: {
                method: scores[index] for method, scores in method_results.items()
            } for dataset_name, method_results in results.items()
        }

    test_scores = extract_from(compare_results, 0)
    train_scores = extract_from(compare_results, 1)

    # save results
    return pd.DataFrame(test_scores).T, pd.DataFrame(train_scores).T


def get_meta_feature_analysis(
        dataset: TabminiDataset,
        results_wide: pd.DataFrame,
        name_of_method_to_compare: str,
        correlation_method: Literal["pearson", "kendall", "spearman"] = "spearman"
) -> pd.DataFrame:
    """
    Analyze the meta-features of the dataset.

    Args:
        dataset: The dataset to be used for comparison.
        results_wide: The results of the comparison. A DataFrame containing the results of all the methods on the
        datasets.
        name_of_method_to_compare: The name of the method to be compared.
        correlation_method: The correlation method to be used. Default is "spearman".

    Returns:
        pd.DataFrame: The analysis of the meta-features.

    """
    return meta_feature.get_meta_feature_analysis(dataset, results_wide, name_of_method_to_compare, correlation_method)
