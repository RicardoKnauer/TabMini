from pathlib import Path
from typing import Tuple, Literal

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

from tabmini import analysis, estimators
from tabmini.data import data_handler
from tabmini.analysis import meta_feature, auto_ml


def load_dataset(reduced: bool = False) -> dict[str, tuple[DataFrame, DataFrame]]:
    """
    Load the dataset for AutoML.

    Args:
        reduced (bool): Whether to exclude the datasets that have been used to train TabPFN. Default is False.

    Returns:
        dict[str, Tuple[pd.DataFrame, pd.DataFrame]]: A dictionary containing the loaded dataset.

    """
    return data_handler.load_dataset(reduced)


def load_dummy_dataset() -> dict[str, tuple[DataFrame, DataFrame]]:
    """
    Load the dataset for AutoML.

    Returns:
        dict[str, Tuple[pd.DataFrame, pd.DataFrame]]: A dictionary containing the loaded dataset.

    """
    return data_handler.load_dummy_dataset()


def save_dataset(dataset: dict[str, Tuple[pd.DataFrame, pd.DataFrame]], path: Path = Path("datasets")) -> None:
    """
    Save the dataset for AutoML.

    Args:
        dataset (dict[str, pd.DataFrame]): A dictionary containing the loaded dataset.
        path (Path): The path to save the datasets.

    """
    data_handler.save_dataset(dataset, path)


def get_trained_estimator_and_score(
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray
) -> tuple[BaseEstimator, float]:
    """
    Train the estimator and return the training score.

    Args:
        estimator: The estimator to be trained.
        X: The input features.
        y: The target variable.

    Returns:
        tuple[BaseEstimator, float]: A tuple containing:
            - The trained estimator.
            - The training score.

    """
    return auto_ml.get_trained_estimator_and_train_score(estimator, X, y)


def get_test_score(
        estimator: BaseEstimator,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.DataFrame,
        cv: int = 3,
        scoring: str = "roc_auc"
) -> float:
    """
    Score the estimator using cross-validation.

    Args:
        estimator: The estimator to be scored.
        X: The input features.
        y: The target variable.
        cv: The cross-validation strategy.
        scoring: The scoring metric.

    """
    return cross_val_score(estimator, X, y, cv=cv, scoring=scoring).mean()


def compare(
        method_name: str,
        estimator: BaseEstimator,
        dataset: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
        working_directory: Path,
        scoring_method: str = "roc_auc",
        cv: int = 3,
        time_limit: int = 3600,
        methods: set[str] = estimators.get_available_methods(),
        device: str = "cpu",
        kwargs_per_classifier: dict[str, dict] = {}

) -> dict[str, dict[str, tuple[float, float]]]:
    """
    Compare the performance of the estimator on the given dataset against all predefined estimators.

    Args:
        method_name: The name of the estimator to be compared.
        estimator: The estimator to be compared.
        dataset: The dataset to be used for comparison.
        working_directory: The working directory to save the results.
        scoring_method: The scoring method to be used. Default is "roc_auc".
        cv: The cross-validation strategy to be used. Default is 3.
        time_limit: The time limit for the comparison. Default is 3600.
        methods: The methods to be compared. Default is all available methods.
        device: The device to be used. Default is "cpu".
        kwargs_per_classifier: The keyword arguments for each classifier.

    Returns:
        dict[str, dict[str, tuple[float, float]]]: A dictionary containing:
            - The dataset name as the key.
            - A dictionary containing the estimator name as the key and a tuple containing the training and test scores
              as the value.

    """
    return auto_ml.compare(
        method_name,
        estimator,
        dataset,
        working_directory,
        scoring_method=scoring_method,
        cv=cv,
        time_limit=time_limit,
        methods=methods,
        device=device,
        kwargs_per_classifier=kwargs_per_classifier
    )


def get_meta_feature_analysis(
        dataset: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
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
