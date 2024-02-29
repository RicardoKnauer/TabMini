from typing import Literal
import numpy as np

import pandas as pd
from pymfe.mfe import MFE


def get_meta_feature_analysis(
        dataset: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
        results_wide: pd.DataFrame,
        method_to_compare: str,
        correlation_method: Literal["pearson", "kendall", "spearman"] = "spearman"
) -> pd.DataFrame:
    """
    Analyze the meta features of the datasets.
    :param dataset: The dataset that was used for comparing the methods.
    :param results_wide: DataFrame with the results of all the methods on given dataset.
    :param method_to_compare: The method to be compared with the others.
    :param correlation_method: The correlation method to be used. Default is "spearman".
    :return: DataFrame with the results of the analysis. The columns are: method, coeffs, nr_inst, inst_to_attr, EPV.
    """

    meta_features_of_dataset = _get_meta_features_of(dataset)

    # We need to drop the first column of the results_wide DataFrame, which contains the dataset names
    results_wide = results_wide.drop(columns=results_wide.columns[0])

    correlations = _calculate_correlation_of_features(
        meta_features_of_dataset,
        results_wide,
        correlation_method,
        method_to_compare
    )

    return correlations


def _get_meta_features_of(dataset: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    meta_features = []
    column_names = None

    for _, (X, y) in dataset:
        mfe = MFE(groups="all", summary="all", num_cv_folds=3, random_state=42)
        X = np.array(X)
        y = np.array(y)
        ft = mfe.fit(X, y).extract(suppress_warnings=True)

        if column_names is None:
            column_names = ft[0] + ["EPV"]

        row = ft[1] + [min(sum(y), len(y) - sum(y))]
        meta_features.append(row)

    # Create a DataFrame with the meta-features
    results_meta_features = pd.DataFrame(meta_features, columns=column_names)
    results_meta_features["EPV"] = results_meta_features["EPV"] / results_meta_features["nr_attr"]

    return results_meta_features


def _calculate_correlation_of_features(
        meta_features: pd.DataFrame,
        dataset: pd.DataFrame,
        correlation_method: Literal["pearson", "kendall", "spearman"],
        method_to_compare: str
) -> pd.DataFrame:
    results = {}

    for method in dataset.columns:
        results_meta_features_new = meta_features.copy()
        try:
            dataset[method]
        except KeyError:
            continue
        results_meta_features_new[f"diff_{method}"] = dataset[method] - dataset[method_to_compare]
        coeffs = results_meta_features_new.corr(correlation_method)[f"diff_{method}"]
        results[method] = {
            'method': method,
            'nr_inst': coeffs['nr_inst'],
            'inst_to_attr': coeffs['inst_to_attr'],
            'EPV': coeffs['EPV']
        }
        # largest correlation methods
        largest = round(coeffs.abs().nlargest(11), 2)
        # add the largest correlation methods to the results
        for m, value in largest.items():
            results[method][m] = value

    df = pd.DataFrame(results).T

    # replace empty values with NaN
    df = df.replace("", "NaN")

    return df
