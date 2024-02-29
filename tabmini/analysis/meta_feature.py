from typing import Literal
import numpy as np

import pandas as pd
from pymfe.mfe import MFE

from tabmini.types import TabminiDataset


def get_meta_feature_analysis(
        dataset: TabminiDataset,
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

    correlations = _calculate_correlation_of_features(
        meta_features_of_dataset,
        results_wide,
        correlation_method,
        method_to_compare
    )

    return correlations


def _get_meta_features_of(dataset: TabminiDataset) -> pd.DataFrame:
    meta_features = []
    column_names = None

    for _, (X, y) in dataset.items():
        mfe = MFE(groups="all", summary="all", num_cv_folds=3, random_state=42)
        X = np.array(X)
        y = np.array(y)
        ft = mfe.fit(X, y).extract(suppress_warnings=True)

        if column_names is None:
            column_names = ft[0] + ["EPV"]

        row = ft[1] + [min(sum(y), len(y) - sum(y))]
        meta_features.append(row)

    # Create a DataFrame with the meta-features
    results_meta_features = pd.DataFrame(meta_features, columns=column_names, index=list(dataset.keys()))
    results_meta_features["EPV"] = results_meta_features["EPV"] / results_meta_features["nr_attr"]

    return results_meta_features


def _calculate_correlation_of_features(
        meta_features: pd.DataFrame,
        comparison_results: pd.DataFrame,
        correlation_method: Literal["pearson", "kendall", "spearman"],
        method_to_compare: str
) -> pd.DataFrame:
    results = {}

    for method in comparison_results.columns:

        results_meta_features_new = meta_features.copy()

        try:
            comparison_results[method]
        except KeyError:
            continue

        diff = comparison_results[method] - comparison_results[method_to_compare]
        results_meta_features_new[f"diff_{method}"] = diff
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
    df = df.replace("", "NaN")

    return df
