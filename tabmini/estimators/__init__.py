from pathlib import Path
from typing import FrozenSet

from sklearn.base import BaseEstimator

from tabmini.estimators.AutoGluon import AutoGluon
from tabmini.estimators.AutoPrognosis import AutoPrognosis
from tabmini.estimators.HyperFast import HyperFast
from tabmini.estimators.TabPFN import TabPFN


def get_available_methods() -> frozenset[str]:
    return frozenset({
        "AutoGluon",
        "AutoPrognosis",
        "TabPFN",
        "HyperFast"
    })


def get_estimators_with(
        base_path: Path,
        time_limit: int,
        device: str = "cpu",
        kwargs_per_classifier: dict[str, dict] = {}
) -> dict[str, BaseEstimator]:
    # lower and strip the kwargs keys
    kwargs_per_classifier = {
        method_name: {key.lower().strip(): value for key, value in kwargs.items()}
        for method_name, kwargs in kwargs_per_classifier.items()
    }

    estimators = {
        "AutoGluon": AutoGluon(
            path=base_path / "autogluon",
            time_limit=time_limit,
            kwargs=kwargs_per_classifier.get("autogluon", {})
        ),
        "AutoPrognosis": AutoPrognosis(
            path=base_path / "autoprognosis",
            time_limit=time_limit,
            kwargs=kwargs_per_classifier.get("autoprognosis", {})
        ),
        "TabPFN": TabPFN(
            path=base_path / "tabpfn",
            time_limit=time_limit,
            device=device,
            kwargs=kwargs_per_classifier.get("tabpfn", {})
        ),
        "HyperFast": HyperFast(
            path=base_path / "hyperfast",
            time_limit=time_limit,
            device=device,
            kwargs=kwargs_per_classifier.get("hyperfast", {})
        ),
    }

    return estimators
