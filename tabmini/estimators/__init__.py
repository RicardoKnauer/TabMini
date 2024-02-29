from pathlib import Path
from typing import FrozenSet, Callable

from sklearn.base import BaseEstimator

from tabmini.estimators.AutoGluon import AutoGluon
from tabmini.estimators.AutoPrognosis import AutoPrognosis
from tabmini.estimators.HyperFast import HyperFast
from tabmini.estimators.TabPFN import TabPFN

# These are the methods that are not threadsafe. They will be run with n_jobs=1.
# If your method is not threadsafe, add it to this set.
_NON_THREADSAFE_METHODS = frozenset({
    "autogluon",
})

# This is where the scikit-learn compatible estimators are registered for use as a classifier. Every estimator
# in this list will always be instantiated on startup, even if they are not selected for generating a baseline.
_ESTIMATORS: dict[str, Callable[[Path, int, str, dict], BaseEstimator]] = {
    "AutoGluon": lambda base_path, time_limit, _, kwargs: AutoGluon(
        path=base_path / "autogluon",
        time_limit=time_limit,
        kwargs=kwargs
    ),
    "AutoPrognosis": lambda base_path, time_limit, _, kwargs: AutoPrognosis(
        path=base_path / "autoprognosis",
        time_limit=time_limit,
        kwargs=kwargs
    ),
    "TabPFN": lambda base_path, time_limit, device, kwargs: TabPFN(
        path=base_path / "tabpfn",
        time_limit=time_limit,
        device=device,
        kwargs=kwargs
    ),
    "HyperFast": lambda base_path, time_limit, device, kwargs: HyperFast(
        path=base_path / "hyperfast",
        time_limit=time_limit,
        device=device,
        kwargs=kwargs
    ),
}


def is_threadsafe(method_name: str) -> bool:
    return method_name.lower().strip() not in [m.lower().strip() for m in _NON_THREADSAFE_METHODS]


def is_sklearn_compatible(estimator: BaseEstimator) -> bool:
    return hasattr(estimator, "fit") and hasattr(estimator, "predict_proba") and hasattr(estimator, "decision_function")


def get_available_methods() -> frozenset[str]:
    return frozenset(_ESTIMATORS.keys())


def get_estimators_with(
        base_path: Path,
        time_limit: int,
        device: str,
        kwargs_per_classifier: dict[str, dict]
) -> dict[str, BaseEstimator]:
    if kwargs_per_classifier is None:
        kwargs_per_classifier = {}

    # Lower and strip the method names in the kwargs dictionary
    kwargs_per_classifier = {
        method_name.lower().strip(): values
        for method_name, values in kwargs_per_classifier.items()
    }

    # Instantiate the estimators
    estimators = {}
    for estimator_name, estimator_constructor in _ESTIMATORS.items():
        estimator_name = estimator_name.lower().strip()
        kwargs = kwargs_per_classifier.get(estimator_name, {})

        estimators[estimator_name] = estimator_constructor(base_path, time_limit, device, kwargs)

    return estimators
