from pathlib import Path
from typing import Callable

from sklearn.base import BaseEstimator

from tabmini.estimators.AutoGluon import AutoGluon
from tabmini.estimators.AutoPrognosis import AutoPrognosis
from tabmini.estimators.HyperFast import HyperFast
from tabmini.estimators.LightGBM import LightGBM
from tabmini.estimators.TabPFN import TabPFN
from tabmini.estimators.XGBoost import XGBoost
from tabmini.estimators.CatBoost import CatBoost

_SEED = 42

# These are the methods that are not threadsafe. They will be run with n_jobs=1.
# If your method is not threadsafe, add it to this set.
_NON_THREADSAFE_METHODS = frozenset({
    "autogluon",
})

# These are the minimum time limits for each method. If the time limit is less than this value, 
# the method will be skipped.
_MINIMUM_TIME_LIMITS = {
    "autogluon": 10,
}

# This is where the scikit-learn compatible estimators are registered for use as a classifier. Every estimator
# in this list will always be instantiated on startup, even if they are not selected for generating a baseline.
_ESTIMATORS: dict[str, Callable[[Path, int, str, int, dict], BaseEstimator]] = {
    "AutoGluon": lambda base_path, time_limit, _, seed, kwargs: AutoGluon(
        path=base_path / "autogluon",
        time_limit=time_limit,
        seed=seed,
        kwargs=kwargs
    ),
    "AutoPrognosis": lambda base_path, time_limit, _, seed, kwargs: AutoPrognosis(
        path=base_path / "autoprognosis",
        time_limit=time_limit,
        seed=seed,
        kwargs=kwargs
    ),
    "TabPFN": lambda base_path, time_limit, device, seed, kwargs: TabPFN(
        path=base_path / "tabpfn",
        time_limit=time_limit,
        seed=seed,
        device=device,
        kwargs=kwargs
    ),
    "HyperFast": lambda base_path, time_limit, device, seed, kwargs: HyperFast(
        path=base_path / "hyperfast",
        time_limit=time_limit,
        seed=seed,
        device=device,
        kwargs=kwargs
    ),
    "LightGBM": lambda base_path, time_limit, _, seed, kwargs: LightGBM(
        path=base_path / "lightgbm",
        time_limit=time_limit,
        seed=seed,
        kwargs=kwargs
    ),
    "XGBoost": lambda base_path, time_limit, _, seed, kwargs: XGBoost(
        path=base_path / "xgboost",
        time_limit=time_limit,
        seed=seed,
        kwargs=kwargs
    ),
    "CatBoost": lambda base_path, time_limit, device, seed, kwargs: CatBoost(
        path=base_path / "catboost",
        time_limit=time_limit,
        seed=seed,
        device=device,
        kwargs=kwargs
    ),

}


def is_valid_time_limit(method_name: str, time_limit: int) -> bool:
    return time_limit >= _MINIMUM_TIME_LIMITS.get(method_name, 0)


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
        kwargs_per_classifier: dict[str, dict],
        seed: int = _SEED
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

        estimators[estimator_name] = estimator_constructor(base_path, time_limit, device, seed, kwargs)

    return estimators
