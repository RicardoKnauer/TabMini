from typing import Callable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, cross_validate

from tabmini.analysis.timer import execute_with_timelimit
from tabmini.estimators import get_estimators_with, is_threadsafe, is_sklearn_compatible, is_valid_time_limit


def get_test_score(estimator: BaseEstimator, X, y, cv=3, scoring="roc_auc") -> float:
    return cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1).mean()


def _get_train_and_test_score_per_estimator_on(
        X, y, working_dir, cv, scoring, time_limit, methods, device, n_jobs, kwargs_per_classifier
) -> dict[str, tuple[float, float]]:
    # lower and strip the method names
    methods = {method.lower().strip() for method in methods}

    results = {}
    for method, estimator in get_estimators_with(working_dir, time_limit, device, kwargs_per_classifier).items():
        if method not in methods:
            continue
        
        if not is_valid_time_limit(method, time_limit):
            print(f"Skipping {method} due to time limit")
            results[method] = (0.0, 0.0)
            continue

        n_jobs_for_method = n_jobs if is_threadsafe(method) else 1

        print(f"Testing {method} with {n_jobs_for_method} job(s)")

        try:
            executor: Callable[[], dict] = lambda: cross_validate(
                estimator,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs_for_method,
                return_train_score=True
            )

            # TODO: This was implemented but does not play nicely with the threaded nature of most of the estimators
            # used for this benchmark. We will consider timing the execution instead and adding runtime to the results
            # scores = execute_with_timelimit(executor, time_limit)

            scores = executor()
            
        except Exception as e:
            print(f"Failed to test {method}. Reason: {e}")
            print("Score for this method will be set to 0.0")
            results[method] = (0.0, 0.0)
            continue

        results[method] = (scores["test_score"].mean(), scores["train_score"].mean())

    return results


def compare(
        method_name,
        estimator,
        dataset,
        working_directory,
        scoring_method,
        cv,
        time_limit,
        methods,
        device,
        kwargs_per_classifier,
        n_jobs=-1
):
    if not (is_sklearn_compatible(estimator)):
        raise ValueError("Estimator needs to implement a fit and predict functions.")

    results = {}
    for dataset_name, (X, y) in dataset.items():
        print(f"Comparing {method_name} on {dataset_name}")

        # We need to make sure our target is in the right format, that means [0, 1].
        y = (y == y.max()).astype(int)

        # Check how our estimators perform on the dataset
        # result is a dictionary shaped like {method_name: (test_score, train_score)}
        results_for_dataset: dict[str, tuple[float, float]] = _get_train_and_test_score_per_estimator_on(
            X,
            y,
            working_dir=working_directory,
            cv=cv,
            scoring=scoring_method,
            time_limit=time_limit,
            methods=methods,
            device=device,
            n_jobs=n_jobs,
            kwargs_per_classifier=kwargs_per_classifier
        )

        # Compare the given estimator with the predefined ones
        print(f"Testing {method_name}")
        scores = cross_validate(
            estimator,
            X,
            y,
            cv=cv,
            scoring=scoring_method,
            n_jobs=n_jobs,
            return_train_score=True,
            return_estimator=True
        )

        # Add our subjects score to the results
        results_for_dataset[method_name] = (scores["test_score"].mean(), scores["train_score"].mean())

        # And get ready to return
        results[dataset_name] = results_for_dataset

        # Addendum: Save the best hyperparameters if available
        i_best_estimator = max(enumerate(scores["test_score"]), key=lambda x: x[1])[0]
        best_estimator = scores["estimator"][i_best_estimator]

        if not hasattr(best_estimator, "best_params_"):
            continue

        best_params = best_estimator.best_params_
        filename = "best_params_" + method_name.replace(" ", "").strip() + ".csv"
        pd.DataFrame(best_params, index=[0]).to_csv(working_directory / filename, index=False)

    return results
