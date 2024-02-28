from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, train_test_split

from automl.estimators import get_estimators_with


def get_trained_estimator_and_train_score(estimator: BaseEstimator, X, y) -> tuple[BaseEstimator, float]:
    if not (hasattr(estimator, "fit") and hasattr(estimator, "predict_proba") and hasattr(estimator, "decision_function")):
        raise ValueError("Estimator needs to implement a fit and predict functions.")

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    trained = estimator.fit(X_train, y_train)
    return trained, _get_test_score(trained, X_train, y_train, cv=2)


def _get_test_score(estimator: BaseEstimator, X, y, cv=3, scoring="roc_auc") -> float:
    return cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1).mean()


def _get_train_and_test_score_per_estimators_on(
        X, y, working_dir, cv, scoring, time_limit, methods, device, kwargs_per_classifier
) -> dict[str, tuple[float, float]]:
    estimators = get_estimators_with(working_dir, time_limit, device, kwargs_per_classifier)
    results = {}

    # lower and strip the method names
    methods = {method.lower().strip() for method in methods}

    for method_name, estimator in estimators.items():
        if method_name.lower().strip() not in methods:
            print(f"Skipping {method_name}, not in list of methods to be tested ({methods})")
            continue

        trained_estimator, training_score = get_trained_estimator_and_train_score(estimator, X, y)
        test_score = _get_test_score(trained_estimator, X, y, cv=cv, scoring=scoring)
        results[method_name] = (test_score, training_score)

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
        kwargs_per_classifier
):
    results = {}
    for dataset_name, (X, y) in dataset:
        print(f"Comparing {estimator} on {dataset_name}")

        # Check how our estimators perform on the dataset
        results_for_dataset: dict[str, tuple[float, float]] = _get_train_and_test_score_per_estimators_on(
            X,
            y,
            working_dir=working_directory,
            cv=cv,
            scoring=scoring_method,
            time_limit=time_limit,
            methods=methods,
            device=device,
            kwargs_per_classifier=kwargs_per_classifier
        )

        # Compare the given estimator with the predefined ones
        trained_estimator, training_score = get_trained_estimator_and_train_score(estimator, X, y)
        test_score = _get_test_score(trained_estimator, X, y, cv, scoring_method)

        # Add our subjects score to the results
        results_for_dataset[method_name] = (test_score, training_score)

        # And get ready to return
        results[dataset_name] = results_for_dataset

    return results
