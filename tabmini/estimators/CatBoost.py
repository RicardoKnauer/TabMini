from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array


class CatBoost(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible estimator that uses CatBoost to fit and predict data."""

    def __init__(
            self,
            path: Path,
            time_limit: int = 3600,
            device: str = "cpu",
            seed: int = 0,
            kwargs: dict = {}
    ):
        self.path = path
        self.time_limit = time_limit
        self.device = device
        self.kwargs = kwargs
        self.seed = seed

        # specify that this is a binary classifier
        self.n_classes_ = 2
        self.classes_ = [0, 1]

        self.catboost = CatBoostClassifier(
            iterations=2,
            depth=2,
            learning_rate=1,
            loss_function='Logloss',
            eval_metric='AUC',
            task_type='GPU' if self.device == 'cuda' else 'CPU',
            devices='0',
            random_seed=self.seed,
            **self.kwargs
        )

    def fit(self, X, y) -> 'CatBoost':
        """

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)

        self.feature_names = [f"f{i}" for i in range(X.shape[1])]
        self.feature_names.insert(0, "target")

        self.model = self.catboost.fit(X, y)

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse=True)

        probability_positive_class = self.model.predict(X)
        probability_positive_class_scaled = (probability_positive_class - probability_positive_class.min()) / (
                probability_positive_class.max() - probability_positive_class.min() + 1e-10)

        # Create a 2D array with probabilities of both classes
        return np.vstack([1 - probability_positive_class_scaled, probability_positive_class_scaled]).T

    def decision_function(self, X):
        # Get the probabilities from predict_proba
        proba = self.predict_proba(X)

        # Calculate the log of ratios for binary classification
        decision = np.log((proba[:, 1] + 1e-10) / (proba[:, 0] + 1e-10))

        return decision
