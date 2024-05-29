from pathlib import Path

import numpy as np
from hyperfast import HyperFastClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class HyperFast(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible estimator that uses HyperFast to fit and predict data."""
    def __init__(
            self,
            path: Path,
            time_limit: int = 3600,
            n_ensemble_configurations: int = 32,
            device: str = "cpu",
            seed: int = 0,
            kwargs: dict = {}
    ):
        self.predictor = HyperFastClassifier(
            custom_path=str(path.absolute()),
            device=device,
            n_ensemble=n_ensemble_configurations,
            optimization=None,
            **kwargs
        )
        self.path = path
        self.time_limit = time_limit
        self.feature_names = []
        self.n_ensemble_configurations = n_ensemble_configurations
        self.device = device
        self.seed = seed
        self.kwargs = kwargs

        # specify that this is a binary classifier
        self.n_classes_ = 2
        self.classes_ = [0, 1]

    def fit(self, X, y) -> 'HyperFast':
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

        self.predictor = self.predictor.fit(X, y)
        self.is_fitted_ = True

        return self

    # NOTE: Predict function has to come first in this file - otherwise, when trying to calculate a score,
    # SKLearn will assume this is a regressor instead of a classifier.

    def predict_proba(self, X) -> np.ndarray:
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        probability_positive_class = self.predictor.predict(X)
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
