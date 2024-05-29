from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tabpfn import TabPFNClassifier


class TabPFN(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible estimator that uses TabPFN to fit and predict data."""
    def __init__(
            self,
            path: Path,
            time_limit: int = 3600,
            n_ensemble_configurations: int = 32,
            device: str = "cpu",
            seed: int = 0,
            kwargs: dict = {}
    ):
        self.predictor = TabPFNClassifier(
            base_path=path,
            N_ensemble_configurations=n_ensemble_configurations,
            subsample_features=True,
            device=device,
            seed=seed,
            **kwargs
        )
        self.path = path
        self.time_limit = time_limit
        self.n_ensemble_configurations = n_ensemble_configurations
        self.device = device
        self.seed = seed
        self.kwargs = kwargs

        # specify that this is a binary classifier
        self.n_classes_ = 2
        self.classes_ = [0, 1]

    def fit(self, X, y) -> 'TabPFN':
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

        # Get the probability of the positive class
        probability_positive_class = self.predictor.predict(X)
        probability_positive_class_scaled = (probability_positive_class - probability_positive_class.min()) / (
                probability_positive_class.max() - probability_positive_class.min())

        # Create a 2D array with probabilities of both classes
        return np.vstack([1 - probability_positive_class_scaled, probability_positive_class_scaled]).T

    def predict(self, X) -> np.ndarray:
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

        # (we know this is a numpy array, because of nograd)
        return self.predictor.predict(X)  # type: ignore

    def decision_function(self, X):
        # Get the probabilities from predict_proba
        proba = self.predict_proba(X)

        # Calculate the log of ratios for binary classification
        # Add a small constant to both the numerator and the denominator
        decision = np.log((proba[:, 1] + 1e-10) / (proba[:, 0] + 1e-10))

        return decision
