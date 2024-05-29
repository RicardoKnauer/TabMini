from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class AutoGluon(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            path: Path,
            time_limit: int = 3600,
            presets=None,
            seed: int = 0,
            kwargs: dict = {}
    ):
        # set the type of the predictor to be a regressor

        if presets is None:
            presets = ["best_quality"]

        self.path = path
        self.feature_names = []
        self.target_name = "label"
        self.time_limit = time_limit
        self.presets = presets
        self.seed = seed
        self.kwargs = kwargs

        self.predictor = TabularPredictor(
            label=self.target_name,
            path=str(path.absolute()),
            verbosity=0,
            log_to_file=True,
            eval_metric="roc_auc",
            **kwargs
        )

        # specify that this is a binary classifier
        self.n_classes_ = 2
        self.classes_ = [0, 1]

    def fit(self, X, y) -> 'AutoGluon':
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

        # For AutoGluon we need data and target to be in the same DataFrame
        self.feature_names = [f"{i}" for i in range(X.shape[1])]

        feature_names_train = self.feature_names.copy()
        feature_names_train.insert(0, self.target_name)

        train_data = pd.DataFrame(
            np.hstack((np.array(y)[:, np.newaxis], X)),
            columns=feature_names_train
        )

        self.predictor = self.predictor.fit(train_data, presets=self.presets, time_limit=self.time_limit)
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

        eval_data = pd.DataFrame(X, columns=self.feature_names, dtype=object)

        probability_positive_class = self.predictor.predict_proba(eval_data, as_pandas=False, as_multiclass=False)
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
