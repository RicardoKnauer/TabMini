from pathlib import Path

import numpy as np
import pandas as pd
from autoprognosis.explorers.core.defaults import default_classifiers_names
from autoprognosis.studies.classifiers import ClassifierStudy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class AutoPrognosis(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible estimator that uses AutoPrognosis to fit and predict data."""

    def __init__(
            self,
            path: Path,
            time_limit: int = 3600,
            classifier_names: list[str] = default_classifiers_names,
            seed: int = 0,
            kwargs: dict = {}
    ):
        self.path = path
        self.feature_names = []
        self.time_limit = time_limit
        self.classifier_names = classifier_names
        self.seed = seed
        self.kwargs = kwargs

        # specify that this is a binary classifier
        self.n_classes_ = 2
        self.classes_ = [0, 1]

        self.target_name = "target"

    def fit(self, X, y) -> 'AutoPrognosis':
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
        self.feature_names.insert(0, self.target_name)

        train_data = pd.DataFrame(
            np.hstack((np.array(y)[:, np.newaxis], X)),
            columns=self.feature_names
        )
        study = ClassifierStudy(
            train_data,
            target=self.target_name,
            workspace=self.path,
            score_threshold=0.3,
            classifiers=self.classifier_names,
            timeout=int(self.time_limit / len(self.classifier_names)),
            random_state=self.seed,
            **self.kwargs
        )
        self.study_ = study.fit()

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
        check_is_fitted(self, ['study_'])

        probability_positive_class = self.study_.predict_proba(X).iloc[:, 1]
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
