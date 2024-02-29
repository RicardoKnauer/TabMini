from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import tabmini

working_directory = Path.cwd() / "workdir"

# define pipeline
method_name = "Logistic Regression"
pipe = Pipeline(
    [
        ("scaling", MinMaxScaler()),
        ("classify", LogisticRegression(random_state=42)),
    ]
)

# define hyperparameters
REGULARIZATION_OPTIONS = ["l2"]
LAMBDA_OPTIONS = [0.5, 0.01, 0.002, 0.0004]
param_grid = [
    {
        "classify__penalty": REGULARIZATION_OPTIONS,
        "classify__C": LAMBDA_OPTIONS,
    }
]

# inner cross-validation for logistic regression
estimator = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="neg_log_loss")

# load dataset
dataset: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = tabmini.load_dummy_dataset()

# compare with the predefined methods
train_scores, test_scores = tabmini.compare(
    method_name,
    estimator,
    dataset,
    working_directory,
    scoring_method="roc_auc",
    cv=3,
    methods={"hyperfast", "tabpfn"},
    time_limit=10,
    device="cpu"
)

test_scores.to_csv(working_directory / "results.csv", index_label="PMLB dataset")

# analyze meta features
meta_features_analysis = tabmini.get_meta_feature_analysis(
    dataset,
    test_scores,
    method_name,
    correlation_method="spearman"
)

meta_features_analysis.to_csv(working_directory / "meta_features_analysis.csv", index=False)
