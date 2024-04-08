from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import tabmini
from tabmini.types import TabminiDataset

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
estimator = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="neg_log_loss", n_jobs=1)

# load dataset
dataset: TabminiDataset = tabmini.load_dataset()

# define a set of time-limits
time_limits = [15 * 60, 30 * 60, 45 * 60]

for time_limit in time_limits:
    # compare with the predefined methods
    train_scores, test_scores = tabmini.compare(
        method_name,
        estimator,
        dataset,
        working_directory,
        scoring_method="roc_auc",
        cv=2,
        #methods={"XGBoost", "LightGBM"},
        time_limit=time_limit,
        device="cpu",
        n_jobs=-1,
    )

    test_scores.to_csv(working_directory / f"results_{time_limits}.csv", index_label="PMLB dataset")

    # analyze meta features
    meta_features_analysis = tabmini.get_meta_feature_analysis(
        dataset,
        test_scores,
        method_name,
        correlation_method="spearman"
    )

    meta_features_analysis.to_csv(working_directory / f"meta_features_analysis_{time_limit}.csv", index=False)
