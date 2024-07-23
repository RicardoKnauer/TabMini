from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import tabmini
from tabmini.types import TabminiDataset

from datetime import datetime
import os

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
estimator = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="neg_log_loss", n_jobs=-1)

# load dataset
dataset: TabminiDataset = tabmini.load_dataset()

def run_experiment(framework, output_path, time_limit):
    # Load dataset
    print(framework)

    print(f'-------------------Time limit: {time_limit}-----------------------------------')
    test_scores, train_scores = tabmini.compare(
        method_name,
        estimator,
        dataset,
        working_directory,
        scoring_method="roc_auc",
        cv=3,
        time_limit=time_limit,
        framework=framework,
        device="cpu",
        n_jobs=-1,  # Time Limit does not play nice with threads
    )

    test_scores.to_csv(output_path / f"results_{time_limit}.csv", index_label="PMLB dataset")

    # analyze meta features
    meta_features_analysis = tabmini.get_meta_feature_analysis(
        dataset,
        test_scores,
        method_name,
        correlation_method="spearman"
    )

    meta_features_analysis.to_csv(output_path /f"meta_features_analysis_{time_limit}.csv", index=False)



if __name__ == "__main__":
    import sys
    framework   = sys.argv[1]
    output_path = Path(sys.argv[2])
    time_limit  = int(sys.argv[3])
    run_experiment(framework, output_path, time_limit)