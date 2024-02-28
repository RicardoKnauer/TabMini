from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import automl

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
LAMBDA_OPTIONS = [0.5, 0.01]  #[0.5, 0.01, 0.002, 0.0004]
param_grid = [
    {
        "classify__penalty": REGULARIZATION_OPTIONS,
        "classify__C": LAMBDA_OPTIONS,
    }
]

# inner cross-validation for logistic regression
estimator = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="neg_log_loss")

# load dataset
dataset: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = automl.load_dataset(reduced=True)

# compare with the predefined methods
results: dict[str, dict[str, tuple[float, float]]] = automl.compare(
    method_name,
    estimator,
    dataset,
    working_directory,
    scoring_method="roc_auc",
    cv=3,
    time_limit=60,
    device="cpu"
)

print(results)

# save results
results_df = pd.DataFrame(results).T
results_df.to_csv(working_directory / "results.csv", index=False)

# save the best hyperparameters to a csv file
best_params = estimator.best_params_
filename = "best_params_" + method_name.replace(" ", "").strip() + ".csv"
pd.DataFrame(best_params, index=[0]).to_csv(working_directory / filename, index=False)

# analyze meta features
loaded_results = pd.read_csv(working_directory / "results.csv")
meta_features_analysis = automl.get_meta_feature_analysis(
    dataset,
    loaded_results,
    method_name,
    correlation_method="spearman"
)

meta_features_analysis.to_csv(working_directory / "meta_features_analysis.csv", index=False)


