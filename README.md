# TabMini

We introduce `TabMini`, the first tabular benchmark suite specifically for the low-data regime with 44 binary 
classification datasets, and use our suite to compare state-of-the-art machine learning methods, 
i.e., automated machine learning (AutoML) frameworks and off-the-shelf deep neural networks, 
against logistic regression.

### Installation/Development

This project was developed using a devcontainer, which is defined in the `.devcontainer` folder.

For development, a `requirements.txt` is included to be installed with pip.

To install the package as a Python package, you can use the following command:

```bash
pip install ./tabmini
```

## Usage - Package

The `TabMini` benchmark suite is designed to be imported into your Python project, however, it can also be used as a
standalone package. The package is designed to be used in the following way:

```python
from pathlib import Path

from yourpackage import YourEstimator

import tabmini

# Load the dataset
# Tabmini also provides a dummy dataset for testing purposes, you can load it with tabmini.load_dummy_dataset() 
# If reduced is set to True, the dataset will exclude all the data that has been used to meta-train TabPFN
dataset = tabmini.load_dataset(reduced=False)

# Prepare the estimator you want to benchmark against the other estimators
estimator = YourEstimator()

# Perform the comparison
train_results, test_results = tabmini.compare(
    "MyEstimator",
    estimator,
    dataset,
    working_directory=Path.cwd() / "results",
    scoring_method="roc_auc",
    cv=3,
    time_limit=3600,
    device="cpu"
)

# Generate the meta-feature analysis
meta_features = tabmini.get_meta_feature_analysis(dataset, test_results, "MyEstimator", correlation_method="spearman")

# Save the results and meta-feature analysis to a CSV file
test_results.to_csv("results.csv")
meta_features.to_csv("meta_features.csv")
```

For more information on the available functions, including passing individual arguments to the estimators, 
see the function documentation in the `tabmini` module.

## Usage - Standalone

To run the benchmark suite in a Docker container, you can execute the provided `execute_tabmini.sh` script. 
This script will build the container and run the benchmark suite. The results will be saved in the
`results` folder.

```bash
./execute_tabmini.sh
```

By default, this will run the `example.py` script (as described in the next section), which demonstrates how to use the `TabMini` benchmark suite.
You may replace our illustrative implementation of logistic regression with your own estimator.

## Example

For example usage, see `example.py`. This file is supposed to show how 
`TabMini` may be used. In the script, we demonstrate how to:

- Implement an estimator that is supposed to be compared to the other estimators (AutoGluon, AutoPrognosis, Hyperfast, TabPFN)
- Load the dataset
- Perform the comparison
- Save the results to a CSV file
- Load the results from a CSV file
- Perform a meta-feature analysis
- Save the meta-feature analysis to a CSV file.

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg