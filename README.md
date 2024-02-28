# TabMini

We introduce `TabMini`, the first tabular benchmark suite specifically for the low-data regime with 44 binary 
classification datasets, and use our suite to compare state-of-the-art machine learning methods, 
i.e., automated machine learning (AutoML) frameworks and off-the-shelf deep neural networks, 
against logistic regression.

### Installation/Development

This project was developed using a devcontainer, which is defined in the `.devcontainer` folder. 

Alternatively, conda environments have been defined for both linux and macOS in the `environment.yml` file 
and `environment_osx.yml` respectively. You can create a conda environment using the following command:

```bash
conda env create -f environment.yml
```

Further, `requirements.txt` file is also provided for those who prefer to use pip.

## Usage

For example usage, see `example.py`. This file is supposed to server as an illustrative example of how 
`TabMINI` may be used. In the script we demonstrate how to:

- Implement an estimator that is supposed to be compared to the other estimators (AutoGluon, AutoPrognosis, Hyperfast, TabPFN)
- Load the dataset
- Perform the comparison
- Save the results to a CSV file
- Loads the results from a CSV file
- Perform a meta features analysis
- Save the meta features analysis to a CSV file.

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg