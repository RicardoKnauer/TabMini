using PyCall


py"""
from contextlib import redirect_stdout
import numpy as np
import pandas as pd

def generate_correlations(methods, path):

    results_wide = pd.read_csv(path, delimiter=";")
    
    with open("correlations.txt", "w") as f:
        with redirect_stdout(f):
            for method in methods:
                metafeatures = pd.read_csv("metafeatures.csv")
                metafeatures[f"diff_{method}"] = results_wide[method] - \
                    results_wide["Logistic regression"]
                coeffs = metafeatures.corr("spearman")[f"diff_{method}"]
                # largest absolute Spearman rank correlation coefficients
                largest_coeffs_pos = coeffs.nlargest(11)[1:]
                largest_coeffs_neg = coeffs.nsmallest(10)
                print(f"{method}, largest positive:\n{round(largest_coeffs_pos, 5)}\n")
                print(f"{method}, largest negative:\n{round(largest_coeffs_neg, 5)}\n")
"""
