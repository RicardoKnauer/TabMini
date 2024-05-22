using PyCall


py"""
import numpy as np
import pandas as pd
from pymfe.mfe import MFE

def generate_metafeatures(path):

    results_wide = pd.read_csv(path, delimiter=";")

    files = [
        # M = 32 - 100 (12 datasets)
        ["analcatdata_aids", "analcatdata_asbestos", "analcatdata_bankruptcy", "analcatdata_creditscore",
        "analcatdata_cyyoung8092", "analcatdata_cyyoung9302", "analcatdata_fraud",
        "analcatdata_japansolvent", "labor", "lupus", "parity5", "postoperative_patient_data"],
        # M = 101 - 200 (10 datasets)
        ["analcatdata_boxing1", "analcatdata_boxing2", "appendicitis", "backache", "corral", "glass2",
        "hepatitis", "molecular_biology_promoters", "mux6", "prnn_crabs"],
        # M = 201 - 300 (9 datasets)
        ["analcatdata_lawsuit", "biomed", "breast_cancer", "heart_h", "heart_statlog", "hungarian",
        "prnn_synth", "sonar", "spect"],
        # M = 301 - 400 (8 datasets)
        ["bupa", "cleve", "colic", "haberman", "heart_c", "horse_colic", "ionosphere", "spectf"],
        # M = 401 - 500 (5 datasets)
        ["clean1", "house_votes_84", "irish", "saheart", "vote"]
    ]

    metafeatures = []
    column_names = []
    for idx, datasets in enumerate(files, 1):
        for dataset in datasets:
            X = pd.read_csv(f"data/{idx}/{dataset}/X.csv")
            y = pd.read_csv(f"data/{idx}/{dataset}/y.csv")
            # 3932 metafeatures (pymfe.readthedocs.io/en/latest/auto_pages/meta_features_description.html)
            mfe = MFE(groups="all", summary="all", num_cv_folds=3, random_state=42)
            mfe.fit(X.to_numpy(), y.to_numpy())
            ft = mfe.extract(suppress_warnings=True)
            metafeatures.append(ft[1] + [min(sum(y.target), len(y.target) - sum(y.target))])
            column_names = ft[0] + ["EPV"]

    results_metafeatures = pd.DataFrame(metafeatures, columns=column_names)
    results_metafeatures["EPV"] = results_metafeatures["EPV"] / results_metafeatures["nr_attr"]

    results_metafeatures.to_csv("metafeatures.csv", index=False)
"""
