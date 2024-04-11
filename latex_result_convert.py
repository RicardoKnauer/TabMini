import pandas as pd

# Read the CSV file
df = pd.read_csv("workdir/your_results.csv")

# Drop the Logistic Regression column
df = df.drop(columns=["Logistic Regression"])

# Round the DataFrame to 2 decimal places
df = df.round(2)

# escape the underscores in the dataset names
df["PMLB dataset"] = df["PMLB dataset"].str.replace("_", "\_")

# Sort the DataFrame so that the dataset name is in the order of the PMLB dataset
order = [
    "parity5",
    "analcatdata\_fraud",
    "analcatdata\_aids",
    "analcatdata\_bankruptcy",
    "analcatdata\_japansolvent",
    "labor",
    "analcatdata\_asbestos",
    "lupus",
    "postoperative\_patient\_data",
    "analcatdata\_cyyoung9302",
    "analcatdata\_cyyoung8092",
    "analcatdata\_creditscore",
    "appendicitis",
    "molecular\_biology\_promoters",
    "analcatdata\_boxing1",
    "mux6",
    "analcatdata\_boxing2",
    "hepatitis",
    "corral",
    "glass2",
    "backache",
    "prnn\_crabs",
    "sonar",
    "biomed",
    "prnn\_synth",
    "analcatdata\_lawsuit",
    "spect",
    "heart\_statlog",
    "breast\_cancer",
    "heart\_h",
    "hungarian",
    "cleve",
    "heart\_c",
    "haberman",
    "bupa",
    "spectf",
    "ionosphere",
    "colic",
    "horse\_colic",
    "house\_votes\_84",
    "vote",
    "saheart",
    "clean1",
    "irish",
]

df["PMLB dataset"] = pd.Categorical(df["PMLB dataset"], categories=order, ordered=True)
df = df.sort_values("PMLB dataset")

# Convert the DataFrame to a LaTeX table
latex_table = df.to_latex(index=False)

# Write the LaTeX table to a .tex file
with open('workdir/your_results_as_latex.tex', 'w') as f:
    f.write(latex_table)
