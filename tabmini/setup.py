from setuptools import setup, find_packages

setup(
    name='tabmini',
    version='1.1.0',
    python_requires='>3.10',
    url='https://github.com/RicardoKnauer/TabMini',
    author='Ricardo Knauer, Marvin Grimm, Erik Rodner',
    author_email='marvin.grimm@htw-berlin.de',
    description='Benchmarking and analysis of binary classifiers for tabular data',
    packages=find_packages(),
    install_requires=[
        "autoprognosis==0.1.21",
        "pandas==2.1.4",
        "pmlb==1.0.1.post3",
        "tabpfn==0.1.10",
        "scikit-learn==1.4.1.post1",
        "numpy==1.26.4",
        "autogluon==1.0.0",
        "hyperfast==1.0.2",
        "pymfe==0.4.3",
        "lightgbm==3.3.1",
        "xgboost==2.0.3",
        "catboost==1.2.3",
    ],
)
