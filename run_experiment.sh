#!/bin/bash

# Ensure that conda is available
source ~/.bashrc

# Activate the conda environment
conda activate $1

# Run the experiment script
python example.py $1 $2 $3

# Deactivate the conda environment
conda deactivate
