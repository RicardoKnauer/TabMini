#!/bin/bash

#SBATCH --job-name=tabmini_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --time=10-12:00:00
#SBATCH --qos=normal

source $HOME/.bashrc
conda activate tabmini

#known good single GPU:
#python cramming/pretrain.py name=test arch=bert-base train=bert-base data=sanity-check-2 impl.microbatch_size=2

python orcastrate.py