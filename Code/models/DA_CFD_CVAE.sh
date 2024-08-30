#!/bin/bash
#PBS -l select=1:ncpus=1:mem=900gb:ngpus=1
#PBS -l walltime=04:00:00

# Activate the virtual environment
# source ~/usr/bin/activate

module load anaconda3/personal
source activate env

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

python DA_CFD_CVAE.py

# Deactivate the virtual environment
# deactivate
