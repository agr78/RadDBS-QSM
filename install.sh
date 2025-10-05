#!/bin/bash
# Ignore HTTPs warnings    
export PYTHONWARNINGS="ignore:Unverified HTTPS request"
# Initialize conda
eval "$(conda shell.bash hook)"
# Make environment
echo Creating and activating environment
conda create -n "RadDBS-QSMenv" python=3.7.0 ipython -y
# Activate it
conda activate RadDBS-QSMenv
# Install
echo Installing packages
$CONDA_PREFIX/bin/pip install -r requirements.txt 
# Apply sklearn patch
mv ./src/patch/_least_angle.py $CONDA_PREFIX/envs/RadDBS-QSMenv/lib/python3.7/site-packages/sklearn/linear_model/_least_angle.py
# Restore environmental variables
unset PYTHONWARNINGS

