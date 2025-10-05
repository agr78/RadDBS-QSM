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
# Install PyTorch
echo Installing packages
$CONDA_PREFIX/bin/pip install -r requirements.txt 
# Restore environmental variables
unset PYTHONWARNINGS
# Install some package from pip
# $CONDA_PREFIX/bin/pip install pyspng
# $CONDA_PREFIX/bin/pip install imageio-ffmpeg==0.4.3
