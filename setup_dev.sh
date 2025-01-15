#!/bin/bash

# Used to setup the development environment for CCC
# Can be loaded by PyCharm on startup

# Find the conda path
CONDA_PATH=$(conda info | grep -i 'base environment' | awk -F': ' '{print $2}' | awk '{print $1}')
source ${CONDA_PATH}/etc/profile.d/conda.sh

# Activate the conda environment
conda activate ccc-gpu

# Set the PYTHONPATH
export PYTHONPATH=`readlink -f ./libs/`:$PYTHONPATH
