#!/bin/bash

# This script installs other dependencies that cannot be directly installed using conda.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Fix tqdm with JupyterLab: https://github.com/tqdm/tqdm/issues/394#issuecomment-384743637
jupyter nbextension enable --py widgetsnbextension

jupyter labextension install @jupyter-widgets/jupyterlab-manager

#
# R dependencies
#
TAR=$(which tar) Rscript ${SCRIPT_DIR}/install_r_packages.r
