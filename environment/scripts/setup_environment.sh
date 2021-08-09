#!/bin/bash
set -e

# create conda environment and install main packages
conda env create --name clustermatch_gene_expr --file environment.yml

# install other packages
conda run -n clustermatch_gene_expr --no-capture-output bash scripts/install_other_packages.sh

# download the data
conda run -n clustermatch_gene_expr --no-capture-output python scripts/setup_data.py

