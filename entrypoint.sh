#!/bin/bash --login
# Taken from here with modifications: https://pythonspeed.com/articles/activate-conda-dockerfile/
# The --login ensures the bash configuration is loaded,
# enabling Conda.
set +eu
conda activate clustermatch_gene_expr
set -euo pipefail

# load environment variables
eval `python libs/conf.py`

exec "$@"

