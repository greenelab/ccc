#!/bin/bash

# This script is intended to be used by the developer not the end user.
#
# It runs the Docker container of this project by mounting the code and
# manuscript directories inside the container. This makes that any file created
# during the execution is locally available and ready to be pushed to the repo.
# Plus, the code is always run inside the same environment (including the full
# operating system).

CODE_DIR="/home/miltondp/projects/labs/greenelab/clustermatch_repos/clustermatch-gene-expr"
DATA_DIR="${CODE_DIR}/base"
MANUSCRIPT_DIR="/home/miltondp/projects/labs/greenelab/clustermatch_repos/clustermatch-gene-expr-manuscript"
N_JOBS=3

# always create data directory before running Docker
mkdir -p ${DATA_DIR}

COMMAND="$@"

docker run --rm \
  -e CM_N_JOBS=${N_JOBS} \
  -v "${CODE_DIR}:/opt/code" \
  -v "${DATA_DIR}:/opt/data" \
  -v "${MANUSCRIPT_DIR}:/opt/manuscript" \
  --user "$(id -u):$(id -g)" \
  miltondp/clustermatch_gene_expr \
  /bin/bash -c "${COMMAND}"

