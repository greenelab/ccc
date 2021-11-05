#!/bin/bash

# This script is intended to be used by the developer not the end user.
#
# It runs the Docker container of this project by mounting the code and
# manuscript directories inside the container. This makes that any file created
# during the execution is locally available and ready to be pushed to the repo.
# Plus, the code is always run inside the same environment (including the full
# operating system).

# We assume the repo code is in the current directory, so the user has to make
# sure this is right.

echo "Configuration:"

CODE_DIR=`pwd`
echo "  Code dir: ${CODE_DIR}"

if [ -z "${CM_ROOT_DIR}" ]; then
  ROOT_DIR="${CODE_DIR}/base"
else
  ROOT_DIR="${CM_ROOT_DIR}"
fi

echo "  Root dir: ${ROOT_DIR}"

if [ -z "${CM_MANUSCRIPT_DIR}" ]; then
  echo "  ERROR: manuscript directory is not set"
  exit 1
fi

echo "  Manuscript dir: ${CM_MANUSCRIPT_DIR}"

echo "  CPU cores: ${CM_N_JOBS}"

echo ""
echo "Waiting 5 seconds before starting"
sleep 5

# always create data directory before running Docker
mkdir -p ${ROOT_DIR}

COMMAND="$@"
if [ -z "${COMMAND}" ]; then
  FULL_COMMAND=()
else
  FULL_COMMAND=(/bin/bash -c "${COMMAND}")
fi

echo "${FULL_COMMAND}"

# show commands being executed
set -x

# run
docker run --rm \
  -e CM_N_JOBS=${CM_N_JOBS} \
  -e NUMBA_NUM_THREADS=${CM_N_JOBS} \
  -e MKL_NUM_THREADS=${CM_N_JOBS} \
  -e OPEN_BLAS_NUM_THREADS=${CM_N_JOBS} \
  -e NUMEXPR_NUM_THREADS=${CM_N_JOBS} \
  -e OMP_NUM_THREADS=${CM_N_JOBS} \
  -v "${CODE_DIR}:/opt/code" \
  -v "${ROOT_DIR}:/opt/data" \
  -v "${CM_MANUSCRIPT_DIR}:/opt/manuscript" \
  -p 8888:8893 \
  --user "$(id -u):$(id -g)" \
  miltondp/clustermatch_gene_expr "${FULL_COMMAND[@]}"

