#!/bin/bash

# It runs the Docker container of this project by mounting the code and
# manuscript directories inside the container. This makes that any file created
# during the execution is locally available and ready to be pushed to the repo.
# Plus, the code is always run inside the same environment (including the full
# operating system).
#
# We assume the repo code is in the current directory, so the user has to make
# sure this is right.

# general settings
DOCKER_IMAGE_NAMESPACE="miltondp"
DOCKER_IMAGE_NAME="ccc"
DOCKER_TAG="${CM_DOCKER_IMAGE_TAG:-latest}"
DOCKER_PUBLISH_HOST="127.0.0.1"
DOCKER_CONTAINER_PORT="8893"
DOCKER_HOST_PORT="8893"

# project-specific environment variables
ROOT_DIR="${CM_ROOT_DIR}"
MANUSCRIPT_DIR="${CM_MANUSCRIPT_DIR}"
N_JOBS_VARNAME="CM_N_JOBS"
N_JOBS=${!N_JOBS_VARNAME}

# parameters parsing
# read arguments
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --docker-args)
      DOCKER_ARGS="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

echo "Configuration:"

CODE_DIR=`pwd`

# root dir
if [ -z "${ROOT_DIR}" ]; then
  ROOT_DIR="${CODE_DIR}/base"
fi

# manuscript dir
if [ -z "${MANUSCRIPT_DIR}" ]; then
  MANUSCRIPT_DIR="/tmp/${DOCKER_IMAGE_NAME}_manuscript"
  mkdir -p ${MANUSCRIPT_DIR}
fi

if [ -z "${N_JOBS}" ]; then
  N_JOBS=1
fi

echo "Configuration:"
echo "  Code dir: ${CODE_DIR}"
echo "  Root dir: ${ROOT_DIR}"
echo "  Manuscript dir: ${MANUSCRIPT_DIR}"
echo "  CPU cores: ${N_JOBS}"
echo "  Docker image tag: ${DOCKER_TAG}"

echo
echo "Waiting 2 seconds before starting"
echo
sleep 2

# always create data directory before running Docker
mkdir -p ${ROOT_DIR}

COMMAND="$@"
PORT_ARG="-p ${DOCKER_PUBLISH_HOST}:${DOCKER_HOST_PORT}:${DOCKER_CONTAINER_PORT}"
if [ -z "${COMMAND}" ]; then
  FULL_COMMAND=()
else
  FULL_COMMAND=(/bin/bash -c "${COMMAND}")
  PORT_ARG=""
fi

echo "Full command: ${FULL_COMMAND}"

if [ -z "${DOCKER_ARGS}" ]; then
  # by default, use interactive mode (enables cancelling run with Ctrl C from console)
  DOCKER_ARGS="-ti"
fi

# show commands being executed
echo
set -x

# run
docker run --rm ${PORT_ARG} ${DOCKER_ARGS} \
  -e ${N_JOBS_VARNAME}=${N_JOBS} \
  -e NUMBA_NUM_THREADS=${N_JOBS} \
  -e MKL_NUM_THREADS=${N_JOBS} \
  -e OPEN_BLAS_NUM_THREADS=${N_JOBS} \
  -e NUMEXPR_NUM_THREADS=${N_JOBS} \
  -e OMP_NUM_THREADS=${N_JOBS} \
  -e "${CM_RUN_NBS_OVERRIDE}":${CM_RUN_NBS_OVERRIDE:-0} \
  -v "${CODE_DIR}:/opt/code" \
  -v "${ROOT_DIR}:/opt/data" \
  -v "${MANUSCRIPT_DIR}:/opt/manuscript" \
  --user "$(id -u):$(id -g)" \
  ${DOCKER_IMAGE_NAMESPACE}/${DOCKER_IMAGE_NAME}:${DOCKER_TAG} "${FULL_COMMAND[@]}"

