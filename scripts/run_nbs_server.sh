#!/bin/bash

# The script allows to run a JupyterLab server, listening to local connections
# only by default.
# It accepts only one argument, which could be:
#   * "--container-mode": it sets some parameters when starting the jupyter server
#   to make it work inside a Docker container.
#   * any other value: it is the token that the server will request from users;
#   in addition, it will listen to any address (*).

PORT=8893
export PYTHONPATH=`pwd`/libs/
echo "PYTHONPATH=${PYTHONPATH}"

# export the configuration as environmental variables (this is
# helpful if the configuration is needed outside python)
eval `python libs/clustermatch/conf.py`

# export variables related to CPU cores usage among several libraries
if [ -z "${CM_N_JOBS}" ]; then
  CM_N_JOBS=1
fi
export NUMBA_NUM_THREADS=${CM_N_JOBS}
export MKL_NUM_THREADS=${CM_N_JOBS}
export OPEN_BLAS_NUM_THREADS=${CM_N_JOBS}
export NUMEXPR_NUM_THREADS=${CM_N_JOBS}
export OMP_NUM_THREADS=${CM_N_JOBS}

IP="127.0.0.1"
TOKEN=""
EXTRA_ARGS=""
if [ "$1" = "--container-mode" ]; then
    IP="*"
    EXTRA_ARGS="--allow-root"
elif [ ! -z "$1" ]; then
	IP="*"
	TOKEN="${1}"
fi

exec jupyter lab --ip="${IP}" --port="${PORT}" --no-browser --ServerApp.token="${TOKEN}" ${EXTRA_ARGS}
