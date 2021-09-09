#!/bin/bash

# This file exports some common environmental variables to run the code.  It
# has to be customized for your need by changing the BASE_DIR and CM_N_JOBS
# below.

# Your settings here
# BASE_DIR is the parent directory where the code and manuscript repos are
# located.
BASE_DIR=/home/miltondp/projects/labs/greenelab/clustermatch_repos
export CM_N_JOBS=3

export CM_ROOT_DIR=${BASE_DIR}/clustermatch-gene-expr/base
export CM_MANUSCRIPT_DIR=${BASE_DIR}/clustermatch-gene-expr-manuscript/

export PYTHONPATH=${BASE_DIR}/clustermatch-gene-expr/libs/

