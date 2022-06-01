#!/bin/bash
set -e

# Intended for internal use only, with very personalized settings.
#
# This script runs rsync with some common parameters to sync with a remote
# machine. For instance, it checks files' hashes instead of timestamp, and
# excludes some huge files not needed.
#
# It accepts one argument, and it is the remote directory path (absolute) where
# the base directory is.

GIT_ROOT_DIR=$(git rev-parse --show-toplevel)
LOCAL_DIR="${GIT_ROOT_DIR}/base/"

REMOTE_DIR="${1}"
if [ -z "${REMOTE_DIR}" ]; then
  # if remote dir not given, use the same as local
  REMOTE_DIR=${LOCAL_DIR}
else
  # default value
  REMOTE_DIR="/home/miltondp/projects/ccc/ccc/base/*"
fi

rsync \
        -chavzP \
        --stats \
        --exclude 'GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz' \
        --exclude 'GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt' \
        --exclude 'recount_data_prep_PLIER.*' \
        --exclude 'recount2_PLIER_data.zip' \
        pcgreene:${REMOTE_DIR} \
        ${LOCAL_DIR}

