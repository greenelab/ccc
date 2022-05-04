#!/bin/bash
set -e

# Intended for internal use only, with very personalized settings.
#
# This script runs rsync with some common parameters to sync with a remote
# machine. For instance, it checks files' hashes instead of timestamp, and
# excludes some huge files not needed.

GIT_ROOT_DIR=$(git rev-parse --show-toplevel)
LOCAL_DIR="${GIT_ROOT_DIR}/base/"

# TODO directory should be [...]/projects/ccc/ccc/base
REMOTE_DIR="/home/miltondp/projects/labs/greenelab/clustermatch_repos/clustermatch-gene-expr/base/*"

rsync \
        -chavzP \
        --stats \
        --exclude 'GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz' \
        --exclude 'GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt' \
        --exclude 'recount_data_prep_PLIER.*' \
        --exclude 'recount2_PLIER_data.zip' \
        pcgreene:${REMOTE_DIR} \
        ${LOCAL_DIR}

