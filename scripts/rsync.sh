#!/bin/bash

# This script runs rsync with some common parameters to sync with a remote
# machine running PhenoPLIER. For instance, it checks files' hashes instead of
# timestamp, and excludes some huge files from recount2.

rsync \
        -chavzP \
        --stats \
        --exclude 'recount_data_prep_PLIER.*' \
        --exclude 'recount2_PLIER_data.zip' \
        --exclude 'recount_PLIER_model.RDS' \
        --exclude 'srp/' \
        $@

