#!/bin/bash

NOTEBOOK="${1}"
if [ -z "${NOTEBOOK}" ]; then
  echo "Provide the notebook path"
  exit 1
fi

jupytext \
  --sync \
  --pipe black \
  ${NOTEBOOK}

RESULT=$?
if [ $RESULT -ne 0 ]; then
    # if jupytext failed, it is very likely because it was an R script.
    # Try again without black
    jupytext \
      --sync \
      ${NOTEBOOK}
fi
