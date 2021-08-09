#!/bin/bash

parallel 'jupytext --sync --pipe black {}' ::: nbs/**/*.ipynb

