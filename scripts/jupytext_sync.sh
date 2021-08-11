#!/bin/bash

# This is used to go through all Jupyter notebooks, run black on the text
# representation of the code, and and sync with the ipynb file.

parallel 'jupytext --sync --pipe black {}' ::: nbs/**/*.ipynb
