#!/bin/bash

# Updates the access date of all python scripts (.py) converted from notebooks.
# This is needed sometimes when git updates files after a pull, otherwise
# jupyter won't load the notebooks in the browser.

find . -type f -wholename "**/py/*.py" -exec touch {} +
