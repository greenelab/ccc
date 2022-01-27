#!/bin/bash

# Executes all python scripts (.py) converted from notebooks to update their
# access date. This is needed sometimes when git updates files after a push,
# otherwise jupyter won't load the notebooks in the browser.

find . -type f -wholename "**/py/*.py" -exec touch {} +

