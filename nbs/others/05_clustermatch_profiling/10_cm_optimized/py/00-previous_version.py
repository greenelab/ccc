# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown]
# Creates a point of comparison with non-optimized version of clustermatch.

# %% [markdown] tags=[]
# # Remove pycache dir

# %%
# !echo ${CODE_DIR}

# %%
# !find ${CODE_DIR} -regex '^.*\(__pycache__\)$' -print

# %%
# !find ${CODE_DIR} -regex '^.*\(__pycache__\)$' -exec rm -rf {} \;

# %%
# !find ${CODE_DIR} -regex '^.*\(__pycache__\)$' -print

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import numpy as np

from clustermatch.coef import _cm

# %% [markdown] tags=[]
# # Data

# %%
N_REPS = 100
N_SAMPLES = 10000

# %% tags=[]
np.random.seed(0)

# %%
x = np.random.rand(N_SAMPLES)
y = np.random.rand(N_SAMPLES)


# %% [markdown] tags=[]
# # Run

# %% tags=[]
def func():
    for i in range(N_REPS):
        # py_func accesses the original python function, not the numba-optimized one
        # this is needed to be able to profile the function
        _cm.py_func(x, y)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 20 -T 00-previous_version.txt
func()

# %% [markdown] tags=[]
# The bottleneck functions are, in order of importance:
# 1. `_get_parts`
# 1. `cdist_parts`

# %%
