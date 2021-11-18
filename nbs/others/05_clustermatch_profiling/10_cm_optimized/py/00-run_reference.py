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
# Creates a point of reference/comparison with non-optimized version of clustermatch.

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
# # Settings

# %%
N_REPS = 10

# %% tags=[]
np.random.seed(0)

# %% [markdown] tags=[]
# # Setup

# %%
# let numba compile all the code before profiling
_cm.py_func(np.random.rand(10), np.random.rand(10))

# %% [markdown] tags=[]
# # Run with `n_samples` small

# %%
N_SAMPLES = 100

# %%
x = np.random.rand(N_SAMPLES)
y = np.random.rand(N_SAMPLES)


# %% tags=[]
def func():
    for i in range(N_REPS):
        # py_func accesses the original python function, not the numba-optimized one
        # this is needed to be able to profile the function
        _cm.py_func(x, y)


# %% tags=[]
# %%timeit -n1 -r1 func()
func()

# %% tags=[]
# %%prun -s cumulative -l 20 -T 00-n_samples_small.txt
func()

# %% [markdown] tags=[]
# The bottleneck functions are, in order of importance:
# 1. `cdist_parts`
# 1. `_get_parts`

# %% [markdown] tags=[]
# # Run with `n_samples` large

# %%
N_SAMPLES = 100000

# %%
x = np.random.rand(N_SAMPLES)
y = np.random.rand(N_SAMPLES)


# %% tags=[]
def func():
    for i in range(N_REPS):
        # py_func accesses the original python function, not the numba-optimized one
        # this is needed to be able to profile the function
        _cm.py_func(x, y)


# %% tags=[]
# %%timeit -n1 -r1 func()
func()

# %% tags=[]
# %%prun -s cumulative -l 20 -T 00-n_samples_large.txt
func()

# %% [markdown] tags=[]
# The bottleneck functions now are **different**, in order of importance:
# 1. `_get_parts`
# 1. `cdist_parts`

# %%
