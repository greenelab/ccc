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
# Now `_get_parts` has been optimized with previous profiling tests.
#
# Here we profile function `cdist_parts` again.
#
# This version is the final test with the completely optimized clustermatch's functions.
#
# This notebook was run on my laptop.

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

from clustermatch.coef import cm

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
cm(np.random.rand(10), np.random.rand(10))

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
        cm(x, y)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 20 -T 08-n_samples_small.txt
func()

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
        cm(x, y)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 20 -T 08-n_samples_large.txt
func()

# %%
