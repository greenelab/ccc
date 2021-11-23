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

# %% [markdown] tags=[]
# Exactly the same code as in `08`, but here I run the notebook in a different machine (desktop).

# %% [markdown] tags=[]
# # Remove pycache dir

# %% tags=[]
# !echo ${CODE_DIR}

# %% tags=[]
# !find ${CODE_DIR} -regex '^.*\(__pycache__\)$' -print

# %% tags=[]
# !find ${CODE_DIR} -regex '^.*\(__pycache__\)$' -exec rm -rf {} \;

# %% tags=[]
# !find ${CODE_DIR} -regex '^.*\(__pycache__\)$' -print

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import numpy as np

from clustermatch.coef import cm

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
N_REPS = 10

# %% tags=[]
np.random.seed(0)

# %% [markdown] tags=[]
# # Setup

# %% tags=[]
# let numba compile all the code before profiling
cm(np.random.rand(10), np.random.rand(10))

# %% [markdown] tags=[]
# # Run with `n_samples` small

# %% tags=[]
N_SAMPLES = 100

# %% tags=[]
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
# %%prun -s cumulative -l 20 -T 09-n_samples_small.txt
func()

# %% [markdown] tags=[]
# # Run with `n_samples` large

# %% tags=[]
N_SAMPLES = 100000

# %% tags=[]
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
# %%prun -s cumulative -l 20 -T 09-n_samples_large.txt
func()

# %% tags=[]
