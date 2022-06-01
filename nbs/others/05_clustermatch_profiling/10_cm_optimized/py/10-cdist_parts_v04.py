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
# Exactly the same code as in `09`, but here I disable numba.

# %% [markdown] tags=[]
# # Disable numba

# %% tags=[]
# %env NUMBA_DISABLE_JIT=1

# %% [markdown] tags=[]
# # Remove pycache dir

# %% tags=[]
# !echo ${CODE_DIR}

# %% tags=[]
# !find ${CODE_DIR} -regex '^.*\(__pycache__\)$' -print

# %% tags=[]
# !find ${CODE_DIR} -regex '^.*\(__pycache__\)$' -prune -exec rm -rf {} \;

# %% tags=[]
# !find ${CODE_DIR} -regex '^.*\(__pycache__\)$' -print

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import numpy as np

from clustermatch.coef import ccc

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
ccc(np.random.rand(10), np.random.rand(10))

# %% [markdown] tags=[]
# # Run with `n_samples` small

# %% [markdown] tags=[]
# ## `n_samples=50`

# %% tags=[]
N_SAMPLES = 50

# %% tags=[]
x = np.random.rand(N_SAMPLES)
y = np.random.rand(N_SAMPLES)


# %% tags=[]
def func():
    for i in range(N_REPS):
        ccc(x, y)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 20 -T 10-n_samples_small_50.txt
func()

# %% [markdown] tags=[]
# ## `n_samples=100`

# %% tags=[]
N_SAMPLES = 100

# %% tags=[]
x = np.random.rand(N_SAMPLES)
y = np.random.rand(N_SAMPLES)


# %% tags=[]
def func():
    for i in range(N_REPS):
        ccc(x, y)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 20 -T 10-n_samples_small_100.txt
func()

# %% [markdown] tags=[]
# ## `n_samples=500`

# %% tags=[]
N_SAMPLES = 500

# %% tags=[]
x = np.random.rand(N_SAMPLES)
y = np.random.rand(N_SAMPLES)


# %% tags=[]
def func():
    for i in range(N_REPS):
        ccc(x, y)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 20 -T 10-n_samples_small_500.txt
func()

# %% [markdown] tags=[]
# ## `n_samples=1000`

# %% tags=[]
N_SAMPLES = 1000

# %% tags=[]
x = np.random.rand(N_SAMPLES)
y = np.random.rand(N_SAMPLES)


# %% tags=[]
def func():
    for i in range(N_REPS):
        ccc(x, y)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 20 -T 10-n_samples_small_1000.txt
func()

# %% [markdown] tags=[]
# **CONCLUSION:** as expected, with relatively small samples, the numba-compiled version (`09-cdist_parts_v04`) performs much better than the non-compiled one.

# %% [markdown] tags=[]
# # Run with `n_samples` large

# %% [markdown] tags=[]
# ## `n_samples=50000`

# %% tags=[]
N_SAMPLES = 50000

# %% tags=[]
x = np.random.rand(N_SAMPLES)
y = np.random.rand(N_SAMPLES)


# %% tags=[]
def func():
    for i in range(N_REPS):
        ccc(x, y)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 20 -T 10-n_samples_large_50000.txt
func()

# %% [markdown] tags=[]
# ## `n_samples=100000`

# %% tags=[]
N_SAMPLES = 100000

# %% tags=[]
x = np.random.rand(N_SAMPLES)
y = np.random.rand(N_SAMPLES)


# %% tags=[]
def func():
    for i in range(N_REPS):
        ccc(x, y)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 20 -T 10-n_samples_large_100000.txt
func()

# %% [markdown] tags=[]
# **CONCLUSION:** this is unexpected. With very large samples, the python version performs better! Something to look at in the future. The profiling file for 100,000 samples () shows that the `cdist_parts_parallel` is taking more time in the numba-compiled version than in the python version. Maybe the compiled ARI implementation could be improved in these cases with large samples.

# %% tags=[]
