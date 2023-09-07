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
# Clustermatch run using a larger number of samples.

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

from ccc.coef import ccc

# %% tags=[]
# let numba compile all the code before profiling
ccc(np.random.rand(10), np.random.rand(10))

# %% [markdown] tags=[]
# # Data

# %% tags=[]
n_genes, n_samples = 10, 30000

# %% tags=[]
np.random.seed(0)

# %% tags=[]
data = np.random.rand(n_genes, n_samples)

# %% tags=[]
data.shape


# %% [markdown] tags=[]
# # With default `internal_n_clusters`

# %% tags=[]
def func():
    n_clust = list(range(2, 10 + 1))
    return ccc(data, internal_n_clusters=n_clust)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 50 -T 07-cm_many_samples-default_internal_n_clusters.txt
func()


# %% [markdown] tags=[]
# # With reduced `internal_n_clusters`

# %% tags=[]
def func():
    n_clust = list(range(2, 5 + 1))
    return ccc(data, internal_n_clusters=n_clust)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 50 -T 07-cm_many_samples-less_internal_n_clusters.txt
func()

# %% tags=[]
