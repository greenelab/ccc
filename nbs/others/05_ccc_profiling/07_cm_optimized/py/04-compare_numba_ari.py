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
# This file actually does not compare different ari implementations. The name is kept to ease comparison with the previous runs from `05_cm_optimized` and `06_cm_optimized`.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import numpy as np

from ccc import coef

# %% [markdown] tags=[]
# # Data

# %% tags=[]
n_genes, n_samples = 100, 1000

# %% tags=[]
np.random.seed(0)

# %% tags=[]
data = np.random.rand(n_genes, n_samples)

# %% tags=[]
data.shape


# %% [markdown] tags=[]
# # Improved implementation (ARI implemented in numba)

# %% tags=[]
def func():
    n_clust = list(range(2, 10 + 1))
    return coef.ccc(data, internal_n_clusters=n_clust)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 50 -T 04-cm_ari_numba.txt
func()

# %% tags=[]
