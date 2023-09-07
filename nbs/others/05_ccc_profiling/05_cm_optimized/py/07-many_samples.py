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
# Clustermatch run using a larger number of samples.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import numpy as np

from ccc.coef import ccc

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
# # With defeault `internal_n_clusters`

# %% tags=[]
def func():
    return ccc(data, internal_n_clusters=range(2, 10 + 1), precompute_parts=True)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 20 -T 07-cm_many_samples-default_internal_n_clusters.txt
func()


# %% [markdown] tags=[]
# # With defeault `internal_n_clusters`

# %% tags=[]
def func():
    return ccc(data, internal_n_clusters=range(2, 5 + 1), precompute_parts=True)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 20 -T 07-cm_many_samples-less_internal_n_clusters.txt
func()

# %% tags=[]
