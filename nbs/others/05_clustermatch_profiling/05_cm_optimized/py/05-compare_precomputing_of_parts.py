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
# Compares two different ccc implementations: one using precomputation of internal clusterings, and the other one using the original implementation that does not perform such precomputation.

# %% [markdown]
# # Modules

# %% tags=[]
import numpy as np

from ccc.coef import ccc

# %% [markdown]
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
# # Improved implementation (`precompute_parts=True`)

# %% tags=[]
def func():
    return ccc(data, internal_n_clusters=range(2, 10 + 1), precompute_parts=True)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 20 -T 05-cm_precompute_parts_true.txt
func()


# %% [markdown] tags=[]
# # Original implementation (`precompute_parts=False`)

# %% tags=[]
def func():
    return ccc(data, internal_n_clusters=range(2, 10 + 1), precompute_parts=False)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 20 -T 05-cm_precompute_parts_false.txt
func()

# %% tags=[]
