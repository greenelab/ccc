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

# %% tags=[]
import numpy as np

from clustermatch import coef

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
    return coef.cm(data, internal_n_clusters=range(2, 10 + 1), precompute_parts=True)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 20 -T 04-cm_ari_numba.txt
func()

# %% [markdown] tags=[]
# # Original implementation (ARI from sklearn)

# %% tags=[]
from sklearn.metrics import adjusted_rand_score

# %% tags=[]
coef.ari = adjusted_rand_score


# %% tags=[]
def func():
    return coef.cm(data, internal_n_clusters=range(2, 10 + 1), precompute_parts=True)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 20 -T 04-cm_ari_sklearn.txt
func()

# %% tags=[]
