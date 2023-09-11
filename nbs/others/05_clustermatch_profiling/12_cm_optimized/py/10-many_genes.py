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
# Similar as `06` but it computes across gene pairs instead of data matrix.

# %% [markdown] tags=[]
# # Remove pycache dir

# %% tags=[]
# !echo ${CODE_DIR}

# %% tags=[]
# !find ${CODE_DIR}/libs -regex '^.*\(__pycache__\)$' -print

# %% tags=[]
# !find ${CODE_DIR}/libs -regex '^.*\(__pycache__\)$' -exec rm -rf {} \;

# %% tags=[]
# !find ${CODE_DIR}/libs -regex '^.*\(__pycache__\)$' -print

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
n_genes, n_samples = 500, 1000

# %% tags=[]
np.random.seed(0)

# %% tags=[]
data = np.random.rand(n_genes, n_samples)

# %% tags=[]
data.shape


# %% [markdown] tags=[]
# # Profile

# %% tags=[]
def func():
    res = np.full(int((data.shape[0] * (data.shape[0] - 1)) / 2), np.nan)

    n_clust = list(range(2, 10 + 1))
    idx = 0
    for i in range(data.shape[0] - 1):
        for j in range(i + 1, data.shape[0]):
            res[idx] = ccc(data[i], data[j], internal_n_clusters=n_clust)
            idx += 1


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 50 -T 10-cm_many_genes.txt
func()

# %% tags=[]
