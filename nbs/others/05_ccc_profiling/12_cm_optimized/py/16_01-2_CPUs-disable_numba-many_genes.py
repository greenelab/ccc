# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# Like `06_01` but using more CPU cores.

# %% [markdown] tags=[]
# # Use multiple CPU core

# %% tags=[]
# %env CM_N_JOBS=2
# %env NUMBA_NUM_THREADS=2
# %env MKL_NUM_THREADS=2
# %env OPEN_BLAS_NUM_THREADS=2
# %env NUMEXPR_NUM_THREADS=2
# %env OMP_NUM_THREADS=2

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

from ccc.coef import ccc

# %% tags=[]
# let numba compile all the code before profiling
ccc(np.random.rand(10), np.random.rand(10))

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
# # Profile

# %% tags=[]
def func():
    n_clust = list(range(2, 10 + 1))
    return ccc(data, internal_n_clusters=n_clust, n_jobs=2)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
