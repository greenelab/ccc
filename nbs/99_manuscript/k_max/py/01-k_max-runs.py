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
# Runs CCC with different values for parameter $k_{\mathrm{max}}$ to assess the constant baseline property empirically.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
import numpy as np
import pandas as pd
from tqdm import tqdm

from ccc import conf
from ccc.coef import ccc

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
display(conf.GENERAL["N_JOBS"])

# %% tags=[]
DATA_SIZES = [
    200,
    600,
    1800,
]

# split data size in this many points
K_MAX_N_SPLITS = 10

# always include this value since it is the default we use in CCC
DEFAULT_K_MAX = 10

# N_REPS = 10

# %% tags=[]
np.random.seed(0)

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
OUTPUT_DIR = conf.RESULTS_DIR / "k_max_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Run

# %% tags=[]
# initialize (i.e., compile with numba)
ccc(np.random.rand(100), np.random.rand(100))

# %% tags=[]
results = pd.DataFrame(columns=["data_size", "k_max", "k_max_as_n_fraction", "coef"])

idx = 0
for data_size in tqdm(DATA_SIZES):
    # get the values for k_max to try...
    k_max_splits = np.linspace(2, data_size, K_MAX_N_SPLITS)
    # ... but always add the default k_max used by CCC
    k_max_splits = [int(i) for i in np.sort(np.append(k_max_splits, DEFAULT_K_MAX))]

    # generate random data
    # TODO: if I generate normal data, what happens?
    # d1 = np.random.rand(data_size)
    # d2 = np.random.rand(data_size)
    d1 = np.random.normal(size=data_size)
    d2 = np.random.normal(size=data_size)

    for k_max in tqdm(k_max_splits):
        c = ccc(d1, d2, internal_n_clusters=k_max, n_jobs=conf.GENERAL["N_JOBS"])

        results.loc[idx] = [data_size, k_max, k_max / data_size, c]
        idx += 1

        # save
        results.to_pickle(OUTPUT_DIR / "k_max-results.pkl")

# %% [markdown] tags=[]
# # Check

# %% tags=[]
results.shape

# %% tags=[]
assert results.shape[0] == int(len(DATA_SIZES) * (K_MAX_N_SPLITS + 1))

# %% tags=[]
results.head()

# %% tags=[]
