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
# TODO

# %% [markdown] tags=[]
# # Modules loading

# %% [markdown]
# Make sure only one core is used everywhere.

# %%
# %env CM_N_JOBS=1
# %env NUMBA_NUM_THREADS=1
# %env MKL_NUM_THREADS=1
# %env OPEN_BLAS_NUM_THREADS=1
# %env NUMEXPR_NUM_THREADS=1
# %env OMP_NUM_THREADS=1

# %% tags=[]
from time import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from clustermatch import conf
from clustermatch.coef import cm
from clustermatch.methods import mic

# %% [markdown] tags=[]
# # Settings

# %%
OUTPUT_FILENAME = "time_test-ccc_p_s.pkl"

# %% tags=[]
DATA_SIZES = [
    100,
    500,
    1000,
    5000,
    10000,
    100000,
    1000000,
]

N_REPS = 10

# %%
np.random.seed(0)

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
OUTPUT_DIR = (
    conf.RESULTS_DIR / "time_test"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Functions

# %%
time_results = pd.DataFrame(columns=['data_size', 'method', 'time', 'sim'])


# %%
def run_method(func, method_name, size):
    for r in range(N_REPS):
        d1 = np.random.rand(size)
        d2 = np.random.rand(size)

        start_time = time()
        sim = func(d1, d2)
        end_time = time()
        met_time = end_time - start_time

        idx = time_results.shape[0]
        time_results.loc[idx] = [d1.shape[0], method_name, met_time, sim]


# %% [markdown]
# # Run

# %%
for s in DATA_SIZES:
    print(f'Size: {s}')

    print(f'  p')
    run_method(lambda x, y: pearsonr(x, y)[0], 'p', s)
    
    print(f'  s')
    run_method(lambda x, y: spearmanr(x, y)[0], 's', s)
    
    print(f'  cm')
    run_method(lambda x, y: cm(x, y), 'cm', s)

    print('Saving to pickle')
    time_results.to_pickle(OUTPUT_DIR / OUTPUT_FILENAME)

# %%
