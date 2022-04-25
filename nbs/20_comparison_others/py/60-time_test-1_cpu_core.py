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
# It generates random variables of varying sizes to compare the time taken by CCC and MIC.
#
# This notebook uses 1 CPU core.

# %% [markdown] tags=[]
# # Modules loading

# %% [markdown] tags=[]
# Make sure only one core is used everywhere.

# %% tags=[]
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

# %% tags=[]
OUTPUT_FILENAME = "time_test.pkl"

# %% tags=[]
DATA_SIZES = [
    100,
    500,
    1000,
    5000,
    10000,
    50000,
    100000,
    1000000,
]

N_REPS = 10

# %% tags=[]
np.random.seed(0)

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
OUTPUT_DIR = conf.RESULTS_DIR / "time_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Functions

# %% tags=[]
time_results = pd.DataFrame(columns=["data_size", "method", "time", "sim"])


# %% tags=[]
def run_method(func, method_name, size):
    n_reps = N_REPS
    if size < 500:
        n_reps = 1000

    for r in range(n_reps):
        d1 = np.random.rand(size)
        d2 = np.random.rand(size)

        start_time = time()
        sim = func(d1, d2)
        end_time = time()
        met_time = end_time - start_time

        idx = time_results.shape[0]
        time_results.loc[idx] = [d1.shape[0], method_name, met_time, sim]


# %% [markdown] tags=[]
# # Run

# %% tags=[]
# initialize methods
cm(np.random.rand(100), np.random.rand(100))

# %% tags=[]
for s in DATA_SIZES:
    print(f"Size: {s}")

    print(f"  p")
    run_method(lambda x, y: pearsonr(x, y)[0], "p-1", s)

    print(f"  s")
    run_method(lambda x, y: spearmanr(x, y)[0], "s-1", s)

    print(f"  cm")
    run_method(lambda x, y: cm(x, y), "cm-1", s)

    if s <= 50000:
        print(f"  mic_e")
        run_method(lambda x, y: mic(x, y, estimator="mic_e"), "mic_e-1", s)

    if s <= 10000:
        print(f"  mic")
        run_method(lambda x, y: mic(x, y), "mic-1", s)

    print("Saving to pickle")
    time_results.to_pickle(OUTPUT_DIR / OUTPUT_FILENAME)

# %% tags=[]
