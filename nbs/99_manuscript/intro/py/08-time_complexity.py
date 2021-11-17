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
# # Modules

# %% tags=[]
# import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.preprocessing import minmax_scale

from clustermatch import conf
from clustermatch.coef import cm
from clustermatch.methods import mic, distcorr

# %% [markdown] tags=[]
# # Settings

# %%
N_REPS = 100
N_SAMPLES_LIST = [10, 50, 100, 500]

# %% [markdown] tags=[]
# # Paths

# %%
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None and conf.MANUSCRIPT["BASE_DIR"].exists()
), "Manuscript dir not set"

# %% tags=[]
OUTPUT_FIGURE_DIR = conf.MANUSCRIPT["FIGURES_DIR"] / "intro"
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% [markdown]
# # Run

# %%
cm_times = []
mic_times = []
distcorr_times = []

# %%
for i, n_samples in enumerate(N_SAMPLES_LIST):
        x = np.random.rand(size=n_samples)
        y = np.random.rand(size=n_samples)
        
        avl_times[i] = timeit(avl, number=n_times)
