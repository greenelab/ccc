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
# Generates a distribution of pvalues under the null hypothesis of no association.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
import numpy as np

from ccc.coef import ccc
from ccc import conf

# %% [markdown] tags=[]
# # Settings

# %%
rs = np.random.RandomState(0)

# %%
DATA_N_OBJS, DATA_N_FEATURES = 100, 1000
PVALUE_N_PERMS = 1000

# %% [markdown]
# # Paths

# %%
OUTPUT_DIR = conf.RESULTS_DIR / "ccc_null-pvalues"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
OUTPUT_DIR

# %% [markdown] tags=[]
# # Generate random data

# %% tags=[]
data = rs.rand(DATA_N_OBJS, DATA_N_FEATURES)

# %% tags=[]
data.shape

# %% [markdown] tags=[]
# # Run CCC

# %% tags=[]
res = ccc(data, n_jobs=conf.GENERAL["N_JOBS"], pvalue_n_perms=PVALUE_N_PERMS, use_ari_numba=True)

# %%
cm_values, cm_pvalues = res

# %%
cm_values.shape

# %%
cm_pvalues.shape

# %% [markdown] tags=[]
# # Save

# %%
output_file = OUTPUT_DIR / "cm_values.npy"
display(output_file)

np.save(output_file, cm_values)

# %%
output_file = OUTPUT_DIR / "cm_pvalues.npy"
np.save(output_file, cm_pvalues)

# %%
