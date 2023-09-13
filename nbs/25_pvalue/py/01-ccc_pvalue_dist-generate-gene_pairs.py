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
# Generates a distribution of pvalues under the null hypothesis of no association.
#
# This notebook uses individual gene pairs as input for CCC and parallelizes permutations.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
import numpy as np
from joblib import Parallel, delayed

from ccc.coef import ccc
from ccc import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
rs = np.random.RandomState(0)

# %% tags=[]
N_JOBS = 1
display(N_JOBS)

PVALUE_N_JOBS = conf.GENERAL["N_JOBS"]
display(PVALUE_N_JOBS)

# %% tags=[]
DATA_N_OBJS, DATA_N_FEATURES = 100, 1000
PVALUE_N_PERMS = 1000

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
OUTPUT_DIR = conf.RESULTS_DIR / "ccc_null-pvalues"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=[]
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
def ccc_single(x, y):
    return ccc(x, y, pvalue_n_perms=PVALUE_N_PERMS, n_jobs=PVALUE_N_JOBS)


# %% tags=[]
results = Parallel(n_jobs=N_JOBS)(
    delayed(ccc_single)(data[i], data[j])
    for i in range(data.shape[0] - 1)
    for j in range(i + 1, data.shape[0])
)

# %% tags=[]
assert len(results) == (DATA_N_OBJS * (DATA_N_OBJS - 1)) / 2

# %% tags=[]
results[0]

# %% tags=[]
cm_values = [x[0] for x in results]

# %% tags=[]
cm_pvalues = [x[1] for x in results]

# %% tags=[]
assert len(cm_values) == len(cm_pvalues)
assert len(cm_values) == (DATA_N_OBJS * (DATA_N_OBJS - 1)) / 2

# %% tags=[]
cm_values = np.array(cm_values)
cm_pvalues = np.array(cm_pvalues)

# %% tags=[]
cm_values.shape

# %% tags=[]
cm_values

# %% tags=[]
cm_pvalues.shape

# %% tags=[]
cm_pvalues

# %% [markdown] tags=[]
# # Save

# %% tags=[]
output_file = OUTPUT_DIR / "gene_pairs-cm_values.npy"
display(output_file)

np.save(output_file, cm_values)

# %% tags=[]
output_file = OUTPUT_DIR / "gene_pairs-cm_pvalues.npy"
display(output_file)

np.save(output_file, cm_pvalues)

# %% tags=[]
