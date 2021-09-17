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
# According to the settings specified below, this notebook:
#  1. reads all the data from one source (GTEx, recount2, etc) according to the gene selection method (`GENE_SELECTION_STRATEGY`),
#  2. runs a quick performance test using the correlation coefficient specified (`CORRELATION_METHOD`), and
#  3. computes the correlation matrix across all the genes using the correlation coefficient specified.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd

from clustermatch import conf
from clustermatch.corr import clustermatch


# %% [markdown] tags=[]
# # Settings

# %% tags=[]
# we don't have gene subsets for recount2
# GENE_SELECTION_STRATEGY = "var_raw"

# %% tags=[]
def clustermatch_k2(data):
    n_clusters = list(range(2, 2 + 1))
    return clustermatch(data, internal_n_clusters=n_clusters)


CORRELATION_METHOD = clustermatch_k2to5

method_name = CORRELATION_METHOD.__name__
display(method_name)

# %% tags=[]
PERFORMANCE_TEST_N_TOP_GENES = 500

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_FILE = conf.RECOUNT2["DATA_FILE"]
display(INPUT_FILE)

assert INPUT_FILE.exists()

# %% tags=[]
OUTPUT_DIR = conf.RECOUNT2["SIMILARITY_MATRICES_DIR"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Data loading

# %% tags=[]
data = pd.read_pickle(INPUT_FILE)

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% [markdown] tags=[]
# # Compute similarity

# %% [markdown] tags=[]
# ## Performance test

# %% tags=[]
# select a subset of the genes
test_data = data.sample(n=PERFORMANCE_TEST_N_TOP_GENES, random_state=0)

# %% tags=[]
test_data.shape

# %% tags=[]
test_data.head()

# %% [markdown] tags=[]
# This is a quick performance test of the correlation measure. The following line (`_tmp = ...`) is the setup code, which is needed in case the correlation method was optimized using `numba` and needs to be compiled before performing the test.

# %% tags=[]
_tmp = CORRELATION_METHOD(test_data.iloc[:3])

display(_tmp.shape)
display(_tmp)

# %% tags=[]
# %timeit CORRELATION_METHOD(test_data)

# %% [markdown] tags=[]
# ## Run

# %% tags=[]
# compute correlations
data_corrs = CORRELATION_METHOD(data)

# %% tags=[]
display(data_corrs.shape)

assert data.shape[0] == data_corrs.shape[0]

# %% tags=[]
data_corrs.head()

# %% tags=[]
output_filename = OUTPUT_DIR / f"{INPUT_FILE.stem}-{method_name}.pkl"
display(output_filename)

# %% tags=[]
# save
data_corrs.to_pickle(output_filename)

# %% tags=[]
