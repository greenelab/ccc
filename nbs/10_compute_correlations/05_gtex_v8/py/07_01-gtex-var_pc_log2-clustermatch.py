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
from time import time

import pandas as pd

from clustermatch import conf
from clustermatch.utils import simplify_string
from clustermatch.corr import clustermatch

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
GENE_SELECTION_STRATEGY = "var_pc_log2"

# %% tags=[]
# select the top 5 tissues (according to sample size, see nbs/05_preprocessing/00-gtex_v8-split_by_tissue.ipynb)
TISSUES = [
    # "Muscle - Skeletal",
    "Whole Blood",
    # "Skin - Sun Exposed (Lower leg)",
    # "Adipose - Subcutaneous",
    # "Artery - Tibial",
]

# %% tags=[]
CORRELATION_METHOD = clustermatch

method_name = CORRELATION_METHOD.__name__
display(method_name)

# %% tags=[]
PERFORMANCE_TEST_N_TOP_GENES = 500

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_DIR = conf.GTEX["GENE_SELECTION_DIR"]
display(INPUT_DIR)

assert INPUT_DIR.exists()

# %% tags=[]
OUTPUT_DIR = conf.GTEX["SIMILARITY_MATRICES_DIR"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Data loading

# %% tags=[]
tissue_in_file_names = [f"_data_{simplify_string(t.lower())}-" for t in TISSUES]

# %% tags=[]
input_files = sorted(list(INPUT_DIR.glob(f"*-{GENE_SELECTION_STRATEGY}.pkl")))
input_files = [
    f for f in input_files if any(tn in f.name for tn in tissue_in_file_names)
]
display(len(input_files))

assert len(input_files) == len(TISSUES), len(TISSUES)
display(input_files)

# %% [markdown] tags=[]
# # Compute similarity

# %% [markdown] tags=[]
# ## Performance test

# %% tags=[]
display(input_files[0])
test_data = pd.read_pickle(input_files[0]).iloc[:PERFORMANCE_TEST_N_TOP_GENES]

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
# %timeit -r1 CORRELATION_METHOD(test_data)

# %% [markdown] tags=[]
# ## Run

# %% tags=[]
for tissue_data_file in input_files:
    display(tissue_data_file.stem)

    # read
    data = pd.read_pickle(tissue_data_file)

    # compute correlations
    start_time = time()

    data_corrs = CORRELATION_METHOD(data)

    end_time = time()
    elapsed_time = end_time - start_time
    display(elapsed_time)

    # save
    output_filename = f"{tissue_data_file.stem}-{method_name}.pkl"
    data_corrs.to_pickle(path=OUTPUT_DIR / output_filename)

# %% tags=[]
