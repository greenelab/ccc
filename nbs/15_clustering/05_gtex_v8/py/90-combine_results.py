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
# Combines all gene enrichment results found in input directory.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
import re

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from clustermatch import conf

# %% [markdown] tags=[]
# # Settings

# %%
DATASET_CONFIG = conf.GTEX

# %% tags=[]
# ENRICH_FUNCTION = "enrichGO"

# %% tags=[]
# CORRELATION_METHOD_NAME = "clustermatch"

# %% tags=[]
# GENE_SELECTION_STRATEGY = "var_pc_log2"

# %% tags=[]
# # clusterProfiler settings
# ENRICH_FUNCTION = "enrichGO"
# SIMPLIFY_CUTOFF = 0.7
# GO_ONTOLOGIES = ("BP", "CC", "MF")

# %% tags=[]
# SIMILARITY_MATRICES_DIR = conf.GTEX["SIMILARITY_MATRICES_DIR"]
# display(SIMILARITY_MATRICES_DIR)

# %% tags=[]
# SIMILARITY_MATRIX_FILENAME_TEMPLATE = conf.GTEX["SIMILARITY_MATRIX_FILENAME_TEMPLATE"]
# display(SIMILARITY_MATRIX_FILENAME_TEMPLATE)

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_DIR = DATASET_CONFIG["CLUSTERING_DIR"]
display(INPUT_DIR)
assert INPUT_DIR.exists()

# %% tags=[]
OUTPUT_FILE = DATASET_CONFIG["CLUSTERING_COMBINED_FILE"]
display(OUTPUT_FILE)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# %% [markdown] tags=[]
# # Get data files

# %% tags=[]
filename_pattern = re.compile(DATASET_CONFIG["CLUSTERING_FILENAME_PATTERN"])

# %% tags=[]
# get input data files according to Settings
input_files = sorted(
    [
        f
        for f in INPUT_DIR.iterdir()
        if (m := re.search(filename_pattern, str(f))) is not None
        #         and m.group("corr_method") == CORRELATION_METHOD_NAME
    ]
)
display(len(input_files))
display(input_files[:5])

assert len(input_files) > 0

# %% [markdown] tags=[]
# ## Preview data

# %% tags=[]
display(input_files[0])

# %% tags=[]
_tmp_df = pd.read_pickle(input_files[0])

# %% tags=[]
_tmp_df.shape

# %% tags=[]
_tmp_df.sample(n=5, random_state=0)

# %% [markdown] tags=[]
# # Run

# %% [markdown] tags=[]
# ## Read data, convert dtypes, add new metrics

# %% tags=[]
all_results = []

for f_full in tqdm(input_files, ncols=100):
    f_name = f_full.name

    f_data = pd.read_pickle(f_full).reset_index()

    # add metadata
    metadata = re.search(filename_pattern, f_name)

    f_data = f_data[
        [
            "id",
            "n_clusters",
            "partition",
            "si_score",
        ]
    ]

    f_data["tissue"] = metadata.group("tissue")
    f_data["gene_sel_strategy"] = metadata.group("gene_sel_strategy")
    f_data["corr_method"] = metadata.group("corr_method")
    f_data["clust_method"] = metadata.group("clust_method")

    all_results.append(f_data)

# %%
df = pd.concat(all_results, ignore_index=True)

# %%
df.shape

# %%
df.head()

# %%
df.dtypes

# %%
# convert to int32
df["n_clusters"] = df["n_clusters"].astype("int32")

# to category dtype
df["id"] = df["id"].astype("category")
df["tissue"] = df["tissue"].astype("category")
df["gene_sel_strategy"] = df["gene_sel_strategy"].astype("category")
df["corr_method"] = df["corr_method"].astype("category")
df["clust_method"] = df["clust_method"].astype("category")

# %%
display(df.dtypes)
assert df.dtypes.loc["id"] == "category"

# %%
df.iloc[0]["partition"]

# %% tags=[]
df.sample(n=5)

# %% [markdown] tags=[]
# ## Some stats

# %% tags=[]
df["n_clusters"].unique()

# %% tags=[]
df["tissue"].unique()

# %% tags=[]
df["gene_sel_strategy"].unique()

# %% tags=[]
df["corr_method"].unique()

# %% tags=[]
df["clust_method"].unique()

# %% [markdown] tags=[]
# ## Testing

# %% tags=[]
assert not df.isna().any().any()

# %% [markdown] tags=[]
# # Save

# %% tags=[]
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

saveRDS = ro.r["saveRDS"]
readRDS = ro.r["readRDS"]

# %% tags=[]
data = df

# %% [markdown] tags=[]
# ## Pickle

# %% tags=[]
display(OUTPUT_FILE)

# %% tags=[]
data.to_pickle(OUTPUT_FILE)

# %% [markdown] tags=[]
# ## RDS

# %% tags=[]
# output_rds_file = OUTPUT_FILE.with_suffix(".rds")
# display(output_rds_file)

# %% tags=[]
# with localconverter(ro.default_converter + pandas2ri.converter):
# #     data["partition"] = data["partition"].apply(lambda x: ro.IntVector(x.tolist()))
#     data_r = ro.conversion.py2rpy(data)

# %% tags=[]
# data_r

# %% tags=[]
# saveRDS(data_r, str(output_rds_file))

# %% tags=[]
# # testing
# data_r = readRDS(str(output_rds_file))

# %% tags=[]
# with localconverter(ro.default_converter + pandas2ri.converter):
#     data_again = ro.conversion.rpy2py(data_r)

#     # convert index to int, otherwise it's converted to string
#     data_again.index = data_again.index.astype(int)

# %% tags=[]
# data_again.shape

# %% tags=[]
# data_again.head()

# %% tags=[]
# pd.testing.assert_frame_equal(
#     data,
#     data_again,
#     check_names=False,  # do not check "name" attribute of index and column
#     check_exact=True,  # since this is a binary format, it should match exactly
# )

# %% [markdown] tags=[]
# ## tsv.gz

# %% tags=[]
output_text_file = OUTPUT_FILE.with_suffix(".tsv.gz")
display(output_text_file)

# %%
data_text = data.copy()
data_text["partition"] = data_text["partition"].apply(lambda x: repr(x.tolist()))

# %% tags=[]
data_text.to_csv(output_text_file, sep="\t", index=False, float_format="%.5e")

# %% tags=[]
# testing
data_again = pd.read_csv(output_text_file, sep="\t")  # , index_col=0)
data_again["partition"] = data_again["partition"].apply(
    lambda x: np.array(eval(x), dtype="int32")
)

# %% tags=[]
data_again.shape

# %% tags=[]
data_again.head()

# %% tags=[]
pd.testing.assert_frame_equal(
    data,
    data_again,
    check_names=False,  # do not check "name" attribute of index and column
    check_dtype=False,  # do not check dtypes: do not distinguish between int64 and int32, for instance
    check_categorical=False,
    check_exact=False,
    rtol=1e-5,
    atol=5e-5,
)

# %% tags=[]
