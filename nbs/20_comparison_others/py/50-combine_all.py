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
# It combines all coefficient values in one tissue (see `Settings` below) into a single dataframe for easier processing later.
#
# This notebook incorporates results using MIC, which was computed only in a subset of gene pairs due to its computational complexity.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
import pandas as pd

from clustermatch import conf
from clustermatch.utils import get_upper_triag

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
DATASET_CONFIG = conf.GTEX
# whole blood by default, but this is a parameters cells that can be changed when running papermill
GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"

METHOD_NAME = "mic"

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
COMPARISONS_DIR = DATASET_CONFIG["RESULTS_DIR"] / "comparison_others"
display(COMPARISONS_DIR)

# %% tags=[]
INPUT_DIR = COMPARISONS_DIR / METHOD_NAME
display(INPUT_DIR)

# %% tags=[]
INPUT_CORR_FILE_TEMPLATE = (
    DATASET_CONFIG["SIMILARITY_MATRICES_DIR"]
    / DATASET_CONFIG["SIMILARITY_MATRIX_FILENAME_TEMPLATE"]
)
display(INPUT_CORR_FILE_TEMPLATE)

# %% tags=[]
INPUT_CORR_FILE_TEMPLATE = (
    DATASET_CONFIG["SIMILARITY_MATRICES_DIR"]
    / DATASET_CONFIG["SIMILARITY_MATRIX_FILENAME_TEMPLATE"]
)
display(INPUT_CORR_FILE_TEMPLATE)

# %% tags=[]
OUTPUT_FILE = (
    COMPARISONS_DIR / f"gtex_v8_data_{GTEX_TISSUE}-{GENE_SEL_STRATEGY}-all.pkl"
)
display(OUTPUT_FILE)

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## Clustermatch

# %% tags=[]
clustermatch_df = pd.read_pickle(
    str(INPUT_CORR_FILE_TEMPLATE).format(
        tissue=GTEX_TISSUE,
        gene_sel_strategy=GENE_SEL_STRATEGY,
        corr_method="clustermatch",
    )
)

# %% tags=[]
clustermatch_df.shape

# %% tags=[]
clustermatch_df.head()

# %% tags=[]
clustermatch_df = get_upper_triag(clustermatch_df)

# %% tags=[]
clustermatch_df = (
    clustermatch_df.unstack()
    .rename_axis((None, None))
    .dropna()
    .sort_index()
    .rename("clustermatch")
)

# %% tags=[]
clustermatch_df.shape

# %% tags=[]
clustermatch_df.head()

# %% [markdown] tags=[]
# ## Pearson

# %% tags=[]
pearson_df = pd.read_pickle(
    str(INPUT_CORR_FILE_TEMPLATE).format(
        tissue=GTEX_TISSUE,
        gene_sel_strategy=GENE_SEL_STRATEGY,
        corr_method="pearson",
    )
)

# %% tags=[]
pearson_df.shape

# %% tags=[]
pearson_df.head()

# %% tags=[]
pearson_df = get_upper_triag(pearson_df)

# %% tags=[]
# make pearson abs
pearson_df = (
    pearson_df.unstack()
    .rename_axis((None, None))
    .dropna()
    .abs()
    .sort_index()
    .rename("pearson")
)

# %% tags=[]
pearson_df.shape

# %% tags=[]
pearson_df.head()

# %% [markdown] tags=[]
# ## Spearman

# %% tags=[]
spearman_df = pd.read_pickle(
    str(INPUT_CORR_FILE_TEMPLATE).format(
        tissue=GTEX_TISSUE,
        gene_sel_strategy=GENE_SEL_STRATEGY,
        corr_method="spearman",
    )
)

# %% tags=[]
spearman_df.shape

# %% tags=[]
spearman_df.head()

# %% tags=[]
spearman_df = get_upper_triag(spearman_df)

# %% tags=[]
# make spearman abs
spearman_df = (
    spearman_df.unstack()
    .rename_axis((None, None))
    .dropna()
    .abs()
    .sort_index()
    .rename("spearman")
)

# %% tags=[]
spearman_df.shape

# %% tags=[]
spearman_df.head()

# %% [markdown] tags=[]
# ## MIC

# %%
mic_all_df = (
    pd.read_pickle(
        INPUT_DIR
        / f"gtex_v8_data_{GTEX_TISSUE}-{GENE_SEL_STRATEGY}-all-gene_pairs-sample_0-mic.pkl"
    )
    .rename("mic")
    .to_frame()
)

# %%
mic_all_df["mic_subset"] = "all"

# %%
mic_all_df.shape

# %%
mic_agree_df = (
    pd.read_pickle(
        INPUT_DIR
        / f"gene_pair_intersections-gtex_v8-{GTEX_TISSUE}-{GENE_SEL_STRATEGY}-agreements_sample_0-mic.pkl"
    )
    .rename("mic")
    .to_frame()
)

# %%
mic_agree_df["mic_subset"] = "agree"

# %%
mic_agree_df.shape

# %%
mic_disagree_df = (
    pd.read_pickle(
        INPUT_DIR
        / f"gene_pair_intersections-gtex_v8-{GTEX_TISSUE}-{GENE_SEL_STRATEGY}-disagreements_sample_0-mic.pkl"
    )
    .rename("mic")
    .to_frame()
)

# %%
mic_disagree_df["mic_subset"] = "disagree"

# %%
mic_disagree_df.shape

# %%
mic_df = pd.concat([mic_all_df, mic_agree_df, mic_disagree_df], axis=0).sort_index()

# %%
mic_df

# %%
mic_df.index.is_unique

# %%
# indexes could not be unique (because one of the sames is from the entire universe of gene pairs)
mic_df = mic_df[~mic_df.index.duplicated(keep="first")]

# %%
mic_df.shape

# %% [markdown] tags=[]
# ## Checks

# %%
assert (
    clustermatch_df.index.intersection(mic_df.index).shape[0] == mic_df.index.shape[0]
)

# %%
assert pearson_df.index.intersection(mic_df.index).shape[0] == mic_df.index.shape[0]

# %%
assert spearman_df.index.intersection(mic_df.index).shape[0] == mic_df.index.shape[0]

# %% [markdown] tags=[]
# ## Merge

# %%
df = pd.concat(
    [clustermatch_df, pearson_df, spearman_df, mic_df], join="inner", axis=1
).sort_index()

# %%
display(df.shape)
assert df.shape[0] == mic_df.shape[0]

# %% tags=[]
assert not df.isna().any().any()

# %%
df.head()

# %% [markdown] tags=[]
# # Save

# %% tags=[]
df.to_pickle(OUTPUT_FILE)

# %% tags=[]
