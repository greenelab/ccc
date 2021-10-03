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

# %% tags=[]
import re

import numpy as np
import pandas as pd
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

# %%
# # clusterProfiler settings
# ENRICH_FUNCTION = "enrichGO"
# SIMPLIFY_CUTOFF = 0.7
# GO_ONTOLOGIES = ("BP", "CC", "MF")

# %%
# SIMILARITY_MATRICES_DIR = DATASET_CONFIG["SIMILARITY_MATRICES_DIR"]
# display(SIMILARITY_MATRICES_DIR)

# %%
# SIMILARITY_MATRIX_FILENAME_TEMPLATE = DATASET_CONFIG["SIMILARITY_MATRIX_FILENAME_TEMPLATE"]
# display(SIMILARITY_MATRIX_FILENAME_TEMPLATE)

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_FILE = DATASET_CONFIG["GENE_ENRICHMENT_COMBINED_FILE"]
display(INPUT_FILE)
assert INPUT_FILE.exists()

# %% [markdown] tags=[]
# # Load data

# %%
df = pd.read_pickle(INPUT_FILE)

# %%
df.shape

# %%
df.columns

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    tmp = df.head()
    display(tmp)

# %% [markdown]
# # QQ plot

# %%
# CLUSTERMATCH_METHOD = "clustermatch_k2"
CLUSTERMATCH_METHOD = "clustermatch"

# %%
# PERFORMANCE_MEASURE = "fdr"
# PERFORMANCE_MEASURE = "rich_factor"
PERFORMANCE_MEASURE = "fold_enrich"

# %%
QUANTILES = np.linspace(0, 1, 10000)
display(QUANTILES[:10])
display(QUANTILES[-10:])

# %%
df["enrich_params"].unique()

# %%
df_subset = df[
    (np.ones(df.shape[0]).astype(bool))
    & (df.fdr < 0.05)  # only significant results
    #     & (df.tissue == "adipose_subcutaneous")
    & (df.gene_sel_strategy == "var_pc_log2")
    & (df.clust_method == "SpectralClustering")
    & (df.enrich_func == "enrichGO")
    & (df.enrich_params.str.contains("_full"))
]

# %%
assert df_subset["fdr"].max() < 0.05

# %%
df_subset.shape

# %%
df_subset.head()

# %%
df_methods = df_subset["corr_method"].unique()
display(df_methods)

# %%
results_per_method = {}

for m in df_methods:
    df_values = df_subset[df_subset.corr_method == m][PERFORMANCE_MEASURE]
    display(f"{m} - {df_values.shape[0]}")

    if PERFORMANCE_MEASURE == "fdr":
        df_values = -np.log10(df_values)

    results_per_method[m] = df_values.quantile(QUANTILES).to_numpy()

# %%
quantiles_df = pd.DataFrame(results_per_method)

# %%
quantiles_df.shape

# %%
quantiles_df.head()

# %%
quantiles_df.tail()

# %%
quantiles_df.describe()

# %%
fig, ax = plt.subplots(figsize=(10, 8))

sns.scatterplot(
    data=quantiles_df,
    x="pearson_full",
    y=CLUSTERMATCH_METHOD,
    label="vs Pearson (full)",
    ax=ax,
)

sns.scatterplot(
    data=quantiles_df,
    x="spearman_full",
    y=CLUSTERMATCH_METHOD,
    label="vs Spearman (full)",
    ax=ax,
)

ax.set_xlabel(None)
# ax.set_ylabel(None)

min_val = min((quantiles_df.iloc[:, 0].min(), quantiles_df.iloc[:, 1].min()))
max_val = max((quantiles_df.iloc[:, 0].max(), quantiles_df.iloc[:, 1].max()))
ax.plot([min_val, max_val], [min_val, max_val], "k", linewidth=0.5)

ax.set_title(f"Gene Ontology ({PERFORMANCE_MEASURE})")

# %%
fig, ax = plt.subplots(figsize=(10, 8))

sns.scatterplot(
    data=quantiles_df,
    x="pearson_abs",
    y=CLUSTERMATCH_METHOD,
    label="vs Pearson (abs)",
    ax=ax,
)

sns.scatterplot(
    data=quantiles_df,
    x="spearman_abs",
    y=CLUSTERMATCH_METHOD,
    label="vs Spearman (abs)",
    ax=ax,
)

ax.set_xlabel(None)
# ax.set_ylabel(None)

min_val = min((quantiles_df.iloc[:, 0].min(), quantiles_df.iloc[:, 1].min()))
max_val = max((quantiles_df.iloc[:, 0].max(), quantiles_df.iloc[:, 1].max()))
ax.plot([min_val, max_val], [min_val, max_val], "k", linewidth=0.5)

ax.set_title(f"Gene Ontology ({PERFORMANCE_MEASURE})")

# %% [markdown]
# **UPDATE**
#
# Clustermatch (multi pattern) outperforms pearson (linear and abs), although pearson find most significant associations towards the
# right end of the distribution.
#
# However, Clustermatch does not outperform spearman (monotonic and abs), which provides more significant results across the entire distribution.

# %%
