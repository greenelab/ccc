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
DATASET_CONFIG = conf.RECOUNT2

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
# SIMILARITY_MATRICES_DIR = conf.GTEX["SIMILARITY_MATRICES_DIR"]
# display(SIMILARITY_MATRICES_DIR)

# %%
# SIMILARITY_MATRIX_FILENAME_TEMPLATE = conf.GTEX["SIMILARITY_MATRIX_FILENAME_TEMPLATE"]
# display(SIMILARITY_MATRIX_FILENAME_TEMPLATE)

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_FILE = DATASET_CONFIG["CLUSTERING_COMBINED_FILE"]
display(INPUT_FILE)
assert INPUT_FILE.exists()

# %% [markdown] tags=[]
# # Load data

# %%
df = pd.read_pickle(INPUT_FILE)

# %%
df.shape

# %%
df["corr_method"].unique()

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    tmp = df.head()
    display(tmp)

# %% [markdown]
# # SI score by method

# %%
# plot_data = df.groupby(["n_clusters", "corr_method", "gene_sel_strategy", "clust_method"])["si_score"].mean().reset_index()

# %%
# plot_data.shape

# %%
# plot_data.sort_values(["n_clusters", "corr_method"]).head(20)

# %%
selected_corr_methods = [
    "clustermatch_k2to5",
    "clustermatch_k2",
    "spearman_abs",
    "pearson_abs",
]

plot_data = df[
    (np.ones(df.shape[0]).astype(bool)) & (df.corr_method.isin(selected_corr_methods))
]

# %%
plot_data.shape

# %%
plot_data.corr_method.unique()

# %%
PERFORMANCE_MEASURE = "si_score"
# PERFORMANCE_MEASURE = "rich_factor"
# PERFORMANCE_MEASURE = "fold_enrich"

# %%
# fig, ax = plt.subplots(figsize=(10, 8))

sns.catplot(
    data=plot_data,
    x="n_clusters",
    y="si_score",
    hue="corr_method",
    hue_order=selected_corr_methods,
    kind="point",
    height=5,
    aspect=2,
    #     ax=ax,
)

# ax.set_xlabel(None)
# # ax.set_ylabel(None)

# min_val = min((quantiles_df.iloc[:, 0].min(), quantiles_df.iloc[:, 1].min()))
# max_val = max((quantiles_df.iloc[:, 0].max(), quantiles_df.iloc[:, 1].max()))
# ax.plot([min_val, max_val], [min_val, max_val], "k", linewidth=0.5)

# ax.set_title(f"Gene Ontology ({PERFORMANCE_MEASURE})")

# %% [markdown]
# # Size of clusters

# %%
# plot_data = df.groupby(["n_clusters", "corr_method", "gene_sel_strategy", "clust_method"])["si_score"].mean().reset_index()

# %%
# plot_data.shape

# %%
# plot_data.sort_values(["n_clusters", "corr_method"]).head(20)

# %%
plot_data = df[
    (np.ones(df.shape[0]).astype(bool)) & (df.corr_method.isin(selected_corr_methods))
]

# %%
plot_data.shape

# %%
plot_data.head()

# %%
pd.Series(plot_data.iloc[0]["partition"]).value_counts().to_dict()

# %%
from scipy.stats import entropy


# %%
def _get_partition_stats(part):
    return pd.Series(part).value_counts().to_numpy()


def _get_max_entropy(part_stats):
    n_clusters = len(part_stats)
    return entropy([1 / n_clusters for i in range(n_clusters)])


def _compute_score(part_stats):
    return entropy(part_stats) / _get_max_entropy(part_stats)


plot_data = plot_data.assign(
    partition_stats=plot_data["partition"].apply(_get_partition_stats)
)
plot_data = plot_data.assign(
    cluster_score=plot_data["partition_stats"].apply(_compute_score)
)

# %%
entropy([4364, 636])

# %%
entropy([4960, 40])

# %%
entropy([1 / 2, 1 / 2])

# %%
entropy([2500, 2500])

# %%
entropy([4364, 636])

# %%
_get_max_entropy([4364, 636])

# %%
_compute_score([4364, 636])

# %%
plot_data["cluster_score"].describe()

# %%
# plot_data.sort_values(["n_clusters", "corr_method"]).head(20)

# %%
# fig, ax = plt.subplots(figsize=(10, 8))

sns.catplot(
    data=plot_data,
    x="n_clusters",
    y="cluster_score",
    hue="corr_method",
    hue_order=selected_corr_methods,
    kind="point",
    height=5,
    aspect=2,
    #     ax=ax,
)

# ax.set_xlabel(None)
# # ax.set_ylabel(None)

# min_val = min((quantiles_df.iloc[:, 0].min(), quantiles_df.iloc[:, 1].min()))
# max_val = max((quantiles_df.iloc[:, 0].max(), quantiles_df.iloc[:, 1].max()))
# ax.plot([min_val, max_val], [min_val, max_val], "k", linewidth=0.5)

# ax.set_title(f"Gene Ontology ({PERFORMANCE_MEASURE})")

# %%
