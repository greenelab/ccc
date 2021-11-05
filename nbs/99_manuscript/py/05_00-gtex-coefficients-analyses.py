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
GTEX_TISSUE = "whole_blood"

# %%
GENE_SEL_STRATEGY = "var_pc_log2"

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
INPUT_GENE_EXPR_DATA_FILE_TEMPLATE = (
    DATASET_CONFIG["GENE_SELECTION_DIR"]
    / f"gtex_v8_data_{GTEX_TISSUE}-{GENE_SEL_STRATEGY}.pkl"
)

display(INPUT_GENE_EXPR_DATA_FILE_TEMPLATE)
# assert INPUT_FILE.exists()

# %% tags=[]
INPUT_FILE_TEMPLATE = (
    DATASET_CONFIG["SIMILARITY_MATRICES_DIR"]
    / DATASET_CONFIG["SIMILARITY_MATRIX_FILENAME_TEMPLATE"]
)

display(INPUT_FILE_TEMPLATE)
# assert INPUT_FILE.exists()

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## Gene Ensembl ID -> Symbol mapping

# %%
gene_map = pd.read_pickle(
    DATASET_CONFIG["DATA_DIR"] / "gtex_gene_id_symbol_mappings.pkl"
)

# %%
gene_map = gene_map.set_index("gene_ens_id")["gene_symbol"].to_dict()

# %%
assert gene_map["ENSG00000145309.5"] == "CABS1"

# %% [markdown] tags=[]
# ## Gene expression

# %%
data = pd.read_pickle(INPUT_GENE_EXPR_DATA_FILE_TEMPLATE)

# %%
data.shape

# %%
data.head()

# %% [markdown] tags=[]
# ## Clustermatch

# %%
clustermatch_df = pd.read_pickle(
    str(INPUT_FILE_TEMPLATE).format(
        tissue=GTEX_TISSUE,
        gene_sel_strategy=GENE_SEL_STRATEGY,
        corr_method="clustermatch",
    )
)

# %%
clustermatch_df.shape

# %%
clustermatch_df.head()

# %%
assert data.index.equals(clustermatch_df.index)

# %% [markdown] tags=[]
# ## Pearson

# %%
pearson_df = pd.read_pickle(
    str(INPUT_FILE_TEMPLATE).format(
        tissue=GTEX_TISSUE,
        gene_sel_strategy=GENE_SEL_STRATEGY,
        corr_method="pearson",
    )
)

# %%
pearson_df.shape

# %%
pearson_df.head()

# %%
assert data.index.equals(pearson_df.index)

# %% [markdown] tags=[]
# ## Spearman

# %%
spearman_df = pd.read_pickle(
    str(INPUT_FILE_TEMPLATE).format(
        tissue=GTEX_TISSUE,
        gene_sel_strategy=GENE_SEL_STRATEGY,
        corr_method="spearman",
    )
)

# %%
spearman_df.shape

# %%
spearman_df.head()

# %%
assert data.index.equals(spearman_df.index)


# %% [markdown] tags=[]
# ## Merge

# %%
def get_upper_triag(data, k=1):
    mask = np.triu(np.ones(data.shape), k=k).astype(bool)
    return data.where(mask)


# %%
# make sure genes match
clustermatch_df = clustermatch_df.loc[pearson_df.index, pearson_df.columns]

# %%
clustermatch_df = get_upper_triag(clustermatch_df)

# %%
clustermatch_df = clustermatch_df.unstack().rename_axis((None, None)).dropna()

# %%
clustermatch_df.shape

# %%
clustermatch_df.head()

# %%
pearson_df = get_upper_triag(pearson_df)

# %%
# make pearson abs
pearson_df = pearson_df.unstack().rename_axis((None, None)).dropna().abs()

# %%
pearson_df.shape

# %%
pearson_df.head()

# %%
assert clustermatch_df.index.equals(pearson_df.index)

# %%
spearman_df = get_upper_triag(spearman_df)

# %%
# make spearman abs
spearman_df = spearman_df.unstack().rename_axis((None, None)).dropna().abs()

# %%
spearman_df.shape

# %%
spearman_df.head()

# %%
assert clustermatch_df.index.equals(spearman_df.index)

# %%
df = pd.DataFrame(
    {
        "clustermatch": clustermatch_df,
        "pearson": pearson_df,
        "spearman": spearman_df,
    }
)

# %%
df.shape

# %%
df.head()

# %%
from sklearn.preprocessing import minmax_scale

# %%
df = df.assign(clustermatch_std=minmax_scale(df.clustermatch.to_numpy()))
df = df.assign(pearson_std=minmax_scale(df.pearson.to_numpy()))
df = df.assign(spearman_std=minmax_scale(df.spearman.to_numpy()))

# %%
df.head()

# %% [markdown]
# # QQ-plot

# %% [markdown]
# ## With standardized values

# %%
CLUSTERMATCH_COLUMN = "clustermatch_std"
PEARSON_COLUMN = "pearson_std"
SPEARMAN_COLUMN = "spearman_std"

# %%
QUANTILES = np.linspace(0, 1, 10000)
display(QUANTILES[:10])
display(QUANTILES[-10:])

# %%
quantiles_df = pd.DataFrame(
    {
        "clustermatch": df[CLUSTERMATCH_COLUMN].quantile(QUANTILES).to_numpy(),
        "pearson": df[PEARSON_COLUMN].quantile(QUANTILES).to_numpy(),
        "spearman": df[SPEARMAN_COLUMN].quantile(QUANTILES).to_numpy(),
    }
)

# %%
quantiles_df.shape

# %%
quantiles_df.head()

# %%
quantiles_df.describe()

# %%
fig, ax = plt.subplots(figsize=(10, 8))

sns.scatterplot(
    data=quantiles_df,
    x="pearson",
    y="clustermatch",
    label="vs Pearson (abs)",
    ax=ax,
)

sns.scatterplot(
    data=quantiles_df,
    x="spearman",
    y="clustermatch",
    label="vs Spearman (abs)",
    ax=ax,
)

ax.set_xlabel(None)
if "_k2" in CLUSTERMATCH_COLUMN:
    ax.set_ylabel("clustermatch (linear)")

min_val = min((quantiles_df.iloc[:, 0].min(), quantiles_df.iloc[:, 1].min()))
max_val = max((quantiles_df.iloc[:, 0].max(), quantiles_df.iloc[:, 1].max()))
ax.plot([min_val, max_val], [min_val, max_val], "k", linewidth=0.5)

# ax.set_title(f"{ENRICH_FUNC} ({PERFORMANCE_MEASURE})")

# %% [markdown]
# Pearson/Spearman values seem to be more variable in the range (0, 1) than clustermatch

# %% [markdown]
# ## With original values

# %%
CLUSTERMATCH_COLUMN = "clustermatch"
PEARSON_COLUMN = "pearson"
SPEARMAN_COLUMN = "spearman"

# %%
QUANTILES = np.linspace(0, 1, 10000)
display(QUANTILES[:10])
display(QUANTILES[-10:])

# %%
quantiles_df = pd.DataFrame(
    {
        "clustermatch": df[CLUSTERMATCH_COLUMN].quantile(QUANTILES).to_numpy(),
        "pearson": df[PEARSON_COLUMN].quantile(QUANTILES).to_numpy(),
        "spearman": df[SPEARMAN_COLUMN].quantile(QUANTILES).to_numpy(),
    }
)

# %%
quantiles_df.shape

# %%
quantiles_df.head()

# %%
quantiles_df.describe()

# %%
fig, ax = plt.subplots(figsize=(10, 8))

sns.scatterplot(
    data=quantiles_df,
    x="pearson",
    y="clustermatch",
    label="vs Pearson (abs)",
    ax=ax,
)

sns.scatterplot(
    data=quantiles_df,
    x="spearman",
    y="clustermatch",
    label="vs Spearman (abs)",
    ax=ax,
)

ax.set_xlabel(None)
if "_k2" in CLUSTERMATCH_COLUMN:
    ax.set_ylabel("clustermatch (linear)")

min_val = min((quantiles_df.iloc[:, 0].min(), quantiles_df.iloc[:, 1].min()))
max_val = max((quantiles_df.iloc[:, 0].max(), quantiles_df.iloc[:, 1].max()))
ax.plot([min_val, max_val], [min_val, max_val], "k", linewidth=0.5)

# ax.set_title(f"{ENRICH_FUNC} ({PERFORMANCE_MEASURE})")

# %% [markdown]
# Same as other plot

# %% [markdown]
# # Density plot

# %%
df.head()

# %%
with sns.plotting_context("talk", font_scale=1.1):
    fig, ax = plt.subplots(figsize=(10, 7))

    for method in [x for x in df.columns if not x.endswith("_std")]:
        sns.distplot(x=df[method], hist=True, kde=False, label=method, ax=ax)

    plt.legend()

# %%
# sns.jointplot(
#     data=df,
#     x="pearson",
#     y="clustermatch",
#     kind="hist",
# )

# %%
sns.jointplot(
    data=df,
    x="pearson",
    y="clustermatch",
    kind="hex",
    bins="log",
)

# %%
sns.jointplot(
    data=df,
    x="spearman",
    y="clustermatch",
    kind="hex",
    bins="log",
)

# %%
sns.jointplot(
    data=df,
    x="spearman",
    y="pearson",
    kind="hex",
    bins="log",
)

# %%
df[["clustermatch", "pearson", "spearman"]].corr()

# %%
df[["clustermatch", "pearson", "spearman"]].corr("spearman")

# %%
# sns.jointplot(
#     data=df,
#     x="pearson_std",
#     y="clustermatch_std",
#     kind="hex",
#     bins="log",
# )

# %%
# sns.jointplot(
#     data=df,
#     x="pearson",
#     y="clustermatch",
#     kind="kde",
# )

# %% [markdown]
# # Look at differences in gene pairs

# %%
_q_diff = 0.30

clustermatch_lower_q = df["clustermatch"].quantile(_q_diff)
clustermatch_higher_q = df["clustermatch"].quantile(1.0 - _q_diff)
display((clustermatch_lower_q, clustermatch_higher_q))

pearson_lower_q = df["pearson"].quantile(_q_diff)
pearson_higher_q = df["pearson"].quantile(1.0 - _q_diff)
display((pearson_lower_q, pearson_higher_q))

spearman_lower_q = df["spearman"].quantile(_q_diff)
spearman_higher_q = df["spearman"].quantile(1.0 - _q_diff)
display((spearman_lower_q, spearman_higher_q))

# %% [markdown]
# ## Pearson higher

# %%
_tmp_df_pearson_higher = df[
    (df["clustermatch"] <= clustermatch_lower_q)
    & (df["pearson"] >= pearson_higher_q)
    #     & (df["spearman"] >= spearman_higher_q)
].sort_values("spearman", ascending=False)

display(_tmp_df_pearson_higher.shape)
display(_tmp_df_pearson_higher)

# %%
# clustermatch higher
_tmp_df_clustermatch_higher = df[
    (df["clustermatch"] >= clustermatch_higher_q)
    & (df["pearson"] <= pearson_lower_q)
    #     & (df["spearman"] <= spearman_lower_q)
].sort_values("clustermatch", ascending=False)

display(_tmp_df_clustermatch_higher.shape)
display(_tmp_df_clustermatch_higher)

# %% [markdown] tags=[]
# ### Plot

# %%
gene0, gene1 = _tmp_df_pearson_higher.iloc[0].name
display((gene0, gene1))

gene0_symbol, gene1_symbol = gene_map[gene0], gene_map[gene1]
display((gene0_symbol, gene1_symbol))

_pearson, _spearman, _clustermatch = df.loc[
    (gene0, gene1), ["pearson", "spearman", "clustermatch"]
].tolist()

# %%
_title = f"Clustermatch: {_clustermatch:.2f}\nPearson/Spearman: {_pearson:.2f}/{_spearman:.2f}"

# %%
p = sns.jointplot(
    data=data.T,
    x=gene0,
    y=gene1,
    kind="hex",
    bins="log",
)

gene_x_id = p.ax_joint.get_xlabel()
gene_x_symbol = gene_map[gene_x_id]
p.ax_joint.set_xlabel(f"{gene_x_id}\n{gene_x_symbol}")

gene_y_id = p.ax_joint.get_ylabel()
gene_y_symbol = gene_map[gene_y_id]
p.ax_joint.set_ylabel(f"{gene_y_id}\n{gene_y_symbol}")

p.fig.suptitle(_title)

# %%
_tmp = data.T[[gene0, gene1]]
display(_tmp.shape)
display(_tmp.describe())

_tmp_q = _tmp.quantile([0.90, 0.95, 0.97, 0.98, 0.99])
display(_tmp_q)

# %%
_tmp_sel = _tmp[
    (_tmp[gene0] < _tmp_q.loc[0.97, gene0]) & (_tmp[gene1] < _tmp_q.loc[0.97, gene1])
]
display(_tmp_sel.shape)

display("Pearson again:")
display(_tmp_sel.corr())

display("Spearman again:")
display(_tmp_sel.corr("spearman"))

# %%
df["pearson"].quantile(np.linspace(0.30, 0.71, 10))

# %%
df["spearman"].quantile(np.linspace(0.30, 0.71, 10))

# %% [markdown]
# Some outliers maybe? are driving most of the correlation. If we remove the top 1% of samples, then Pearson is not at the top 30% anymore.

# %% [markdown]
# ## Spearman higher

# %%
_tmp_df_spearman_higher = df[
    (df["clustermatch"] <= clustermatch_lower_q)
    #     & (df["pearson"] >= pearson_higher_q)
    & (df["spearman"] >= spearman_higher_q)
].sort_values("spearman", ascending=False)

display(_tmp_df_spearman_higher.shape)
display(_tmp_df_spearman_higher)

# %%
# clustermatch higher
_tmp_df_clustermatch_higher = df[
    (df["clustermatch"] >= clustermatch_higher_q)
    #     & (df["pearson"] <= pearson_lower_q)
    & (df["spearman"] <= spearman_lower_q)
].sort_values("clustermatch", ascending=False)

display(_tmp_df_clustermatch_higher.shape)
display(_tmp_df_clustermatch_higher)

# %% [markdown] tags=[]
# ### Plot

# %% [markdown]
# ## Clustermatch higher

# %%
# TODO: this is higher than spearman
gene0, gene1 = _tmp_df_clustermatch_higher.iloc[1].name
display((gene0, gene1))

gene0_symbol, gene1_symbol = gene_map[gene0], gene_map[gene1]
display((gene0_symbol, gene1_symbol))

_pearson, _spearman, _clustermatch = df.loc[
    (gene0, gene1), ["pearson", "spearman", "clustermatch"]
].tolist()

# %%
_title = f"Clustermatch: {_clustermatch:.2f}\nPearson/Spearman: {_pearson:.2f}/{_spearman:.2f}"

# %%
p = sns.jointplot(
    data=data.T,
    x=gene0,
    y=gene1,
    kind="hex",
    bins="log",
)

gene_x_id = p.ax_joint.get_xlabel()
gene_x_symbol = gene_map[gene_x_id]
p.ax_joint.set_xlabel(f"{gene_x_id}\n{gene_x_symbol}")

gene_y_id = p.ax_joint.get_ylabel()
gene_y_symbol = gene_map[gene_y_id]
p.ax_joint.set_ylabel(f"{gene_y_id}\n{gene_y_symbol}")

p.fig.suptitle(_title)

# %%
_tmp = data.T[[gene0, gene1]]
display(_tmp.shape)
display(_tmp.describe())

# %%
