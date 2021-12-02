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
import pandas as pd
import numpy as np

# from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.preprocessing import minmax_scale

from clustermatch import conf
from clustermatch.coef import cm

# %% [markdown] tags=[]
# # Settings

# %%
DATASET_CONFIG = conf.GTEX
GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None and conf.MANUSCRIPT["BASE_DIR"].exists()
), "Manuscript dir not set"

# %% tags=[]
OUTPUT_FIGURE_DIR = conf.MANUSCRIPT["FIGURES_DIR"] / "coefs_comp"
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% tags=[]
INPUT_GENE_EXPR_FILE = (
    DATASET_CONFIG["GENE_SELECTION_DIR"]
    / f"gtex_v8_data_{GTEX_TISSUE}-{GENE_SEL_STRATEGY}.pkl"
)
display(INPUT_GENE_EXPR_FILE)

assert INPUT_GENE_EXPR_FILE.exists()

# %% tags=[]
INPUT_CORR_FILE_TEMPLATE = (
    DATASET_CONFIG["SIMILARITY_MATRICES_DIR"]
    / DATASET_CONFIG["SIMILARITY_MATRIX_FILENAME_TEMPLATE"]
)
display(INPUT_CORR_FILE_TEMPLATE)

# %% tags=[]
INPUT_CORR_FILE = DATASET_CONFIG["SIMILARITY_MATRICES_DIR"] / str(
    INPUT_CORR_FILE_TEMPLATE
).format(
    tissue=GTEX_TISSUE,
    gene_sel_strategy=GENE_SEL_STRATEGY,
    corr_method="all",
)
display(INPUT_CORR_FILE)

assert INPUT_CORR_FILE.exists()

# %% [markdown] tags=[]
# # Data

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
gene_expr_df = pd.read_pickle(INPUT_GENE_EXPR_FILE)

# %%
gene_expr_df.shape

# %%
gene_expr_df.head()

# %% [markdown] tags=[]
# ## Correlation

# %%
df = pd.read_pickle(INPUT_CORR_FILE)

# %%
df.shape

# %%
df.head()

# %%
df.apply(lambda x: x.quantile(np.linspace(0.20, 1.0, 20)))

# %% [markdown]
# # Intersection plot

# %%
from upsetplot import plot, from_indicators


# %% [markdown]
# ## Prepare data

# %%
def get_lower_upper_quantile(method_name, q):
    return df[method_name].quantile([q, 1 - q])


# %%
# test
_tmp = get_lower_upper_quantile("clustermatch", 0.20)
display(_tmp)

_tmp0, _tmp1 = _tmp
display((_tmp0, _tmp1))

assert _tmp0 == _tmp.iloc[0]
assert _tmp1 == _tmp.iloc[1]

# %%
# TODO: move this to Settings
_q_diff = 0.30

clustermatch_lq, clustermatch_hq = get_lower_upper_quantile("clustermatch", _q_diff)
display((clustermatch_lq, clustermatch_hq))

pearson_lq, pearson_hq = get_lower_upper_quantile("pearson", _q_diff)
display((pearson_lq, pearson_hq))

spearman_lq, spearman_hq = get_lower_upper_quantile("spearman", _q_diff)
display((spearman_lq, spearman_hq))

# %%
pearson_higher = df["pearson"] >= pearson_hq
display(pearson_higher.sum())

# %%
pearson_lower = df["pearson"] <= pearson_lq
display(pearson_lower.sum())

# %%
spearman_higher = df["spearman"] >= spearman_hq
display(spearman_higher.sum())

# %%
spearman_lower = df["spearman"] <= spearman_lq
display(spearman_lower.sum())

# %%
clustermatch_higher = df["clustermatch"] >= clustermatch_hq
display(clustermatch_higher.sum())

# %%
clustermatch_lower = df["clustermatch"] <= clustermatch_lq
display(clustermatch_lower.sum())

# %% [markdown]
# ## Plot

# %%
df_plot = pd.DataFrame(
    {
        "pearson_higher": pearson_higher,
        "pearson_lower": pearson_lower,
        "spearman_higher": spearman_higher,
        "spearman_lower": spearman_lower,
        "clustermatch_higher": clustermatch_higher,
        "clustermatch_lower": clustermatch_lower,
    }
)

# %%
df_plot = pd.concat([df_plot, df], axis=1)

# %%
df_plot

# %%
categories = sorted(
    [x for x in df_plot.columns if "_" in x],
    reverse=True,
    key=lambda x: x.split("_")[1] + "_" + x.split("_")[0],
)

# %%
categories

# %% [markdown]
# ## Python - UpSet

# %% [markdown]
# ### All subsets

# %%
df_r_data = df_plot

# %%
df_r_data.shape

# %%
gene_pairs_by_cats = from_indicators(categories, data=df_r_data)

# %%
gene_pairs_by_cats

# %%
fig = plt.figure(figsize=(18, 5))

plot(
    gene_pairs_by_cats,
    show_counts=True,
    sort_categories_by=None,
    # sort_by=None,
    # show_percentages=True,
    # min_subset_size=2,
    element_size=None,
    fig=fig,
)

# %% [markdown]
# ### Remove subsets of size one

# %%
# remove cases that are found only in one group
df_r_data = df_plot[df_plot[categories].sum(axis=1) > 1]
display(df_r_data.shape)

# %%
df_r_data.shape

# %%
gene_pairs_by_cats = from_indicators(categories, data=df_r_data)

# %%
fig = plt.figure(figsize=(15, 5))

plot(
    gene_pairs_by_cats,
    show_counts=True,
    sort_categories_by=None,
    # show_percentages=True,
    # min_subset_size=2,
    element_size=None,
    fig=fig,
)

# %% [markdown]
# ### Remove non-interesting subsets

# %%
lower_columns = [x for x in categories if x.endswith("_lower")]
display(lower_columns)

higher_columns = [x for x in categories if x.endswith("_higher")]
display(higher_columns)

# %%
df_r_data = df_plot[
    (df_plot[categories].sum(axis=1) > 1)
    & ~(
        (df_plot[lower_columns].sum(axis=1).isin((0, 3)))
        & (df_plot[higher_columns].sum(axis=1).isin((0, 3)))
    )
]

# %%
df_r_data.shape

# %%
gene_pairs_by_cats = from_indicators(categories, data=df_r_data)

# %%
fig = plt.figure(figsize=(17, 5))

plot(
    gene_pairs_by_cats,
    show_counts=True,
    sort_categories_by=None,
    # show_percentages=True,
    # min_subset_size=2,
    element_size=None,
    fig=fig,
)

# %% [markdown]
# ### Attemp with Casey's suggestions

# %%
df_r_data = df_plot

# %%
df_r_data.shape

# %%
gene_pairs_by_cats = from_indicators(categories, data=df_r_data)

# %%
gene_pairs_by_cats

# %%
gene_pairs_by_cats = gene_pairs_by_cats.sort_index()

# %%
_tmp_index = gene_pairs_by_cats.index.unique().to_frame(False)
display(_tmp_index)

# %%
_tmp_index[_tmp_index.sum(axis=1) == 3]

# %%
_tmp_index.apply(lambda x: x[0:3].sum() == 0, axis=1)

# %%
# agreements on top
_tmp_index.loc[
    _tmp_index[
        _tmp_index.apply(lambda x: x.sum() > 1, axis=1)
        & _tmp_index.apply(lambda x: x[0:3].sum() == 0, axis=1)
        & _tmp_index.apply(lambda x: 3 > x[3:].sum() > 1, axis=1)
    ].index
].apply(tuple, axis=1).to_numpy()

# %%
# agreements on bottom
_tmp_index.loc[
    _tmp_index[
        _tmp_index.apply(lambda x: x.sum() > 1, axis=1)
        & _tmp_index.apply(lambda x: 3 > x[0:3].sum() > 1, axis=1)
        & _tmp_index.apply(lambda x: x[3:].sum() == 0, axis=1)
    ].index
].apply(tuple, axis=1).to_numpy()

# %%
# diagreements
_tmp_index.loc[
    _tmp_index[
        _tmp_index.apply(lambda x: x.sum() > 1, axis=1)
        & _tmp_index.apply(lambda x: x[0:3].sum() > 0, axis=1)
        & _tmp_index.apply(lambda x: x[3:].sum() > 0, axis=1)
    ].index
].apply(tuple, axis=1).to_numpy()

# %%
# order subsets
gene_pairs_by_cats = gene_pairs_by_cats.loc[
    [
        # pairs not included in categories:
        # (False, False, False, False, False, False),
        # full agreements:
        (True, True, True, False, False, False),
        (False, False, False, True, True, True),
        # agreements on top
        (False, False, False, False, True, True),
        (False, False, False, True, False, True),
        (False, False, False, True, True, False),
        # agreements on bottom
        (False, True, True, False, False, False),
        (True, False, True, False, False, False),
        (True, True, False, False, False, False),
        # diagreements
        ## pearson
        (False, False, True, False, True, False),
        (True, False, False, False, True, False),
        (True, False, True, False, True, False),
        ## spearman
        (False, True, False, True, False, False),
        ## clustermatch
        (False, True, False, True, False, True),
        (False, True, False, False, False, True),
        (True, False, False, False, False, True),
        (True, True, False, False, False, True),
    ]
]

# %%
# assert gene_pairs_by_cats.shape[0] == df_r_data.shape[0]
# assert np.array_equal(gene_pairs_by_cats.index.unique(), _tmp_index)

# %%
fig = plt.figure(figsize=(18, 5))

plot(
    gene_pairs_by_cats,
    show_counts=True,
    sort_categories_by=None,
    sort_by=None,
    # show_percentages=True,
    # min_subset_size=2,
    element_size=None,
    fig=fig,
)


# %% [markdown]
# ## Look at specific cases

# %%
def plot_gene_pair(top_pairs_df, idx, bins="log"):
    gene0, gene1 = top_pairs_df.iloc[idx].name
    display((gene0, gene1))

    gene0_symbol, gene1_symbol = gene_map[gene0], gene_map[gene1]
    display((gene0_symbol, gene1_symbol))

    _pearson, _spearman, _clustermatch = top_pairs_df.loc[
        (gene0, gene1), ["pearson", "spearman", "clustermatch"]
    ].tolist()

    _title = f"Clustermatch: {_clustermatch:.2f}\nPearson/Spearman: {_pearson:.2f}/{_spearman:.2f}"

    p = sns.jointplot(
        data=gene_expr_df.T,
        x=gene0,
        y=gene1,
        kind="hex",
        bins=bins,
        # ylim=(0, 500),
    )

    gene_x_id = p.ax_joint.get_xlabel()
    gene_x_symbol = gene_map[gene_x_id]
    p.ax_joint.set_xlabel(f"{gene_x_id}\n{gene_x_symbol}")

    gene_y_id = p.ax_joint.get_ylabel()
    gene_y_symbol = gene_map[gene_y_id]
    p.ax_joint.set_ylabel(f"{gene_y_id}\n{gene_y_symbol}")

    p.fig.suptitle(_title)


# %% [markdown] tags=[]
# ### Clustermatch vs Spearman

# %%
_tmp_df = df_r_data[
    (df_r_data["clustermatch_higher"])
    & ~(df_r_data["spearman_higher"])
    & ~(df_r_data["pearson_higher"])
    & ~(df_r_data["clustermatch_lower"])
    & (df_r_data["spearman_lower"])
    & ~(df_r_data["pearson_lower"])
]

# show this just to make sure of the groups
# display(_tmp_df.head())

_tmp_df = _tmp_df[[x for x in df_r_data.columns if "_" not in x]].sort_values(
    "clustermatch", ascending=False
)

display(_tmp_df.shape)
display(_tmp_df)

# %%
plot_gene_pair(_tmp_df, 0)

# %% [markdown]
# UTY is from chr Y and KDM6A is from chr X, so males and females samples explain this relationship.

# %%
plot_gene_pair(_tmp_df, 1)

# %% [markdown]
# KIAA0040 (chr 1) and CYTIP (chr 2)

# %%
plot_gene_pair(_tmp_df, 2)

# %% [markdown]
# KIAA0040 (chr 1) and CYTIP (chr 2)

# %%
plot_gene_pair(_tmp_df, 9)

# %% [markdown] tags=[]
# ### Clustermatch vs Pearson

# %%
_tmp_df = df_r_data[
    (df_r_data["clustermatch_higher"])
    & ~(df_r_data["spearman_higher"])
    & ~(df_r_data["pearson_higher"])
    & ~(df_r_data["clustermatch_lower"])
    & ~(df_r_data["spearman_lower"])
    & (df_r_data["pearson_lower"])
]

# show this just to make sure of the groups
# display(_tmp_df.head())

_tmp_df = _tmp_df[[x for x in df_r_data.columns if "_" not in x]].sort_values(
    "clustermatch", ascending=False
)

display(_tmp_df.shape)
display(_tmp_df)

# %%
plot_gene_pair(_tmp_df, 0)

# %%
plot_gene_pair(_tmp_df, 1)

# %%
plot_gene_pair(_tmp_df, 2)

# %% [markdown] tags=[]
# ### Clustermatch vs Spearman/Pearson

# %%
_tmp_df = df_r_data[
    (df_r_data["clustermatch_higher"])
    & ~(df_r_data["spearman_higher"])
    & ~(df_r_data["pearson_higher"])
    & ~(df_r_data["clustermatch_lower"])
    & (df_r_data["spearman_lower"])
    & (df_r_data["pearson_lower"])
]

# show this just to make sure of the groups
# display(_tmp_df.head())

_tmp_df = _tmp_df[[x for x in df_r_data.columns if "_" not in x]].sort_values(
    "clustermatch", ascending=False
)

display(_tmp_df.shape)
display(_tmp_df)

# %%
plot_gene_pair(_tmp_df, 0)

# %%
plot_gene_pair(_tmp_df, 1)

# %%
plot_gene_pair(_tmp_df, 2)

# %%
plot_gene_pair(_tmp_df, 3)

# %%
plot_gene_pair(_tmp_df, 4)

# %%
plot_gene_pair(_tmp_df, 5)

# %%
plot_gene_pair(_tmp_df, 6)

# %%
plot_gene_pair(_tmp_df, 7)

# %% [markdown] tags=[]
# ### Clustermatch/Spearman vs Pearson

# %%
_tmp_df = df_r_data[
    (df_r_data["clustermatch_higher"])
    & (df_r_data["spearman_higher"])
    & ~(df_r_data["pearson_higher"])
    & ~(df_r_data["clustermatch_lower"])
    & ~(df_r_data["spearman_lower"])
    & (df_r_data["pearson_lower"])
]

# show this just to make sure of the groups
# display(_tmp_df.head())

_tmp_df = _tmp_df[[x for x in df_r_data.columns if "_" not in x]].sort_values(
    "clustermatch", ascending=False
)

display(_tmp_df.shape)
display(_tmp_df)

# %%
plot_gene_pair(_tmp_df, 0)

# %%
plot_gene_pair(_tmp_df, 1)

# %%
plot_gene_pair(_tmp_df, 2)

# %% [markdown] tags=[]
# ### Pearson vs Spearman

# %%
_tmp_df = df_r_data[
    ~(df_r_data["clustermatch_higher"])
    & ~(df_r_data["spearman_higher"])
    & (df_r_data["pearson_higher"])
    & ~(df_r_data["clustermatch_lower"])
    & (df_r_data["spearman_lower"])
    & ~(df_r_data["pearson_lower"])
]

# show this just to make sure of the groups
# display(_tmp_df.head())

_tmp_df = _tmp_df[[x for x in df_r_data.columns if "_" not in x]].sort_values(
    "pearson", ascending=False
)

display(_tmp_df.shape)
display(_tmp_df)

# %%
plot_gene_pair(_tmp_df, 0)

# %%
plot_gene_pair(_tmp_df, 1)

# %%
plot_gene_pair(_tmp_df, 2)

# %% [markdown] tags=[]
# ### Spearman vs Pearson

# %%
_tmp_df = df_r_data[
    ~(df_r_data["clustermatch_higher"])
    & (df_r_data["spearman_higher"])
    & ~(df_r_data["pearson_higher"])
    & ~(df_r_data["clustermatch_lower"])
    & ~(df_r_data["spearman_lower"])
    & (df_r_data["pearson_lower"])
]

# show this just to make sure of the groups
# display(_tmp_df.head())

_tmp_df = _tmp_df[[x for x in df_r_data.columns if "_" not in x]].sort_values(
    "pearson", ascending=False
)

display(_tmp_df.shape)
display(_tmp_df)

# %%
plot_gene_pair(_tmp_df, 0)

# %%
plot_gene_pair(_tmp_df, 1)

# %% [markdown] tags=[]
# ### Pearson vs Spearman/Clustermatch

# %%
_tmp_df = df_r_data[
    ~(df_r_data["clustermatch_higher"])
    & ~(df_r_data["spearman_higher"])
    & (df_r_data["pearson_higher"])
    & (df_r_data["clustermatch_lower"])
    & (df_r_data["spearman_lower"])
    & ~(df_r_data["pearson_lower"])
]

# show this just to make sure of the groups
# display(_tmp_df.head())

_tmp_df = _tmp_df[[x for x in df_r_data.columns if "_" not in x]].sort_values(
    "pearson", ascending=False
)

display(_tmp_df.shape)
display(_tmp_df)

# %%
plot_gene_pair(_tmp_df, 0)

# %%
plot_gene_pair(_tmp_df, 1)

# %% [markdown] tags=[]
# ### Pearson vs Clustermatch

# %%
_tmp_df = df_r_data[
    ~(df_r_data["clustermatch_higher"])
    & ~(df_r_data["spearman_higher"])
    & (df_r_data["pearson_higher"])
    & (df_r_data["clustermatch_lower"])
    & ~(df_r_data["spearman_lower"])
    & ~(df_r_data["pearson_lower"])
]

# show this just to make sure of the groups
# display(_tmp_df.head())

_tmp_df = _tmp_df[[x for x in df_r_data.columns if "_" not in x]].sort_values(
    "pearson", ascending=False
)

display(_tmp_df.shape)
display(_tmp_df)

# %%
plot_gene_pair(_tmp_df, 0)

# %%
gene0, gene1 = (
    gene_expr_df.loc["ENSG00000130598.15"].copy(),
    gene_expr_df.loc["ENSG00000177791.11"].copy(),
)

# %%
cm(gene0, gene1)

# %%
q = 0.75
gene0[gene0 <= gene0.quantile(q)] = 0
gene0[gene0 > gene0.quantile(q)] = 1

gene1[gene1 <= gene1.quantile(q)] = 0
gene1[gene1 > gene1.quantile(q)] = 1

cm(gene0, gene1)

# %%
plot_gene_pair(_tmp_df, 500)

# %%
gene0, gene1 = (
    gene_expr_df.loc["ENSG00000177409.11"].copy(),
    gene_expr_df.loc["ENSG00000149131.15"].copy(),
)

# %%
cm(gene0, gene1)

# %%
q = 0.75
gene0[gene0 <= gene0.quantile(q)] = 0
gene0[gene0 > gene0.quantile(q)] = 1

gene1[gene1 <= gene1.quantile(q)] = 0
gene1[gene1 > gene1.quantile(q)] = 1

cm(gene0, gene1)

# %%
