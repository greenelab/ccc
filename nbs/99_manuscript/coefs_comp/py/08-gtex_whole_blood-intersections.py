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
# It analyzes how correlation coefficients intersect on different gene pairs. Basically, I take the top gene pairs with the maximum correlation coefficient according to Pearson, Spearman and Clustermatch, and also the equivalent set with the minimum coefficient values, and then compare how these sets intersect each other.
#
# After identifying different intersection sets, I plot some gene pairs to see what's being captured or not by each coefficient.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from upsetplot import plot, from_indicators

from clustermatch.plots import MyUpSet
from clustermatch import conf

# %% [markdown] tags=[]
# # Settings

# %%
DATASET_CONFIG = conf.GTEX
GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"

# %%
# this specificies the threshold to compare coefficients (see below).
# it basically takes the top Q_DIFF coefficient values for gene pairs
# and compare with the bottom Q_DIFF of the other coefficients
Q_DIFF = 0.30

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None and conf.MANUSCRIPT["BASE_DIR"].exists()
), "Manuscript dir not set"

# %% tags=[]
OUTPUT_FIGURE_DIR = (
    conf.MANUSCRIPT["FIGURES_DIR"] / "coefs_comp" / f"gtex_{GTEX_TISSUE}"
)
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
# FIXME: this will not be necessary in new runs of clustermatch on GTEx
df.loc[df["clustermatch"] < 0, "clustermatch"] = 0.0

# %%
df.describe()

# %%
# show quantiles
df.apply(lambda x: x.quantile(np.linspace(0.20, 1.0, 20)))


# %% [markdown]
# # Prepare data for plotting

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
clustermatch_lq, clustermatch_hq = get_lower_upper_quantile("clustermatch", Q_DIFF)
display((clustermatch_lq, clustermatch_hq))

pearson_lq, pearson_hq = get_lower_upper_quantile("pearson", Q_DIFF)
display((pearson_lq, pearson_hq))

spearman_lq, spearman_hq = get_lower_upper_quantile("spearman", Q_DIFF)
display((spearman_lq, spearman_hq))

# %%
pearson_higher = df["pearson"] >= pearson_hq
display(pearson_higher.sum())

pearson_lower = df["pearson"] <= pearson_lq
display(pearson_lower.sum())

# %%
spearman_higher = df["spearman"] >= spearman_hq
display(spearman_higher.sum())

spearman_lower = df["spearman"] <= spearman_lq
display(spearman_lower.sum())

# %%
clustermatch_higher = df["clustermatch"] >= clustermatch_hq
display(clustermatch_higher.sum())

clustermatch_lower = df["clustermatch"] <= clustermatch_lq
display(clustermatch_lower.sum())

# %% [markdown]
# # UpSet plot

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
assert not df_plot.isna().any().any()

# %%
df_plot = df_plot.rename(
    columns={
        "pearson_higher": "Pearson (high)",
        "pearson_lower": "Pearson (low)",
        "spearman_higher": "Spearman (high)",
        "spearman_lower": "Spearman (low)",
        "clustermatch_higher": "Clustermatch (high)",
        "clustermatch_lower": "Clustermatch (low)",
    }
)

# %%
categories = sorted(
    [x for x in df_plot.columns if " (" in x],
    reverse=True,
    key=lambda x: x.split(" (")[1] + " (" + x.split(" (")[0],
)

# %%
categories

# %% [markdown]
# ## All subsets (original full plot)

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

g = plot(
    gene_pairs_by_cats,
    show_counts=True,
    sort_categories_by=None,
    element_size=None,
    fig=fig,
)

# %% [markdown]
# ## Remove subsets of size one

# %% [markdown]
# I leave this part commented out just in case it is useful in the future.

# %%
# # remove cases that are found only in one group
# df_r_data = df_plot[df_plot[categories].sum(axis=1) > 1]
# display(df_r_data.shape)

# %%
# df_r_data.shape

# %%
# gene_pairs_by_cats = from_indicators(categories, data=df_r_data)

# %%
# fig = plt.figure(figsize=(15, 5))

# g = plot(
#     gene_pairs_by_cats,
#     show_counts=True,
#     sort_categories_by=None,
#     element_size=None,
#     fig=fig,
# )

# g["totals"].set_visible(False)

# %% [markdown]
# ## Remove non-interesting subsets

# %% [markdown]
# I leave this part commented out just in case it is useful in the future.

# %%
# lower_columns = [x for x in categories if x.endswith("_lower")]
# display(lower_columns)

# higher_columns = [x for x in categories if x.endswith("_higher")]
# display(higher_columns)

# %%
# df_r_data = df_plot[
#     (df_plot[categories].sum(axis=1) > 1)
#     & ~(
#         (df_plot[lower_columns].sum(axis=1).isin((0, 3)))
#         & (df_plot[higher_columns].sum(axis=1).isin((0, 3)))
#     )
# ]

# %%
# df_r_data.shape

# %%
# gene_pairs_by_cats = from_indicators(categories, data=df_r_data)

# %%
# fig = plt.figure(figsize=(17, 5))

# g = plot(
#     gene_pairs_by_cats,
#     show_counts=True,
#     sort_categories_by=None,
#     element_size=None,
#     fig=fig,
# )

# g["totals"].set_visible(False)

# %% [markdown]
# ## Sort by categories of subsets

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
        # full agreements on high:
        (False, False, False, True, True, True),
        # agreements on top
        (False, False, False, False, True, True),
        (False, False, False, True, False, True),
        (False, False, False, True, True, False),
        # agreements on bottom
        (False, True, True, False, False, False),
        (True, False, True, False, False, False),
        (True, True, False, False, False, False),
        # full agreements on low:
        (True, True, True, False, False, False),
        # diagreements
        ## clustermatch
        (False, True, False, True, False, True),
        (False, True, False, False, False, True),
        (True, False, False, False, False, True),
        (True, True, False, False, False, True),
        ## pearson
        (False, False, True, False, True, False),
        (True, False, False, False, True, False),
        (True, False, True, False, True, False),
        ## spearman
        (False, True, False, True, False, False),
    ]
]

# %%
fig = plt.figure(figsize=(14, 5))

# g = plot(
g = MyUpSet(
    gene_pairs_by_cats,
    show_counts=True,
    sort_categories_by=None,
    sort_by=None,
    show_percentages=True,
    # min_subset_size=2,
    element_size=None,
    # fig=fig,
).plot(fig)

g["totals"].remove()  # set_visible(False)

# display(fig.get_size_inches())
# fig.set_size_inches(12, 5)

plt.savefig(
    OUTPUT_FIGURE_DIR / "upsetplot.svg",
    bbox_inches="tight",
    facecolor="white",
)

# plt.margins(x=-0.4)

# %% [markdown]
# This plot has the sets that represent agreements on the left, and disagreements on the right. The plot shown here is not the final one for the manuscript.

# %% [markdown]
# # Look at specific gene pair cases

# %%
def plot_gene_pair(top_pairs_df, idx, bins="log"):
    gene0, gene1 = top_pairs_df.iloc[idx].name
    display((gene0, gene1))

    gene0_symbol, gene1_symbol = gene_map[gene0], gene_map[gene1]
    display((gene0_symbol, gene1_symbol))

    _pearson, _spearman, _clustermatch = top_pairs_df.loc[
        (gene0, gene1), ["pearson", "spearman", "clustermatch"]
    ].tolist()

    _title = f"$c={_clustermatch:.2f}$  $r={_pearson:.2f}$  $r_s={_spearman:.2f}$"

    # displot DOES SUPPORT HUE!
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

    return p


# %%
# add columns with ranks
df_r_data = pd.concat(
    [
        df_plot,
        df_plot[["clustermatch", "pearson", "spearman"]]
        .rank()
        .rename(
            columns={
                "clustermatch": "clustermatch_rank",
                "pearson": "pearson_rank",
                "spearman": "spearman_rank",
            }
        ),
    ],
    axis=1,
)

# %%
df_r_data.head()

# %% [markdown] tags=[]
# ## Clustermatch/Spearman vs Pearson

# %%
first_coef = "clustermatch"
second_coefs = ("pearson",)

# %%
_tmp_df = df_r_data[
    (df_r_data["Clustermatch (high)"])
    & (df_r_data["Spearman (high)"])
    & ~(df_r_data["Pearson (high)"])
    & ~(df_r_data["Clustermatch (low)"])
    & ~(df_r_data["Spearman (low)"])
    & (df_r_data["Pearson (low)"])
]

if len(second_coefs) > 1:
    _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"].add(
        _tmp_df[f"{second_coefs[1]}_rank"]
    )
else:
    _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"]

_tmp_df = _tmp_df.assign(rank_diff=_tmp_df[f"{first_coef}_rank"].sub(_second_coefs_sum))

# show this just to make sure of the groups
# display(_tmp_df.head())

# sort by rank_diff
_tmp_df = _tmp_df.sort_values("rank_diff", ascending=False)

# # sort by firt_coef value
# _tmp_df = _tmp_df.sort_values(
#     first_coef, ascending=False
# )

_tmp_df = _tmp_df[
    [x for x in _tmp_df.columns if "(high)" not in x and "(low)" not in x]
]

display(_tmp_df.shape)
display(_tmp_df)

# %%
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ## Clustermatch vs Pearson

# %%
first_coef = "clustermatch"
second_coefs = ("pearson",)

# %%
_tmp_df = df_r_data[
    (df_r_data["Clustermatch (high)"])
    & ~(df_r_data["Spearman (high)"])
    & ~(df_r_data["Pearson (high)"])
    & ~(df_r_data["Clustermatch (low)"])
    & ~(df_r_data["Spearman (low)"])
    & (df_r_data["Pearson (low)"])
]

if len(second_coefs) > 1:
    _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"].add(
        _tmp_df[f"{second_coefs[1]}_rank"]
    )
else:
    _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"]

_tmp_df = _tmp_df.assign(rank_diff=_tmp_df[f"{first_coef}_rank"].sub(_second_coefs_sum))

# show this just to make sure of the groups
# display(_tmp_df.head())

# sort by rank_diff
_tmp_df = _tmp_df.sort_values("rank_diff", ascending=False)

# # sort by firt_coef value
# _tmp_df = _tmp_df.sort_values(
#     first_coef, ascending=False
# )

_tmp_df = _tmp_df[
    [x for x in _tmp_df.columns if "(high)" not in x and "(low)" not in x]
]

display(_tmp_df.shape)
display(_tmp_df)

# %%
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ## Clustermatch vs Spearman

# %%
first_coef = "clustermatch"
second_coefs = ("spearman",)

# %%
_tmp_df = df_r_data[
    (df_r_data["Clustermatch (high)"])
    & ~(df_r_data["Spearman (high)"])
    & ~(df_r_data["Pearson (high)"])
    & ~(df_r_data["Clustermatch (low)"])
    & (df_r_data["Spearman (low)"])
    & ~(df_r_data["Pearson (low)"])
]

if len(second_coefs) > 1:
    _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"].add(
        _tmp_df[f"{second_coefs[1]}_rank"]
    )
else:
    _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"]

_tmp_df = _tmp_df.assign(rank_diff=_tmp_df[f"{first_coef}_rank"].sub(_second_coefs_sum))

# show this just to make sure of the groups
# display(_tmp_df.head())

# sort by rank_diff
_tmp_df = _tmp_df.sort_values("rank_diff", ascending=False)

# # sort by firt_coef value
# _tmp_df = _tmp_df.sort_values(
#     first_coef, ascending=False
# )

_tmp_df = _tmp_df[
    [x for x in _tmp_df.columns if "(high)" not in x and "(low)" not in x]
]

display(_tmp_df.shape)
display(_tmp_df)

# %%
for i in range(_tmp_df.shape[0]):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ## Clustermatch vs Spearman/Pearson

# %%
first_coef = "clustermatch"
second_coefs = ("spearman", "pearson")

# %%
_tmp_df = df_r_data[
    (df_r_data["Clustermatch (high)"])
    & ~(df_r_data["Spearman (high)"])
    & ~(df_r_data["Pearson (high)"])
    & ~(df_r_data["Clustermatch (low)"])
    & (df_r_data["Spearman (low)"])
    & (df_r_data["Pearson (low)"])
]

if len(second_coefs) > 1:
    _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"].add(
        _tmp_df[f"{second_coefs[1]}_rank"]
    )
else:
    _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"]

_tmp_df = _tmp_df.assign(rank_diff=_tmp_df[f"{first_coef}_rank"].sub(_second_coefs_sum))

# show this just to make sure of the groups
# display(_tmp_df.head())

# sort by rank_diff
_tmp_df = _tmp_df.sort_values("rank_diff", ascending=False)

# # sort by firt_coef value
# _tmp_df = _tmp_df.sort_values(
#     first_coef, ascending=False
# )

_tmp_df = _tmp_df[
    [x for x in _tmp_df.columns if "(high)" not in x and "(low)" not in x]
]

display(_tmp_df.shape)
display(_tmp_df)

# %%
for i in range(_tmp_df.shape[0]):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ## Pearson vs Clustermatch

# %%
first_coef = "pearson"
second_coefs = ("clustermatch",)

# %%
_tmp_df = df_r_data[
    ~(df_r_data["Clustermatch (high)"])
    & ~(df_r_data["Spearman (high)"])
    & (df_r_data["Pearson (high)"])
    & (df_r_data["Clustermatch (low)"])
    & ~(df_r_data["Spearman (low)"])
    & ~(df_r_data["Pearson (low)"])
]

if len(second_coefs) > 1:
    _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"].add(
        _tmp_df[f"{second_coefs[1]}_rank"]
    )
else:
    _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"]

_tmp_df = _tmp_df.assign(rank_diff=_tmp_df[f"{first_coef}_rank"].sub(_second_coefs_sum))

# show this just to make sure of the groups
# display(_tmp_df.head())

# sort by rank_diff
_tmp_df = _tmp_df.sort_values("rank_diff", ascending=False)

# # sort by firt_coef value
# _tmp_df = _tmp_df.sort_values(
#     first_coef, ascending=False
# )

_tmp_df = _tmp_df[
    [x for x in _tmp_df.columns if "(high)" not in x and "(low)" not in x]
]

display(_tmp_df.shape)
display(_tmp_df)

# %%
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ## Pearson vs Spearman

# %%
first_coef = "pearson"
second_coefs = ("spearman",)

# %%
_tmp_df = df_r_data[
    ~(df_r_data["Clustermatch (high)"])
    & ~(df_r_data["Spearman (high)"])
    & (df_r_data["Pearson (high)"])
    & ~(df_r_data["Clustermatch (low)"])
    & (df_r_data["Spearman (low)"])
    & ~(df_r_data["Pearson (low)"])
]

if len(second_coefs) > 1:
    _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"].add(
        _tmp_df[f"{second_coefs[1]}_rank"]
    )
else:
    _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"]

_tmp_df = _tmp_df.assign(rank_diff=_tmp_df[f"{first_coef}_rank"].sub(_second_coefs_sum))

# show this just to make sure of the groups
# display(_tmp_df.head())

# sort by rank_diff
_tmp_df = _tmp_df.sort_values("rank_diff", ascending=False)

# # sort by firt_coef value
# _tmp_df = _tmp_df.sort_values(
#     first_coef, ascending=False
# )

_tmp_df = _tmp_df[
    [x for x in _tmp_df.columns if "(high)" not in x and "(low)" not in x]
]

display(_tmp_df.shape)
display(_tmp_df)

# %%
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ## Pearson vs Spearman/Clustermatch

# %%
first_coef = "pearson"
second_coefs = ("spearman", "clustermatch")

# %%
_tmp_df = df_r_data[
    ~(df_r_data["Clustermatch (high)"])
    & ~(df_r_data["Spearman (high)"])
    & (df_r_data["Pearson (high)"])
    & (df_r_data["Clustermatch (low)"])
    & (df_r_data["Spearman (low)"])
    & ~(df_r_data["Pearson (low)"])
]

if len(second_coefs) > 1:
    _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"].add(
        _tmp_df[f"{second_coefs[1]}_rank"]
    )
else:
    _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"]

_tmp_df = _tmp_df.assign(rank_diff=_tmp_df[f"{first_coef}_rank"].sub(_second_coefs_sum))

# show this just to make sure of the groups
# display(_tmp_df.head())

# sort by rank_diff
_tmp_df = _tmp_df.sort_values("rank_diff", ascending=False)

# # sort by firt_coef value
# _tmp_df = _tmp_df.sort_values(
#     first_coef, ascending=False
# )

_tmp_df = _tmp_df[
    [x for x in _tmp_df.columns if "(high)" not in x and "(low)" not in x]
]

display(_tmp_df.shape)
display(_tmp_df)

# %%
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ## Spearman vs Pearson

# %%
first_coef = "spearman"
second_coefs = ("pearson",)

# %%
_tmp_df = df_r_data[
    ~(df_r_data["Clustermatch (high)"])
    & (df_r_data["Spearman (high)"])
    & ~(df_r_data["Pearson (high)"])
    & ~(df_r_data["Clustermatch (low)"])
    & ~(df_r_data["Spearman (low)"])
    & (df_r_data["Pearson (low)"])
]

if len(second_coefs) > 1:
    _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"].add(
        _tmp_df[f"{second_coefs[1]}_rank"]
    )
else:
    _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"]

_tmp_df = _tmp_df.assign(rank_diff=_tmp_df[f"{first_coef}_rank"].sub(_second_coefs_sum))

# show this just to make sure of the groups
# display(_tmp_df.head())

# sort by rank_diff
_tmp_df = _tmp_df.sort_values("rank_diff", ascending=False)

# # sort by firt_coef value
# _tmp_df = _tmp_df.sort_values(
#     first_coef, ascending=False
# )

_tmp_df = _tmp_df[
    [x for x in _tmp_df.columns if "(high)" not in x and "(low)" not in x]
]

display(_tmp_df.shape)
display(_tmp_df)

# %%
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %%
