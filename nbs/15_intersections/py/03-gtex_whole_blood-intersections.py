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
# It analyzes how correlation coefficients intersect on different gene pairs. Basically, I take the top gene pairs with the maximum correlation coefficient according to Pearson, Spearman and CCC, and also the equivalent set with the minimum coefficient values, and then compare how these sets intersect each other.
#
# After identifying different intersection sets, I plot some gene pairs to see what's being captured or not by each coefficient.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from upsetplot import plot, from_indicators

from ccc.plots import MyUpSet
from ccc import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
DATASET_CONFIG = conf.GTEX
GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"

# %% tags=[]
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
# ## Correlation

# %% tags=[]
df = pd.read_pickle(INPUT_CORR_FILE)

# %% tags=[]
df.shape

# %% tags=[]
df.head()

# %% tags=[]
df.describe()

# %% tags=[]
# show quantiles
df.apply(lambda x: x.quantile(np.linspace(0.20, 1.0, 20)))


# %% [markdown] tags=[]
# # Prepare data for plotting

# %% tags=[]
def get_lower_upper_quantile(method_name, q):
    return df[method_name].quantile([q, 1 - q])


# %% tags=[]
# test
_tmp = get_lower_upper_quantile("ccc", 0.20)
display(_tmp)

_tmp0, _tmp1 = _tmp
display((_tmp0, _tmp1))

assert _tmp0 == _tmp.iloc[0]
assert _tmp1 == _tmp.iloc[1]

# %% tags=[]
clustermatch_lq, clustermatch_hq = get_lower_upper_quantile("ccc", Q_DIFF)
display((clustermatch_lq, clustermatch_hq))

pearson_lq, pearson_hq = get_lower_upper_quantile("pearson", Q_DIFF)
display((pearson_lq, pearson_hq))

spearman_lq, spearman_hq = get_lower_upper_quantile("spearman", Q_DIFF)
display((spearman_lq, spearman_hq))

# %% tags=[]
pearson_higher = df["pearson"] >= pearson_hq
display(pearson_higher.sum())

pearson_lower = df["pearson"] <= pearson_lq
display(pearson_lower.sum())

# %% tags=[]
spearman_higher = df["spearman"] >= spearman_hq
display(spearman_higher.sum())

spearman_lower = df["spearman"] <= spearman_lq
display(spearman_lower.sum())

# %% tags=[]
clustermatch_higher = df["ccc"] >= clustermatch_hq
display(clustermatch_higher.sum())

clustermatch_lower = df["ccc"] <= clustermatch_lq
display(clustermatch_lower.sum())

# %% [markdown] tags=[]
# # UpSet plot

# %% tags=[]
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

# %% tags=[]
df_plot = pd.concat([df_plot, df], axis=1)

# %% tags=[]
df_plot

# %% tags=[]
assert not df_plot.isna().any().any()

# %% tags=[]
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

# %% tags=[]
categories = sorted(
    [x for x in df_plot.columns if " (" in x],
    reverse=True,
    key=lambda x: x.split(" (")[1] + " (" + x.split(" (")[0],
)

# %% tags=[]
categories

# %% [markdown] tags=[]
# ## All subsets (original full plot)

# %% tags=[]
df_r_data = df_plot

# %% tags=[]
df_r_data.shape

# %% tags=[]
gene_pairs_by_cats = from_indicators(categories, data=df_r_data)

# %% tags=[]
gene_pairs_by_cats

# %% tags=[]
fig = plt.figure(figsize=(18, 5))

g = plot(
    gene_pairs_by_cats,
    show_counts=True,
    sort_categories_by=None,
    element_size=None,
    fig=fig,
)

# %% [markdown] tags=[]
# ## Sort by categories of subsets

# %% tags=[]
df_r_data = df_plot

# %% tags=[]
df_r_data.shape

# %% tags=[]
gene_pairs_by_cats = from_indicators(categories, data=df_r_data)

# %% tags=[]
gene_pairs_by_cats

# %% tags=[]
gene_pairs_by_cats = gene_pairs_by_cats.sort_index()

# %% tags=[]
_tmp_index = gene_pairs_by_cats.index.unique().to_frame(False)
display(_tmp_index)

# %% tags=[]
_tmp_index[_tmp_index.sum(axis=1) == 3]

# %% tags=[]
_tmp_index.apply(lambda x: x[0:3].sum() == 0, axis=1)

# %% tags=[]
# agreements on top
_tmp_index.loc[
    _tmp_index[
        _tmp_index.apply(lambda x: x.sum() > 1, axis=1)
        & _tmp_index.apply(lambda x: x[0:3].sum() == 0, axis=1)
        & _tmp_index.apply(lambda x: 3 > x[3:].sum() > 1, axis=1)
    ].index
].apply(tuple, axis=1).to_numpy()

# %% tags=[]
# agreements on bottom
_tmp_index.loc[
    _tmp_index[
        _tmp_index.apply(lambda x: x.sum() > 1, axis=1)
        & _tmp_index.apply(lambda x: 3 > x[0:3].sum() > 1, axis=1)
        & _tmp_index.apply(lambda x: x[3:].sum() == 0, axis=1)
    ].index
].apply(tuple, axis=1).to_numpy()

# %% tags=[]
# diagreements
_tmp_index.loc[
    _tmp_index[
        _tmp_index.apply(lambda x: x.sum() > 1, axis=1)
        & _tmp_index.apply(lambda x: x[0:3].sum() > 0, axis=1)
        & _tmp_index.apply(lambda x: x[3:].sum() > 0, axis=1)
    ].index
].apply(tuple, axis=1).to_numpy()

# %% tags=[]
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
        #   ccc
        (False, True, False, True, False, True),
        (False, True, False, False, False, True),
        (True, False, False, False, False, True),
        (True, True, False, False, False, True),
        #   pearson
        (False, False, True, False, True, False),
        (True, False, False, False, True, False),
        (True, False, True, False, True, False),
        #   spearman
        (False, True, False, True, False, False),
    ]
]

# %%
gene_pairs_by_cats.head()

# %%
gene_pairs_by_cats = gene_pairs_by_cats.rename(
    columns={
        "Clustermatch (high)": "CCC (high)",
        "Clustermatch (low)": "CCC (low)",
    }
)

gene_pairs_by_cats.index.set_names(
    {
        "Clustermatch (high)": "CCC (high)",
        "Clustermatch (low)": "CCC (low)",
    },
    inplace=True,
)

# %% tags=[]
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

# %% [markdown] tags=[]
# This plot has the sets that represent agreements on the left, and disagreements on the right.

# %% [markdown]
# The plot shown here is **not the final one for the manuscript**:
#
# 1. Open the main output svg file (`upsetplot-main.svg`)
# 1. Include the file generated here (`upsetplot.svg`)
# 1. Rearrange the `1e6` at the top, which is overlapping other numbers.
# 1. Add the triangles (red and green). For this I need to move the category names at the left to make space.
# 1. Add a rectangle and clip it to remove the extra space on the left
# 1. Add the "Agreements" and "Disagreements" labels below.
# 1. Automatically resize page to drawing.
# 1. Add a rectangle that covers the entire drawing with white background. And send it to the background.

# %% [markdown] tags=[]
# # Save groups of gene pairs in each subset

# %% tags=[]
display(df_plot.shape)
display(df_plot.head())

# %% tags=[]
conf.GTEX["GENE_PAIR_INTERSECTIONS"].mkdir(parents=True, exist_ok=True)

# %% tags=[]
output_file = (
    conf.GTEX["GENE_PAIR_INTERSECTIONS"]
    / f"gene_pair_intersections-gtex_v8-{GTEX_TISSUE}-{GENE_SEL_STRATEGY}.pkl"
)
display(output_file)

# %% tags=[]
df_plot.to_pickle(output_file)

# %% tags=[]
