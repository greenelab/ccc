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
# It plot specific gene pairs from different subsets of the UpSet plot (intersections) generated before.
#
# The idea of the notebook is to take a look at the patterns found and not found by different methods.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from clustermatch import conf

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
INPUT_GENE_PAIRS_INTERSECTIONS_FILE = (
    DATASET_CONFIG["GENE_PAIR_INTERSECTIONS"]
    / f"gene_pair_intersections-gtex_v8-{GTEX_TISSUE}-{GENE_SEL_STRATEGY}.pkl"
)
display(INPUT_GENE_PAIRS_INTERSECTIONS_FILE)

assert INPUT_GENE_PAIRS_INTERSECTIONS_FILE.exists()

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
# ## Gene pairs intersection

# %%
df_plot = pd.read_pickle(INPUT_GENE_PAIRS_INTERSECTIONS_FILE)

# %%
df_plot.shape

# %%
df_plot.head()

# %% [markdown]
# # Look at specific gene pair cases

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

# %%
df_r_data_boolean_cols = set(
    [x for x in df_r_data.columns if " (high)" in x or " (low)" in x]
)

# %%
df_r_data_boolean_cols


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
def get_gene_pairs(first_coef, query_set):
    assert all([x in df_r_data_boolean_cols for x in query_set])

    query = np.concatenate(
        [
            # columns that have to be true
            np.concatenate(
                [df_r_data[c].to_numpy().reshape(-1, 1) for c in query_set], axis=1
            )
            .all(axis=1)
            .reshape(-1, 1),
            # rest of the columns, that have to be false
            np.concatenate(
                [
                    ~df_r_data[c].to_numpy().reshape(-1, 1)
                    for c in df_r_data_boolean_cols
                    if c not in query_set
                ],
                axis=1,
            )
            .all(axis=1)
            .reshape(-1, 1),
        ],
        axis=1,
    ).all(axis=1)

    _tmp_df = df_r_data[query]

    # if len(second_coefs) > 1:
    #     _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"].add(
    #         _tmp_df[f"{second_coefs[1]}_rank"]
    #     )
    # else:
    #     _second_coefs_sum = _tmp_df[f"{second_coefs[0]}_rank"]

    # _tmp_df = _tmp_df.assign(rank_diff=_tmp_df[f"{first_coef}_rank"].sub(_second_coefs_sum))

    # show this just to make sure of the groups
    # display(_tmp_df.head())

    # # sort by rank_diff
    # _tmp_df = _tmp_df.sort_values("rank_diff", ascending=False)

    # sort by firt_coef value
    _tmp_df = _tmp_df.sort_values(first_coef, ascending=False)

    _tmp_df = _tmp_df[
        [x for x in _tmp_df.columns if "(high)" not in x and "(low)" not in x]
    ]

    return _tmp_df


# %% [markdown] tags=[]
# ## Clustermatch/Spearman vs Pearson

# %%
_tmp_df = get_gene_pairs(
    "clustermatch",
    {
        "Clustermatch (high)",
        "Spearman (high)",
        "Pearson (low)",
    },
)

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
_tmp_df = get_gene_pairs(
    "clustermatch",
    {
        "Clustermatch (high)",
        "Pearson (low)",
    },
)

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
_tmp_df = get_gene_pairs(
    "clustermatch",
    {
        "Clustermatch (high)",
        "Spearman (low)",
    },
)

display(_tmp_df.shape)
display(_tmp_df)

# %%
for i in range(min(_tmp_df.shape[0], 30)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ## Clustermatch vs Spearman/Pearson

# %%
_tmp_df = get_gene_pairs(
    "clustermatch",
    {
        "Clustermatch (high)",
        "Spearman (low)",
        "Pearson (low)",
    },
)

display(_tmp_df.shape)
display(_tmp_df)

# %%
for i in range(min(_tmp_df.shape[0], 10)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ## Pearson vs Clustermatch

# %%
_tmp_df = get_gene_pairs(
    "pearson",
    {
        "Clustermatch (low)",
        "Pearson (high)",
    },
)

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
_tmp_df = get_gene_pairs(
    "pearson",
    {
        "Spearman (low)",
        "Pearson (high)",
    },
)

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
_tmp_df = get_gene_pairs(
    "pearson",
    {
        "Clustermatch (low)",
        "Spearman (low)",
        "Pearson (high)",
    },
)

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
_tmp_df = get_gene_pairs(
    "spearman",
    {
        "Spearman (high)",
        "Pearson (low)",
    },
)

display(_tmp_df.shape)
display(_tmp_df)

# %%
for i in range(min(_tmp_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_tmp_df, i)
    display(p.fig)
    plt.close(p.fig)

# %%
