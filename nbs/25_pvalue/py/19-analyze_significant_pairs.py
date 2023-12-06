# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
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
import numpy as np
import pandas as pd
# from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import seaborn as sns

from ccc import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
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
OUTPUT_DATA_DIR = DATASET_CONFIG["GENE_PAIR_INTERSECTIONS"]
assert OUTPUT_DATA_DIR.exists()
display(OUTPUT_DATA_DIR)

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

# %%
INPUT_PVALUES_FILE = DATASET_CONFIG["GENE_PAIR_INTERSECTIONS"] / "pvalues" / "gene_pair-samples-pvalues-fdr.pkl"
display(INPUT_PVALUES_FILE)
assert INPUT_PVALUES_FILE.exists()

# %%

# %% [markdown] tags=[]
# # Data

# %% [markdown] tags=[]
# ## Gene Ensembl ID -> Symbol mapping

# %% tags=[]
gene_map = pd.read_pickle(
    DATASET_CONFIG["DATA_DIR"] / "gtex_gene_id_symbol_mappings.pkl"
)

# %% tags=[]
gene_map = gene_map.set_index("gene_ens_id")["gene_symbol"].to_dict()

# %% tags=[]
assert gene_map["ENSG00000145309.5"] == "CABS1"

# %% [markdown] tags=[]
# ## Gene expression

# %% tags=[]
gene_expr_df = pd.read_pickle(INPUT_GENE_EXPR_FILE)

# %% tags=[]
gene_expr_df.shape

# %% tags=[]
gene_expr_df.head()

# %% [markdown] tags=[]
# ## Gene pairs intersection

# %% tags=[]
df_plot = pd.read_pickle(INPUT_GENE_PAIRS_INTERSECTIONS_FILE).sort_index()

# %% tags=[]
df_plot.shape

# %% tags=[]
df_plot.head()

# %% [markdown] tags=[]
# ## p-values

# %%
df_pvalues = pd.read_pickle(INPUT_PVALUES_FILE).sort_index()

# %%
df_pvalues.shape

# %%
df_pvalues.head()

# %%
# remove duplicated gene pairs
df_pvalues = df_pvalues[~df_pvalues.index.duplicated(keep="first")]


# %% [markdown]
# **Note**: Here the "group" column specifies the categories in Figure 3a, followed by `top_[coef]`, where for the same category I sorted gene pairs by `coef`. This allows me, for instance, to take the gene pairs where Pearson is high and CCC is low, and sort by any of those coefficient values.

# %% [markdown] tags=[]
# ## Functions

# %% tags=[]
def plot_gene_pair(top_pairs_df, idx, bins="log", plot_gene_ids=True):
    """
    It plots a gene pair using a hexbin plot. The idea of this function is
    to quickly have an idea of the patterns (if any) present in a couple genes.

    Args:
        top_pairs_df: a dataframe with a preselected group of genes pairs (for instance,
            those where pearson is high and ccc is low. Each row is a gene pair.
            It is the output of function get_gene_pairs.
        idx: an integer that indicates which row in top_pairs_df you want to plot.
        bins: the "bins" parameter of seaborn's jointplot.
        plot_gene_ids: it adds genes' Ensembl IDs to the x and y labels.

    Returns:
        The JointGrid object returned by seaborn.jointplot.
    """
    gene0, gene1 = top_pairs_df.iloc[idx].name
    display((gene0, gene1))

    gene0_symbol, gene1_symbol = gene_map[gene0], gene_map[gene1]
    display((gene0_symbol, gene1_symbol))

    _pearson, _pearson_fdr, _spearman, _spearman_fdr, _ccc, _ccc_fdr = top_pairs_df.loc[
        (gene0, gene1), ["pearson", "pearson_fdr", "spearman", "spearman_fdr", "ccc", "ccc_fdr"]
    ].tolist()

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

    gene_y_id = p.ax_joint.get_ylabel()
    gene_y_symbol = gene_map[gene_y_id]

    if plot_gene_ids:
        p.ax_joint.set_xlabel(f"{gene_x_id}\n{gene_x_symbol}")
        p.ax_joint.set_ylabel(f"{gene_y_id}\n{gene_y_symbol}")
    else:
        p.ax_joint.set_xlabel(f"{gene_x_symbol}", fontstyle="italic")
        p.ax_joint.set_ylabel(f"{gene_y_symbol}", fontstyle="italic")

    _title = f"$c={_ccc:.2f}$ (${_ccc_fdr:.2e}$)  $p={_pearson:.2f}$ (${_pearson_fdr:.2e}$)    $s={_spearman:.2f}$ (${_spearman_fdr:.2e}$)"
    p.fig.suptitle(_title)

    return p


# %% [markdown] tags=[]
# # Analyze each category of gene pairs

# %% [markdown]
# Here I analyze some of the categories of gene pairs in Figure 3b (disagreements in particular). For instance, "CCC high and Pearson low", or "Pearson high and CCC low".

# %%
df_pvalues["group"].sort_values().unique().tolist()

# %% [markdown] tags=[]
# ## CCC/Spearman high, Pearson low

# %%
cat_name = "ccc_spearman_high_and_pearson_low-"

# %%
_df = df_pvalues[df_pvalues["group"].str.startswith(cat_name)]

# %%
_df.shape

# %%
_df = _df.sort_values("pearson_pvalue", ascending=False)
display(_df.head())

# %%
for i in range(min(_df.shape[0], 5)):
    display(f"Index: {i}")
    p = plot_gene_pair(_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown]
# **Note** Using scite_ to detect some of this gave positive results, such as GPNMB and APOC2.

# %% [markdown] tags=[]
# ## CCC high, Pearson low

# %%
cat_name = "ccc_high_and_pearson_low-"

# %%
_df = df_pvalues[df_pvalues["group"].str.startswith(cat_name)]

# %%
_df.shape

# %%
_df = _df.sort_values("pearson_pvalue", ascending=False)
display(_df.head())

# %%
for i in range(min(_df.shape[0], 10)):
    display(f"Index: {i}")
    p = plot_gene_pair(_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ## CCC high, Spearman low

# %%
cat_name = "ccc_high_and_spearman_low-"

# %%
_df = df_pvalues[df_pvalues["group"].str.startswith(cat_name)]

# %%
_df.shape

# %%
_df = _df.sort_values("spearman_pvalue", ascending=False)
display(_df.head())

# %%
for i in range(min(_df.shape[0], 10)):
    display(f"Index: {i}")
    p = plot_gene_pair(_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ## CCC high, Pearson/Spearman low

# %%
cat_name = "ccc_high_and_spearman_pearson_low-"

# %%
_df = df_pvalues[df_pvalues["group"].str.startswith(cat_name)]

# %%
_df.shape

# %%
_df = _df.sort_values("spearman_pvalue", ascending=False)
display(_df.head())

# %%
for i in range(min(_df.shape[0], 10)):
    display(f"Index: {i}")
    p = plot_gene_pair(_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ## Pearson high, CCC low

# %%
cat_name = "pearson_high_and_ccc_low-"

# %%
_df = df_pvalues[df_pvalues["group"].str.startswith(cat_name)]

# %%
_df.shape

# %%
_df = _df.sort_values("ccc_pvalue", ascending=False)
display(_df.head())

# %%
for i in range(min(_df.shape[0], 10)):
    display(f"Index: {i}")
    p = plot_gene_pair(_df, i)
    display(p.fig)
    plt.close(p.fig)

# %% [markdown] tags=[]
# ## Pearson high, Spearman low

# %% [markdown] tags=[]
# ## Pearson high, CCC/Spearman low

# %% [markdown] tags=[]
# ## Spearman high, Pearson low

# %% [markdown] tags=[]
# # Save

# %% tags=[]
INPUT_PVALUES_FILE.parent

# %% tags=[]
INPUT_PVALUES_FILE.stem

# %% tags=[]
INPUT_PVALUES_FILE.suffix

# %% tags=[]
output_file = (
    INPUT_PVALUES_FILE.parent
    / f"{INPUT_PVALUES_FILE.stem}-fdr{INPUT_PVALUES_FILE.suffix}"
)
display(output_file)

# %% tags=[]
pvalues.to_pickle(output_file)

# %% tags=[]
