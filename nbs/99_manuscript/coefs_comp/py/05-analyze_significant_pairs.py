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
# After the manuscript revision, here I analyze each gene pair in Figure 3b to see which one can be replaced taking into account the significance of the correlation coefficients.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
import numpy as np
import pandas as pd

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

# %% tags=[]
INPUT_PVALUES_FILE = (
    DATASET_CONFIG["GENE_PAIR_INTERSECTIONS"]
    / "pvalues"
    / "gene_pair-samples-pvalues-fdr.pkl"
)
display(INPUT_PVALUES_FILE)
assert INPUT_PVALUES_FILE.exists()

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
# ## p-values

# %% tags=[]
df_pvalues = pd.read_pickle(INPUT_PVALUES_FILE).sort_index()

# %% tags=[]
df_pvalues.shape

# %% tags=[]
df_pvalues.head()

# %% tags=[]
# remove duplicated gene pairs
df_pvalues = df_pvalues[~df_pvalues.index.duplicated(keep="first")]


# %% [markdown] tags=[]
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
        (gene0, gene1),
        ["pearson", "pearson_fdr", "spearman", "spearman_fdr", "ccc", "ccc_fdr"],
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
# # Analyze current gene pairs in Figure 3b (before manuscript revision)

# %% [markdown] tags=[]
# Here I look at all the current gene pairs in Figure 3b. We have to decide on whether keep them or remove them.
# This is bacause after doing the p-value analysis and looking for significant gene pairs across all correlation coefficients, we want to keep some pairs where, for example, CCC is high and significant and Pearson is low but also nonsignificant.
# Also, if possible and does not affect the conclusions, we want to keep those gene pairs that we are already discussing in the manuscript.
#
# Here I look at the current gene pairs, and then I also use notebook `nbs/99_manuscript/coefs_comp/08_05-gtex_whole_blood-intersections_plots.ipynb` to sort gene pairs considering *both* the coefficient value and p-values.
# I write down here the decision on each gene pair (keep or replace) and the reasons.

# %% [markdown] tags=[]
# ## IFNG vs SDS

# %% [markdown] tags=[]
# **Decision:** Keep, because:
# * Very high CCC value, although Pearson is significant also.
# * *SDS* is a gene with fewer publications than expected, we are mentioning this argument in the manuscript.

# %% tags=[]
gene0_id = "ENSG00000135094.10"
gene1_id = "ENSG00000111537.4"

_tmp = df_pvalues.loc[[(gene0_id, gene1_id)]]
display(_tmp)

p = plot_gene_pair(_tmp, 0)
display(p)

# %% [markdown] tags=[]
# ## ~JUN vs APOC1~

# %% [markdown] tags=[]
# **Decision:** Remove and do not replace, because:
# * Very high CCC value, but Pearson is significant also.
# * And we do not discuss any of these genes in the manuscript.
# * And we do not need another example of this category.

# %% tags=[]
gene0_id = "ENSG00000130208.9"
gene1_id = "ENSG00000177606.6"

_tmp = df_pvalues.loc[[(gene0_id, gene1_id)]]
display(_tmp)

p = plot_gene_pair(_tmp, 0)
display(p)

# %% tags=[]
# Initially, I though in replacing it by LIF vs APOC1

# gene0_id = "ENSG00000130208.9"
# gene1_id = "ENSG00000128342.4"

# _tmp = df_pvalues.loc[[(gene0_id, gene1_id)]]
# display(_tmp)

# p = plot_gene_pair(_tmp, 0)
# display(p)

# %% [markdown] tags=[]
# ## ~ZDHHC12 vs CCL18~ CCL18 vs PRSS36

# %% [markdown] tags=[]
# **Decision:** Replace, because:
# * Significant in all coefficients.
# * Although we discuss ZDHHC12 as a little studied gene, the ZDHHC family has more studies.
# * Not strong link of ZDHHC12 to the rest of the network: https://hb.flatironinstitute.org/gene/84885+6362
#
# *Replaced by:* CCL18 vs PRSS36
#
# * High CCC, low Pearson and not significant, although still significant with Spearman.
# * Predicted to be expressed in macrophage in GIANT: https://hb.flatironinstitute.org/gene/6362+146547

# %% tags=[]
gene0_id = "ENSG00000275385.1"
gene1_id = "ENSG00000160446.18"

_tmp = df_pvalues.loc[[(gene0_id, gene1_id)]]
display(_tmp)

p = plot_gene_pair(_tmp, 0)
display(p)

# %% tags=[]
gene0_id = "ENSG00000275385.1"
gene1_id = "ENSG00000178226.10"

_tmp = df_pvalues.loc[[(gene0_id, gene1_id)]]
display(_tmp)

p = plot_gene_pair(_tmp, 0)
display(p)

# %% [markdown] tags=[]
# ## UTY vs KDM6A

# %% [markdown] tags=[]
# **Decision:** Keep, because:
# * They are genes in the sex chromosomes.
# * Figure 4 is about them.
# * A drawback is that all coefficients are significant. However, in other GTEx tissues (Figure 4) it is possible to see that only CCC captures this pattern consistently.

# %% tags=[]
gene0_id = "ENSG00000147050.14"
gene1_id = "ENSG00000183878.15"

_tmp = df_pvalues.loc[[(gene0_id, gene1_id)]]
display(_tmp)

p = plot_gene_pair(_tmp, 0)
display(p)

# %% [markdown] tags=[]
# ## RASSF2 vs CYTIP + KDM6A vs DDX3Y

# %% [markdown] tags=[]
# **Decision:** Keep and add another, because:
# * Nice example of two embedded linear relationships.
# * Although all coefficients are significant, we can add another example in this category (KDM6A vs DDX3Y) where spearman is not significant.
# * Predicted to be expressed in leukocytes in GIANT: https://hb.flatironinstitute.org/gene/9770+9595
#
# *Add another pair:* KDM6A vs DDX3Y

# %% tags=[]
gene0_id = "ENSG00000115165.9"
gene1_id = "ENSG00000101265.15"

_tmp = df_pvalues.loc[[(gene0_id, gene1_id)]]
display(_tmp)

p = plot_gene_pair(_tmp, 0)
display(p)

# %% tags=[]
gene0_id = "ENSG00000147050.14"
gene1_id = "ENSG00000067048.16"

_tmp = df_pvalues.loc[[(gene0_id, gene1_id)]]
display(_tmp)

p = plot_gene_pair(_tmp, 0)
display(p)

# %% [markdown] tags=[]
# ## AC068580.6 vs KLHL21

# %% [markdown] tags=[]
# **Decision:** Keep, because:
# * We already talk about this gene pair in the text (about KLHL21).
# * Pearson is not significant.
# * Although spearman is significant, all the other gene pairs in this category are also significant in Spearman. The other gene pair that is less significant for Spearman is C17orf53 vs TPX2 (but not worth replacing).
# * Predicted to be expressed in XXX in GIANT: URL_HERE

# %% tags=[]
gene0_id = "ENSG00000162413.16"
gene1_id = "ENSG00000235027.1"

_tmp = df_pvalues.loc[[(gene0_id, gene1_id)]]
display(_tmp)

p = plot_gene_pair(_tmp, 0)
display(p)

# %% [markdown] tags=[]
# ## MYOZ1 vs TNNI2

# %% [markdown] tags=[]
# **Decision:** Keep, because:
# * We are already analyzing these gene pair: it's included in Figure 5.
# * Pearson value is very high.
# * However, all other coefficients are significant.
# * Predicted to be expressed in skeletal muscle in GIANT: https://hb.flatironinstitute.org/gene/58529+7136

# %% tags=[]
gene0_id = "ENSG00000130598.15"
gene1_id = "ENSG00000177791.11"

_tmp = df_pvalues.loc[[(gene0_id, gene1_id)]]
display(_tmp)

p = plot_gene_pair(_tmp, 0)
display(p)

# %% [markdown] tags=[]
# ## ~PYGM vs TPM2~ C19orf33 vs SCGB3A1

# %% [markdown] tags=[]
# **Decision:** Replace, because:
# * It's significant in all coefficients.
# * Predicted to be expressed in skeletal muscle in GIANT: https://hb.flatironinstitute.org/gene/5837+7169
#
# *Replaced by:* C19orf33 vs SCGB3A1:
# * Only significant in Pearson
# * Predicted to be expressed in the placenta in GIANT: https://hb.flatironinstitute.org/gene/64073+92304

# %% tags=[]
gene0_id = "ENSG00000198467.13"
gene1_id = "ENSG00000068976.13"

_tmp = df_pvalues.loc[[(gene0_id, gene1_id)]]
display(_tmp)

p = plot_gene_pair(_tmp, 0)
display(p)

# %% tags=[]
gene0_id = "ENSG00000167644.11"
gene1_id = "ENSG00000161055.3"

_tmp = df_pvalues.loc[[(gene0_id, gene1_id)]]
display(_tmp)

p = plot_gene_pair(_tmp, 0)
display(p)

# %% tags=[]
