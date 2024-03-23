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
# It loads the correlation values and p-values and perform some analyses and plots.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import seaborn as sns

from ccc import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
# this gene pair was originally found with ccc on whole blood
# interesting: https://clincancerres.aacrjournals.org/content/26/21/5567.figures-only
gene0_id, gene1_id = "ENSG00000147050.14", "ENSG00000067048.16"
gene0_symbol, gene1_symbol = "KDM6A", "DDX3Y"

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
TISSUE_DIR = conf.GTEX["DATA_DIR"] / "data_by_tissue"
assert TISSUE_DIR.exists()

# %% tags=[]
INPUT_DIR = (
    conf.GTEX["RESULTS_DIR"]
    / "other_tissues"
    / f"{gene0_symbol.lower()}_vs_{gene1_symbol.lower()}"
)
display(INPUT_DIR)

# %% tags=[]
OUTPUT_FIGURE_DIR = (
    conf.MANUSCRIPT["FIGURES_DIR"]
    / "coefs_comp"
    / f"{gene0_symbol.lower()}_vs_{gene1_symbol.lower()}"
)
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% [markdown] tags=[]
# # Data

# %% [markdown] tags=[]
# ## GTEx metadata

# %% tags=[]
gtex_metadata = pd.read_pickle(conf.GTEX["DATA_DIR"] / "gtex_v8-sample_metadata.pkl")

# %% tags=[]
gtex_metadata.shape

# %% tags=[]
gtex_metadata.head()

# %% [markdown] tags=[]
# ## Gene Ensembl ID -> Symbol mapping

# %% tags=[]
gene_map = pd.read_pickle(conf.GTEX["DATA_DIR"] / "gtex_gene_id_symbol_mappings.pkl")

# %% tags=[]
gene_map = gene_map.set_index("gene_ens_id")["gene_symbol"].to_dict()

# %% tags=[]
assert gene_map["ENSG00000145309.5"] == "CABS1"

# %% tags=[]
assert gene_map[gene0_id] == gene0_symbol
assert gene_map[gene1_id] == gene1_symbol

# %% [markdown] tags=[]
# ## Correlation on all tissues

# %% tags=[]
res_all = pd.read_pickle(INPUT_DIR / "coef_values.pkl")

# %% tags=[]
res_all.shape

# %% tags=[]
res_all.head()

# %% [markdown] tags=[]
# ## P-values on all tissues

# %% tags=[]
res_pval_all = pd.read_pickle(INPUT_DIR / "coef_pvalues.pkl")

# %% tags=[]
res_pval_all.shape

# %% tags=[]
res_pval_all.head()

# %% tags=[]
-np.log10(res_pval_all).describe()

# %% [markdown] tags=[]
# ### Adjust p-values

# %% tags=[]
for col in ("cm", "pearson", "spearman"):
    adj_pvals = multipletests(res_pval_all[col], alpha=0.05, method="fdr_bh")
    res_pval_all = res_pval_all.assign(**{col: adj_pvals[1]})

# %% tags=[]
res_pval_all.shape

# %% tags=[]
res_pval_all.head()

# %% tags=[]
-np.log10(res_pval_all).describe()


# %% [markdown] tags=[]
# # Plot

# %% tags=[]
def get_tissue_file(name):
    """
    Given a part of a tissue name, it returns a file path to the
    expression data for that tissue in GTEx. It fails if more than
    one files are found.

    Args:
        name: a string with the tissue name (or a part of it).

    Returns:
        A Path object pointing to the gene expression file for the
        given tissue.
    """
    tissue_files = []
    for f in TISSUE_DIR.glob("*.pkl"):
        if name in f.name:
            tissue_files.append(f)

    assert len(tissue_files) == 1
    return tissue_files[0]


# %% tags=[]
# testing
_tmp = get_tissue_file("whole_blood")
assert _tmp.exists()


# %% tags=[]
def simplify_tissue_name(tissue_name):
    return f"{tissue_name[0].upper()}{tissue_name[1:].replace('_', ' ')}"


# %% tags=[]
assert simplify_tissue_name("whole_blood") == "Whole blood"
assert simplify_tissue_name("uterus") == "Uterus"


# %% tags=[]
def pvalue_to_star(pvalue):
    s = ""
    if pvalue < 0.001:
        s = "***"
    elif pvalue < 0.01:
        s = "**"
    elif pvalue < 0.05:
        s = "*"

    return s


# %% tags=[]
def plot_gene_pair(
    tissue_name, gene0, gene1, hue=None, kind="hex", ylim=None, bins="log"
):
    """
    It plots (joint plot) a gene pair from the given tissue. It saves the plot
    for the manuscript.
    """
    # merge gene expression with metadata
    tissue_file = get_tissue_file(tissue_name)
    tissue_data = pd.read_pickle(tissue_file).T[[gene0, gene1]]
    tissue_data = pd.merge(
        tissue_data,
        gtex_metadata,
        how="inner",
        left_index=True,
        right_index=True,
        validate="one_to_one",
    )

    # get gene symbols
    gene0_symbol, gene1_symbol = gene_map[gene0], gene_map[gene1]
    display((gene0_symbol, gene1_symbol))

    # compute correlations for this gene pair
    tissue_key = tissue_file.stem.split("_data_")[1]

    _ccc = res_all.loc[tissue_key, "cm"]
    _ccc_p = pvalue_to_star(res_pval_all.loc[tissue_key, "cm"])
    _pearson = res_all.loc[tissue_key, "pearson"]
    _pearson_p = pvalue_to_star(res_pval_all.loc[tissue_key, "pearson"])
    _spearman = res_all.loc[tissue_key, "spearman"]
    _spearman_p = pvalue_to_star(res_pval_all.loc[tissue_key, "spearman"])

    _title = f"{simplify_tissue_name(tissue_name)}\n$c={_ccc:.2f}${_ccc_p}  $p={_pearson:.2f}${_pearson_p}  $s={_spearman:.2f}${_spearman_p}"

    other_args = {
        "kind": kind,  # if hue is None else "scatter",
        "rasterized": True,
    }
    if hue is None:
        other_args["bins"] = bins
    else:
        other_args["hue_order"] = ["Male", "Female"]

    with sns.plotting_context("paper", font_scale=1.5):
        p = sns.jointplot(
            data=tissue_data,
            x=gene0,
            y=gene1,
            hue=hue,
            **other_args,
            # ylim=(0, 500),
        )

        if ylim is not None:
            p.ax_joint.set_ylim(ylim)

        gene_x_id = p.ax_joint.get_xlabel()
        gene_x_symbol = gene_map[gene_x_id]
        p.ax_joint.set_xlabel(f"{gene_x_symbol}", fontstyle="italic")

        gene_y_id = p.ax_joint.get_ylabel()
        gene_y_symbol = gene_map[gene_y_id]
        p.ax_joint.set_ylabel(f"{gene_y_symbol}", fontstyle="italic")

        p.fig.suptitle(_title)

        # save
        output_file = (
            OUTPUT_FIGURE_DIR
            / f"gtex_{tissue_name}-{gene_x_symbol}_vs_{gene_y_symbol}.svg"
        )
        display(output_file)

        plt.savefig(
            output_file,
            bbox_inches="tight",
            dpi=300,
            facecolor="white",
        )

    return tissue_data


# %% [markdown] tags=[]
# ## In whole blood (where this gene pair was found)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "whole_blood",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% [markdown] tags=[]
# ## Lowest tissues in ccc

# %% tags=[]
_tissue_data = plot_gene_pair(
    "uterus",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "ovary",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "vagina",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "brain_cerebellum",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "small_intestine_terminal_ileum",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "brain_spinal_cord_cervical_c1",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "testis",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% [markdown] tags=[]
# ## Highest tissues in ccc

# %% tags=[]
_tissue_data = plot_gene_pair(
    "cells_cultured_fibroblasts",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "breast_mammary_tissue",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% [markdown] tags=[]
# ## Pearson low, CCC high

# %% tags=[]
_tissue_data = plot_gene_pair(
    "brain_anterior_cingulate_cortex_ba24",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "brain_amygdala",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "brain_frontal_cortex_ba9",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "bladder",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "heart_atrial_appendage",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% [markdown] tags=[]
# ## Spearman low, CCC high

# %% tags=[]
_tissue_data = plot_gene_pair(
    "heart_left_ventricle",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "adipose_visceral_omentum",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "skin_not_sun_exposed_suprapubic",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% tags=[]
_tissue_data = plot_gene_pair(
    "pancreas",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% [markdown] tags=[]
# # Create final figure

# %% tags=[]
from svgutils.compose import Figure, SVG, Panel

# %% tags=[]
name_suffix = f"{gene0_symbol}_vs_{gene1_symbol}"
display(name_suffix)

# %% tags=[]
Figure(
    "6.0767480cm",
    "8.7045984cm",
    Panel(
        SVG(OUTPUT_FIGURE_DIR / f"gtex_whole_blood-{name_suffix}.svg").scale(0.005),
        SVG(OUTPUT_FIGURE_DIR / f"gtex_testis-{name_suffix}.svg")
        .scale(0.005)
        .move(2, 0),
        SVG(OUTPUT_FIGURE_DIR / f"gtex_cells_cultured_fibroblasts-{name_suffix}.svg")
        .scale(0.005)
        .move(2 * 2, 0),
    ),
    Panel(
        SVG(OUTPUT_FIGURE_DIR / f"gtex_brain_cerebellum-{name_suffix}.svg").scale(
            0.005
        ),
        SVG(
            OUTPUT_FIGURE_DIR / f"gtex_small_intestine_terminal_ileum-{name_suffix}.svg"
        )
        .scale(0.005)
        .move(2, 0),
        SVG(OUTPUT_FIGURE_DIR / f"gtex_breast_mammary_tissue-{name_suffix}.svg")
        .scale(0.005)
        .move(2 * 2, 0),
    ).move(0, 2.2),
    Panel(
        SVG(
            OUTPUT_FIGURE_DIR
            / f"gtex_brain_anterior_cingulate_cortex_ba24-{name_suffix}.svg"
        ).scale(0.005),
        SVG(OUTPUT_FIGURE_DIR / f"gtex_brain_amygdala-{name_suffix}.svg")
        .scale(0.005)
        .move(2, 0),
        SVG(OUTPUT_FIGURE_DIR / f"gtex_heart_atrial_appendage-{name_suffix}.svg")
        .scale(0.005)
        .move(2 * 2, 0),
    ).move(0, 2.2 * 2),
    Panel(
        SVG(OUTPUT_FIGURE_DIR / f"gtex_vagina-{name_suffix}.svg").scale(0.005),
        SVG(OUTPUT_FIGURE_DIR / f"gtex_ovary-{name_suffix}.svg")
        .scale(0.005)
        .move(2, 0),
        SVG(OUTPUT_FIGURE_DIR / f"gtex_uterus-{name_suffix}.svg")
        .scale(0.005)
        .move(2 * 2, 0),
    ).move(0, 2.2 * 3),
).save(OUTPUT_FIGURE_DIR / f"gtex-{name_suffix}-main.svg")

# %% [markdown] tags=[]
# Now open the file, reside to fit drawing to page, and add a white rectangle to the background.

# %% tags=[]
