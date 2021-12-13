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
import re

import pandas as pd
import numpy as np

from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.preprocessing import minmax_scale

from clustermatch import conf
from clustermatch.coef import cm

# %% [markdown] tags=[]
# # Settings

# %%
# DATASET_CONFIG = conf.GTEX
# GTEX_TISSUE = "whole_blood"
# GENE_SEL_STRATEGY = "var_pc_log2"

# %% [markdown] tags=[]
# # Paths

# %%
TISSUE_DIR = conf.GTEX["DATA_DIR"] / "data_by_tissue"
assert TISSUE_DIR.exists()

# %% [markdown]
# # Data

# %% [markdown] tags=[]
# ## GTEx samples info

# %%
gtex_samples = pd.read_csv(
    conf.GTEX["DATA_DIR"] / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt",
    sep="\t",
    index_col="SAMPID",
)

# %%
display(gtex_samples.shape)
assert gtex_samples.index.is_unique

# %%
gtex_samples.head()

# %% [markdown] tags=[]
# ## GTEx subject phenotypes

# %%
gtex_phenotypes = pd.read_csv(
    conf.GTEX["DATA_DIR"] / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt",
    sep="\t",
)

# %%
gtex_phenotypes.shape

# %%
gtex_phenotypes.head()

# %% [markdown] tags=[]
# ## GTEx gene expression sample

# %%
pd.read_pickle(next(TISSUE_DIR.glob("*.pkl"))).head()

# %% [markdown] tags=[]
# ## Gene Ensembl ID -> Symbol mapping

# %%
gene_map = pd.read_pickle(conf.GTEX["DATA_DIR"] / "gtex_gene_id_symbol_mappings.pkl")

# %%
gene_map = gene_map.set_index("gene_ens_id")["gene_symbol"].to_dict()

# %%
assert gene_map["ENSG00000145309.5"] == "CABS1"

# %% [markdown]
# # Run

# %%
# there are the two sex chromosome genes:
# ('ENSG00000147050.14', 'ENSG00000183878.15')
# ('KDM6A', 'UTY')
gene0_id, gene1_id = "ENSG00000147050.14", "ENSG00000183878.15"

# %%
res_all = pd.DataFrame(
    {
        f.stem.split("_data_")[1]: {
            "cm": cm(data[gene0_id], data[gene1_id]),
            "pearson": pearsonr(data[gene0_id], data[gene1_id])[0],
            "spearman": spearmanr(data[gene0_id], data[gene1_id])[0],
        }
        for f in TISSUE_DIR.glob("*.pkl")
        if (data := pd.read_pickle(f).T[[gene0_id, gene1_id]].dropna()) is not None
        and data.shape[0] > 10
    }
).T.abs()

# %%
res_all.shape

# %%
res_all.head()

# %%
res_all.sort_values("cm")

# %%
res_all.sort_values("pearson")

# %%
res_all.sort_values("spearman")

# %% [markdown]
# # Get GTEx sample metadata

# %%
gtex_samples_list = gtex_samples.index.to_list()
display(gtex_samples_list[:5])

# %%
_tmp_cols_df = pd.Series(gtex_samples_list).rename("SAMPID")

# %%
_tmp_cols_df

# %%
_tmp_cols_subjects = _tmp_cols_df.str.extract(
    r"([\w\d]+\-[\w\d]+)", flags=re.IGNORECASE, expand=True
)[0].rename("SUBJID")

# %%
_tmp_cols_subjects

# %%
_tmp_cols_df = pd.concat([_tmp_cols_df, _tmp_cols_subjects], axis=1)

# %%
_tmp_cols_df

# %%
gtex_phenotypes

# %%
_tmp_cols_df = pd.merge(_tmp_cols_df, gtex_phenotypes).set_index("SAMPID")

# %%
_tmp_cols_df = _tmp_cols_df.replace(
    {
        "SEX": {
            1: "Male",
            2: "Female",
        }
    }
)

# %%
_tmp_cols_df


# %% [markdown]
# # Plot

# %%
def plot_gene_pair(
    tissue_name, gene0, gene1, hue=None, kind="hex", ylim=None, bins="log"
):
    # gene0, gene1 = top_pairs_df.iloc[idx].name
    # display((gene0, gene1))

    tissue_file = get_tissue_file(tissue_name)
    tissue_data = pd.read_pickle(tissue_file).T[[gene0, gene1]]
    n_samples = tissue_data.shape[0]
    tissue_data = pd.merge(
        tissue_data,
        _tmp_cols_df,
        how="inner",
        left_index=True,
        right_index=True,
        validate="one_to_one",
    )
    # tissue_data = pd.concat([tissue_data, _tmp_cols_df], axis=1, verify_integrity=True)
    # if n_samples 1= tissue_data.shape[0]:
    #     print("WARNING: merging failed: {}

    gene0_symbol, gene1_symbol = gene_map[gene0], gene_map[gene1]
    display((gene0_symbol, gene1_symbol))

    # _pearson, _spearman, _clustermatch = top_pairs_df.loc[
    #     (gene0, gene1), ["pearson", "spearman", "clustermatch"]
    # ].tolist()

    _clustermatch = cm(tissue_data[gene0], tissue_data[gene1])
    _pearson = pearsonr(tissue_data[gene0], tissue_data[gene1])[0]
    _spearman = spearmanr(tissue_data[gene0], tissue_data[gene1])[0]

    _title = f"{tissue_name}\nClustermatch: {_clustermatch:.2f}\nPearson/Spearman: {_pearson:.2f}/{_spearman:.2f}"

    other_args = {
        "kind": kind,  # if hue is None else "scatter",
    }
    if hue is None:
        other_args["bins"] = bins

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
    p.ax_joint.set_xlabel(f"{gene_x_id}\n{gene_x_symbol}")

    gene_y_id = p.ax_joint.get_ylabel()
    gene_y_symbol = gene_map[gene_y_id]
    p.ax_joint.set_ylabel(f"{gene_y_id}\n{gene_y_symbol}")

    p.fig.suptitle(_title)

    return tissue_data


# %%
def get_tissue_file(name):
    for f in TISSUE_DIR.glob("*.pkl"):
        if name in f.name:
            return f


# %%
get_tissue_file("whole_blood")

# %% [markdown]
# ## In whole blood (where this gene pair was found)

# %%
plot_gene_pair(
    "whole_blood",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% [markdown] tags=[]
# ## Lowest tissues in clustermatch

# %%
plot_gene_pair(
    "uterus",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %%
# plot_gene_pair(
#     "fallopian_tube",
#     gene0_id,
#     gene1_id,
#     hue="SEX",
#     kind="scatter",
# )

# %%
# plot_gene_pair(
#     "kidney_medulla",
#     gene0_id,
#     gene1_id,
#     hue="SEX",
#     kind="scatter",
# )

# %%
plot_gene_pair(
    "ovary",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %%
plot_gene_pair(
    "vagina",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %%
# plot_gene_pair(
#     "cervix_endocervix",
#     gene0_id,
#     gene1_id,
#     hue="SEX",
#     kind="scatter",
# )

# %%
plot_gene_pair(
    "brain_cerebellum",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %%
plot_gene_pair(
    "small_intestine_terminal_ileum",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %%
plot_gene_pair(
    "brain_spinal_cord_cervical_c1",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %%
plot_gene_pair(
    "testis",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% [markdown] tags=[]
# ## Highest tissues in clustermatch

# %%
plot_gene_pair(
    "cells_cultured_fibroblasts",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %%
plot_gene_pair(
    "breast_mammary_tissue",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% [markdown]
# ## Pearson low, Clustermatch high

# %%
plot_gene_pair(
    "brain_anterior_cingulate_cortex_ba24",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %%
plot_gene_pair(
    "brain_amygdala",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %%
plot_gene_pair(
    "brain_frontal_cortex_ba9",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %%
plot_gene_pair(
    "bladder",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %%
plot_gene_pair(
    "heart_atrial_appendage",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %% [markdown]
# ## Spearman low, Clustermatch high

# %%
plot_gene_pair(
    "heart_left_ventricle",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %%
plot_gene_pair(
    "adipose_visceral_omentum",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %%
plot_gene_pair(
    "skin_not_sun_exposed_suprapubic",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %%
plot_gene_pair(
    "pancreas",
    gene0_id,
    gene1_id,
    hue="SEX",
    kind="scatter",
)

# %%
