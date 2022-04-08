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

# %%
# # %load_ext rpy2.ipython

# %% tags=[]
import json
import tempfile
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from clustermatch import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
# DATASET_CONFIG = conf.GTEX
# GTEX_TISSUE = "whole_blood"
# GENE_SEL_STRATEGY = "var_pc_log2"

# %% tags=[]
N_TOP_GENE_PAIRS = 100

# %%
# CLUSTERMATCH_LABEL = "Clustermatch"
# PEARSON_LABEL = "Pearson"
# SPEARMAN_LABEL = "Spearman"

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None and conf.MANUSCRIPT["BASE_DIR"].exists()
), "Manuscript dir not set"

# %% tags=[]
COEF_COMP_DIR = conf.MANUSCRIPT["FIGURES_DIR"] / "coefs_comp"
COEF_COMP_DIR.mkdir(parents=True, exist_ok=True)
display(COEF_COMP_DIR)

# %% tags=[]
OUTPUT_FIGURE_DIR = COEF_COMP_DIR / "giant_networks"
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% tags=[]
# INPUT_DIR = DATASET_CONFIG["GENE_PAIR_INTERSECTIONS"]
# display(INPUT_DIR)

# assert INPUT_DIR.exists()

# %% tags=[]
OUTPUT_DIR = conf.GIANT["RESULTS_DIR"] / "intersection_genes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Summarize

# %%
all_subsets_dfs = []

# %% [markdown]
# ## Clustermatch vs Pearson

# %%
subset = "clustermatch_vs_pearson"

# %%
subset_files = sorted(list((OUTPUT_DIR / subset).glob("*.h5")))
display(len(subset_files))
display(subset_files[:3])

# %%
_dfs = [
    pd.read_hdf(f, key="data").assign(
        gene_pair=f.name.split("-")[1].split(".")[0].upper(),
        tissue=pd.read_hdf(f, key="metadata")["tissue"].squeeze(),
        subset=subset,
    )
    for f in subset_files
]

display(_dfs[0].head())
all_subsets_dfs.extend(_dfs)
display(len(all_subsets_dfs))

# %% [markdown]
# ## Clustermatch vs Pearson/Spearman

# %%
subset = "clustermatch_vs_pearson_spearman"

# %%
subset_files = sorted(list((OUTPUT_DIR / subset).glob("*.h5")))
display(len(subset_files))
display(subset_files[:3])

# %%
_dfs = [
    pd.read_hdf(f, key="data").assign(
        gene_pair=f.name.split("-")[1].split(".")[0].upper(),
        tissue=pd.read_hdf(f, key="metadata")["tissue"].squeeze(),
        subset=subset,
    )
    for f in subset_files
]

display(_dfs[0].head())
all_subsets_dfs.extend(_dfs)
display(len(all_subsets_dfs))

# %% [markdown]
# ## Clustermatch vs Spearman

# %%
subset = "clustermatch_vs_spearman"

# %%
subset_files = sorted(list((OUTPUT_DIR / subset).glob("*.h5")))
display(len(subset_files))
display(subset_files[:3])

# %%
_dfs = [
    pd.read_hdf(f, key="data").assign(
        gene_pair=f.name.split("-")[1].split(".")[0].upper(),
        tissue=pd.read_hdf(f, key="metadata")["tissue"].squeeze(),
        subset=subset,
    )
    for f in subset_files
]

display(_dfs[0].head())
all_subsets_dfs.extend(_dfs)
display(len(all_subsets_dfs))

# %% [markdown]
# ## Pearson vs Clustermatch

# %%
subset = "pearson_vs_clustermatch"

# %%
subset_files = sorted(list((OUTPUT_DIR / subset).glob("*.h5")))
display(len(subset_files))
display(subset_files[:3])

# %%
_dfs = [
    pd.read_hdf(f, key="data").assign(
        gene_pair=f.name.split("-")[1].split(".")[0].upper(),
        tissue=pd.read_hdf(f, key="metadata")["tissue"].squeeze(),
        subset=subset,
    )
    for f in subset_files
]

display(_dfs[0].head())
all_subsets_dfs.extend(_dfs)
display(len(all_subsets_dfs))

# %% [markdown]
# ## Pearson vs Clustermatch/Spearman

# %%
subset = "pearson_vs_clustermatch_spearman"

# %%
subset_files = sorted(list((OUTPUT_DIR / subset).glob("*.h5")))
display(len(subset_files))
display(subset_files[:3])

# %%
_dfs = [
    pd.read_hdf(f, key="data").assign(
        gene_pair=f.name.split("-")[1].split(".")[0].upper(),
        tissue=pd.read_hdf(f, key="metadata")["tissue"].squeeze(),
        subset=subset,
    )
    for f in subset_files
]

display(_dfs[0].head())
all_subsets_dfs.extend(_dfs)
display(len(all_subsets_dfs))

# %% [markdown]
# # Combine

# %%
df = pd.concat(all_subsets_dfs, ignore_index=True)

# %%
df.shape

# %%
df.head()

# %% [markdown]
# # Stats

# %%
df_stats = df.groupby(["subset"])["gene_pair"].nunique()
display(df_stats)

# %%
assert df_stats["clustermatch_vs_pearson"] == N_TOP_GENE_PAIRS
assert df_stats["pearson_vs_clustermatch"] == N_TOP_GENE_PAIRS
assert df_stats["pearson_vs_clustermatch_spearman"] == N_TOP_GENE_PAIRS

# %% [markdown]
# # Combine subsets

# %%
subset_cm_vs_rest = "clustermatch_vs_rest"
subset_p_vs_rest = "pearson_vs_rest"

# %%
df = df.replace(
    {
        "subset": {
            "clustermatch_vs_pearson": subset_cm_vs_rest,
            "clustermatch_vs_spearman": subset_cm_vs_rest,
            "clustermatch_vs_pearson_spearman": subset_cm_vs_rest,
            "pearson_vs_clustermatch": subset_p_vs_rest,
            "pearson_vs_clustermatch_spearman": subset_p_vs_rest,
        },
    }
)

# %%
df.head()

# %%
assert df["subset"].unique().shape[0] == 2

# %% [markdown]
# # Analyses

# %%
plot_stats = df.groupby(["subset", "tissue"])["gene_pair"].nunique()

# %%
plot_stats.head()

# %%
plot_stats = (
    plot_stats.groupby("subset").apply(lambda grp: grp.nlargest(5)).droplevel(0)
)

# by percentage
# plot_stats = plot_stats.groupby("subset").apply(lambda x: x / x.sum())

# %%
plot_stats

# %%
plot_stats.index.get_level_values("tissue").unique().shape

# %%
plot_stats = plot_stats.reset_index()
display(plot_stats.head())

# %%
plot_stats["subset"].unique()

# %%
plot_stats["tissue"].unique()

# %% [markdown]
# # Plots

# %%
PREDICTED_TISSUE_LABEL = "Predicted tissue/cell type\nin GIANT"
N_GENE_PAIRS_LABEL = "Number of gene pairs"
AVG_PROB_INTERACTION_LABEL = (
    "Average probability of interaction\nin tissue-specific networks"
)

# %% [markdown]
# ## Tissues

# %%
subset_renames = {
    "clustermatch_vs_rest": "CCC vs others",
    # "clustermatch_vs_pearson_spearman": "CCC vs Pearson/Spearman",
    # "clustermatch_vs_spearman": "CCC vs Spearman",
    "pearson_vs_rest": "Pearson vs others",
    # "pearson_vs_clustermatch_spearman": "Pearson vs CCC/Spearman",
}

tissue_renames = {
    "blood": "Blood",
    "mononuclear-phagocyte": "Mononuclear phagocyte",
    "natural-killer-cell": "Natural killer cell",
    "leukocyte": "Leukocyte",
    "macrophage": "Macrophage",
    "central-nervous-system": "Central nervous system",
    "granulocyte": "Granulocyte",
    "b-lymphocyte": "B-lymphocyte",
    "skeletal-muscle": "Skeletal muscle",
    "liver": "Liver",
    "placenta": "Placenta",
    "renal-tubule": "Renal tubule",
    "placenta": "Placenta",
    "placenta": "Placenta",
}

# %%
plot_data = plot_stats.replace(
    {
        "subset": subset_renames,
        "tissue": tissue_renames,
    }
)

# %%
blood_related_tissues = set(
    [
        "Blood",
        "Mononuclear phagocyte",
        "Natural killer cell",
        "Leukocyte",
        "Macrophage",
        "Granulocyte",
        "B-lymphocyte",
    ]
)

# %%
tissues_order = [
    "Blood",
    "Mononuclear phagocyte",
    "Natural killer cell",
    "Leukocyte",
    "Macrophage",
    # "Granulocyte",
    # "B-lymphocyte",
    "Skeletal muscle",
    "Liver",
    "Placenta",
    # "Renal tubule",
    # "Central nervous system",
]

# %%
deep_colors = sns.color_palette("deep")
display(deep_colors)

# %%
blood_color = deep_colors[3]
others_color = deep_colors[0]

# %%
tissue_colors = {
    t: blood_color if t in blood_related_tissues else others_color
    for t in tissues_order
}

# %%
# first plot to show order of subset
with sns.plotting_context("paper", font_scale=1.0):
    g = sns.FacetGrid(
        plot_data,
        row="subset",
        sharex=True,
        sharey=True,
        height=2.0,
        aspect=2.1,
    )
    g.map(
        sns.barplot, "tissue", "gene_pair", order=tissues_order, palette=tissue_colors
    )
    g.set_xticklabels(rotation=35, ha="right")
    g.set_axis_labels(PREDICTED_TISSUE_LABEL, "")

    g.fig.text(0, 0.30, N_GENE_PAIRS_LABEL, rotation=90)

# %%
# now again without row titles and save the plot
with sns.plotting_context("paper", font_scale=1.0):
    g = sns.FacetGrid(
        plot_data,
        row="subset",
        sharex=True,
        sharey=True,
        height=2.0,
        aspect=2.1,
    )
    g.map(
        sns.barplot, "tissue", "gene_pair", order=tissues_order, palette=tissue_colors
    )
    g.set_xticklabels(rotation=30, ha="right")
    g.set_axis_labels(PREDICTED_TISSUE_LABEL, "")

    g.set_titles(row_template="")

    g.fig.text(0, 0.30, N_GENE_PAIRS_LABEL, rotation=90)

    g.savefig(
        OUTPUT_FIGURE_DIR / "top_gene_pairs-tissue_count.svg",
        bbox_inches="tight",
        dpi=300,
        # facecolor="white",
        transparent=True,
    )

# %% [markdown]
# ## Connectivity

# %%
plot_data = pd.merge(
    df,
    plot_stats,
    left_on=["subset", "tissue"],
    right_on=["subset", "tissue"],
    how="inner",
)

# %%
plot_data = plot_data.replace(
    {
        "subset": subset_renames,
        "tissue": tissue_renames,
    }
)

# %%
plot_data.shape

# %%
plot_data.head()

# %%
# g = sns.catplot(data=plot_df, x="subset", y="weight", kind="box", orient="v", hue="tissue", height=5, aspect=1.5, sharex=False)
# g.set_xticklabels(rotation=30)

# %%
with sns.plotting_context("paper", font_scale=1.0):
    g = sns.FacetGrid(
        plot_data,
        row="subset",
        sharex=True,
        sharey=True,
        height=2.0,
        aspect=2.1,
    )
    g.map(sns.boxplot, "tissue", "weight", order=tissues_order, palette=tissue_colors)
    g.set_xticklabels(rotation=30, ha="right")
    g.set_axis_labels(PREDICTED_TISSUE_LABEL, "")

    g.set_titles(row_template="")

    for ax in g.axes:
        ax = ax[0]
        sns.despine(ax=ax, left=True, right=False)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    g.fig.text(1.09, 0.25, AVG_PROB_INTERACTION_LABEL, rotation=90, ha="center")
    # g.axes[2][0].set_ylabel(AVG_PROB_INTERACTION_LABEL)

    g.savefig(
        OUTPUT_FIGURE_DIR / "top_gene_pairs-tissue_avg_weight.svg",
        bbox_inches="tight",
        dpi=300,
        # facecolor="white",
        transparent=True,
    )

# %% [markdown] tags=[]
# # Raw numbers

# %%
plot_data.groupby(["subset", "tissue"])["weight"].describe()

# %% [markdown] tags=[]
# # Create final figure

# %%
from svgutils.compose import Figure, SVG, Panel, Text

# %%
Figure(
    "309.02272cm",
    "164.7096cm",
    SVG(OUTPUT_FIGURE_DIR / "top_gene_pairs-tissue_count.svg").scale(0.5),
    # cm vs rest
    SVG(COEF_COMP_DIR / "triangles-c_vs_p.svg").scale(1.50).move(35, 4),
    Text("+", 67, 10, size=6),
    SVG(COEF_COMP_DIR / "triangles-c_vs_ps.svg").scale(1.50).move(75, 4),
    Text("+", 108, 10, size=6),
    SVG(COEF_COMP_DIR / "triangles-c_vs_s.svg").scale(1.50).move(115, 4),
    # p vs rest
    SVG(COEF_COMP_DIR / "triangles-p_vs_c.svg").scale(1.50).move(35, 65),
    Text("+", 67, 71, size=6),
    SVG(COEF_COMP_DIR / "triangles-p_vs_cs.svg").scale(1.50).move(75, 65),
    # another
    SVG(OUTPUT_FIGURE_DIR / "top_gene_pairs-tissue_avg_weight.svg")
    .scale(0.5)
    .move(130, 0),
).save(OUTPUT_FIGURE_DIR / "top_gene_pairs-main.svg")

# %% [markdown]
# Now open `top_gene_pairs-main.svg`, reside to fit drawing to page, and add a white rectangle to the background.

# %% [markdown]
# I think it's important to open the file with Inkscape and save it, just to make sure the content is right.
# Because sometimes Inkscape crashed when opening it.

# %%
