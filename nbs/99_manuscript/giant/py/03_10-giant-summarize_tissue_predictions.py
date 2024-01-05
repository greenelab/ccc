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
# Creates a plot to summarize predicted cell types and interaction probabilities across the top genes for each correlation coefficients.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd
import seaborn as sns

from ccc import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
N_TOP_GENE_PAIRS = 100
N_TOP_TISSUES = 5

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
OUTPUT_DIR = conf.GIANT["RESULTS_DIR"] / "intersection_genes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Summarize

# %% tags=[]
all_subsets_dfs = []


# %% [markdown] tags=[]
# ## Functions

# %% tags=[]
def read_hdf(filepath, subset):
    return pd.read_hdf(filepath, key="data").assign(
        order=int(filepath.name.split("-")[0]),
        gene_pair=filepath.name.split("-")[1].split(".")[0].upper(),
        tissue=pd.read_hdf(filepath, key="metadata")["tissue"].squeeze(),
        subset=subset,
    )


# %% [markdown] tags=[]
# ## CCC vs Pearson

# %% tags=[]
subset = "clustermatch_vs_pearson"

# %% tags=[]
subset_files = sorted(list((OUTPUT_DIR / subset).glob("*.h5")))
display(len(subset_files))
display(subset_files[:3])

# %% tags=[]
_dfs = [read_hdf(f, subset) for f in subset_files]

display(_dfs[0].head())
all_subsets_dfs.extend(_dfs)
display(len(all_subsets_dfs))

# %% [markdown] tags=[]
# ## CCC vs Pearson/Spearman

# %% tags=[]
subset = "clustermatch_vs_pearson_spearman"

# %% tags=[]
subset_files = sorted(list((OUTPUT_DIR / subset).glob("*.h5")))
display(len(subset_files))
display(subset_files[:3])

# %% tags=[]
_dfs = [read_hdf(f, subset) for f in subset_files]

display(_dfs[0].head())
all_subsets_dfs.extend(_dfs)
display(len(all_subsets_dfs))

# %% [markdown] tags=[]
# ## CCC vs Spearman

# %% tags=[]
subset = "clustermatch_vs_spearman"

# %% tags=[]
subset_files = sorted(list((OUTPUT_DIR / subset).glob("*.h5")))
display(len(subset_files))
display(subset_files[:3])

# %% tags=[]
_dfs = [read_hdf(f, subset) for f in subset_files]

display(_dfs[0].head())
all_subsets_dfs.extend(_dfs)
display(len(all_subsets_dfs))

# %% [markdown] tags=[]
# ## Pearson vs CCC

# %% tags=[]
subset = "pearson_vs_clustermatch"

# %% tags=[]
subset_files = sorted(list((OUTPUT_DIR / subset).glob("*.h5")))
display(len(subset_files))
display(subset_files[:3])

# %% tags=[]
_dfs = [read_hdf(f, subset) for f in subset_files]

display(_dfs[0].head())
all_subsets_dfs.extend(_dfs)
display(len(all_subsets_dfs))

# %% [markdown] tags=[]
# ## Pearson vs CCC/Spearman

# %% tags=[]
subset = "pearson_vs_clustermatch_spearman"

# %% tags=[]
subset_files = sorted(list((OUTPUT_DIR / subset).glob("*.h5")))
display(len(subset_files))
display(subset_files[:3])

# %% tags=[]
_dfs = [read_hdf(f, subset) for f in subset_files]

display(_dfs[0].head())
all_subsets_dfs.extend(_dfs)
display(len(all_subsets_dfs))

# %% [markdown] tags=[]
# # Combine

# %% tags=[]
df = pd.concat(all_subsets_dfs, ignore_index=True)

# %% tags=[]
df.shape

# %% tags=[]
df.head()

# %% tags=[]
df["order"].unique()

# %% [markdown] tags=[]
# # Stats

# %% tags=[]
df_stats = df.groupby(["subset"])["gene_pair"].nunique()
display(df_stats)

# %% tags=[]
assert df_stats["clustermatch_vs_pearson"] == N_TOP_GENE_PAIRS
assert df_stats["pearson_vs_clustermatch"] == N_TOP_GENE_PAIRS
assert df_stats["pearson_vs_clustermatch_spearman"] == N_TOP_GENE_PAIRS

# %% [markdown] tags=[]
# # Combine subsets

# %% tags=[]
subset_cm_vs_rest = "clustermatch_vs_rest"
subset_p_vs_rest = "pearson_vs_rest"

# %% tags=[]
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

# %% tags=[]
df.head()

# %% tags=[]
assert df["subset"].unique().shape[0] == 2

# %% [markdown] tags=[]
# # Analyses

# %% [markdown] tags=[]
# Here we read the gene pair queries sent to GIANT to see 1) which cell type were predicted for each of them and 2) the characteristics of the predicted networks for each gene pair.
#
# This allows us to assess whether gene pairs found by our correlation coefficients in GTEx (whole blood) replicate in GIANT by seeing if the predicted cell types is a blood cell lineage (leukocyte, etc) and the network connectivity is high.

# %% tags=[]
# for each subset (ccc vs etc, pearson vs etc), sort by top gene pairs
# for that, for each subset, sort by "order", which indicates, for each subset
# the gene pairs with the largest correlation value
top_gene_pairs = df.groupby(["subset"], group_keys=False).apply(
    lambda x: x.sort_values(["order", "gene_pair"], ascending=True)[
        "gene_pair"
    ].unique()
)

# %% tags=[]
top_gene_pairs

# %% tags=[]
# now use the order of gene pairs within subsets to actually select the top ones
top_df = df.groupby(["subset"], group_keys=False).apply(
    lambda x: x[x["gene_pair"].isin(top_gene_pairs.loc[x.name][:N_TOP_GENE_PAIRS])]
)

# %% tags=[]
top_df.shape

# %% tags=[]
top_df.head()

# %% tags=[]
plot_stats = top_df.groupby(["subset", "tissue"])["gene_pair"].nunique()

# %% tags=[]
plot_stats.head()

# %% tags=[]
plot_stats = (
    plot_stats.groupby("subset")
    .apply(lambda grp: grp.nlargest(N_TOP_TISSUES))
    .droplevel(0)
).rename("n_gene_pairs")

# by percentage
# plot_stats = plot_stats.groupby("subset").apply(lambda x: x / x.sum())

# %% tags=[]
plot_stats

# %% tags=[]
# sum gene pairs, it should be less than 100 (100 is the total number of gene pairs taken)
tmp = plot_stats.groupby("subset").sum()
display(tmp)
assert (tmp < N_TOP_GENE_PAIRS).all()

# %% tags=[]
plot_stats.index.get_level_values("tissue").unique().shape

# %% tags=[]
plot_stats = plot_stats.reset_index()
display(plot_stats.head())

# %% tags=[]
plot_stats["subset"].unique()

# %% tags=[]
plot_stats["tissue"].unique()

# %% [markdown] tags=[]
# # Plots

# %% tags=[]
PREDICTED_TISSUE_LABEL = "Predicted tissue/cell type\nin GIANT"
N_GENE_PAIRS_LABEL = "Number of gene pairs"
AVG_PROB_INTERACTION_LABEL = (
    "Average probability of interaction\nin tissue-specific networks"
)

# %% tags=[]
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
    "granulocyte": "Granulocyte",
    "skeletal-muscle": "Skeletal muscle",
    "liver": "Liver",
    "placenta": "Placenta",
}

# %% [markdown] tags=[]
# ## First plot version

# %% [markdown] tags=[]
# These first plot versions are drafts, just to see how the pattern are.
# The final plots are generated in the `Second plot version` below.

# %% tags=[]
count_data = plot_stats.replace(
    {
        "subset": subset_renames,
        "tissue": tissue_renames,
    }
)

# %% [markdown] tags=[]
# ### Tissues order

# %% tags=[]
blood_related_tissues = set(
    [
        "Blood",
        "Mononuclear phagocyte",
        "Natural killer cell",
        "Leukocyte",
        "Macrophage",
    ]
)

# %% tags=[]
tissues_order = [
    "Blood",
    "Mononuclear phagocyte",
    "Natural killer cell",
    "Leukocyte",
    "Macrophage",
    "Skeletal muscle",
    "Liver",
    "Placenta",
]

# %% [markdown] tags=[]
# ### Tissues colors

# %% tags=[]
deep_colors = sns.color_palette("tab10")
display(deep_colors)

# %% tags=[]
blood_color = deep_colors[3]
others_color = deep_colors[0]

# %% tags=[]
tissue_colors = {
    t: blood_color if t in blood_related_tissues else others_color
    for t in tissues_order
}

# %% [markdown] tags=[]
# ### Plot: number of gene pairs by tissue and method

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.0):
    g = sns.FacetGrid(
        count_data,
        row="subset",
        sharex=True,
        sharey=True,
        height=2.0,
        aspect=2.1,
    )
    g.map(
        sns.barplot,
        "tissue",
        "n_gene_pairs",
        order=tissues_order,
        palette=tissue_colors,
    )
    g.set_xticklabels(rotation=35, ha="right")
    g.set_axis_labels(PREDICTED_TISSUE_LABEL, "")

    g.fig.text(0, 0.30, N_GENE_PAIRS_LABEL, rotation=90)

# %% [markdown] tags=[]
# ### Plot: gene networks connectivity by tissue and method

# %% tags=[]
conn_data = pd.merge(
    top_df,
    plot_stats,
    left_on=["subset", "tissue"],
    right_on=["subset", "tissue"],
    how="inner",
)

# %% tags=[]
conn_data = conn_data.replace(
    {
        "subset": subset_renames,
        "tissue": tissue_renames,
    }
)

# %% tags=[]
conn_data.shape

# %% tags=[]
conn_data.head()

# %% tags=[]
# # only keep connections with query genes
# conn_data = conn_data.assign(query_genes=conn_data["gene_pair"].apply(lambda x: set(x.split("_"))))

# %% tags=[]
# conn_data = conn_data[conn_data.apply(
#     lambda x:
#         (x["gene1"] in x["query_genes"]) | (x["gene2"] in x["query_genes"]),
#     axis=1
# )]

# %% tags=[]
# conn_data.shape

# %% tags=[]
# conn_data.head()

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.0):
    g = sns.FacetGrid(
        conn_data,
        row="subset",
        sharex=True,
        sharey=True,
        height=2.0,
        aspect=2.1,
    )
    g.map(sns.boxplot, "tissue", "weight", order=tissues_order, palette=tissue_colors)
    g.set_xticklabels(rotation=30, ha="right")
    g.set_axis_labels(PREDICTED_TISSUE_LABEL, "")

    for ax in g.axes:
        ax = ax[0]
        sns.despine(ax=ax, left=True, right=False)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    g.fig.text(1.10, 0.25, AVG_PROB_INTERACTION_LABEL, rotation=90, ha="center")

# %% [markdown] tags=[]
# ## Second plot version

# %% [markdown] tags=[]
# ### Tissues order

# %% tags=[]
ccc_tissues = {
    "Natural killer cell",
    "Leukocyte",
    "Macrophage",
}

pcc_tissues = {
    "Skeletal muscle",
    "Liver",
    "Placenta",
}

shared_tissues = {
    "Blood",
    "Mononuclear phagocyte",
}

# make sure I'm not missing a tissue
assert set(tissues_order) == (ccc_tissues | pcc_tissues | shared_tissues)

# new tissues_order
tissues_order = [
    "Macrophage",
    "Leukocyte",
    "Natural killer cell",
    "Blood",
    "Mononuclear phagocyte",
    "Skeletal muscle",
    "Liver",
    "Placenta",
]

assert set(tissues_order) == (ccc_tissues | pcc_tissues | shared_tissues)

# %% [markdown] tags=[]
# ### Tissues colors

# %% tags=[]
display(deep_colors)

# %% tags=[]
# by specific or shared
ccc_color = deep_colors[3]
shared_color = deep_colors[4]
pcc_color = deep_colors[0]

# %% tags=[]
tissue_colors = {
    t: ccc_color
    if t in ccc_tissues
    else pcc_color
    if t in pcc_tissues
    else shared_color
    for t in tissues_order
}

# %% [markdown] tags=[]
# ### Plot: number of gene pairs by tissue and method

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.0):
    g = sns.FacetGrid(
        count_data,
        row="subset",
        sharex=True,
        sharey=True,
        height=2.0,
        aspect=2.1,
    )
    g.map(
        sns.barplot,
        "tissue",
        "n_gene_pairs",
        order=tissues_order,
        palette=tissue_colors,
    )
    g.set_xticklabels(rotation=30, ha="right")
    g.set_axis_labels(PREDICTED_TISSUE_LABEL, "")

    g.set_titles(row_template="")

    g.fig.text(0.03, 0.30, N_GENE_PAIRS_LABEL, rotation=90)

    g.savefig(
        OUTPUT_FIGURE_DIR / "top_gene_pairs-tissue_count.svg",
        bbox_inches="tight",
        dpi=300,
        # facecolor="white",
        transparent=True,
    )

# %% [markdown] tags=[]
# ### Plot: gene networks connectivity by tissue and method

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.0):
    g = sns.FacetGrid(
        conn_data,
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

    g.fig.text(1.10, 0.25, AVG_PROB_INTERACTION_LABEL, rotation=90, ha="center")
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

# %% tags=[]
count_data.sort_values(["subset", "n_gene_pairs"], ascending=[True, False])

# %% tags=[]
conn_data.groupby(["subset", "tissue"])["weight"].describe()

# %% [markdown] tags=[]
# # Create final figure

# %% tags=[]
from svgutils.compose import Figure, SVG, Panel, Text

# %% tags=[]
BLOOD_NETWORKS_DIR = OUTPUT_FIGURE_DIR / "blood_tissues" / "gene_pair_networks"
AUTO_SELECTED_NETWORKS_DIR = (
    OUTPUT_FIGURE_DIR / "auto_selected_tissues" / "gene_pair_networks"
)

# %% tags=[]
Figure(
    "30.50629cm",
    "24.44741cm",
    # white background
    Panel(
        SVG(COEF_COMP_DIR / "white_background.svg"),
    )
    .scale(0.5)
    .move(0, 0),
    Panel(
        Panel(
            SVG(BLOOD_NETWORKS_DIR / "GIANT-RASSF2_vs_CYTIP-blood.svg").move(10, 0),
            SVG(
                AUTO_SELECTED_NETWORKS_DIR / "GIANT-RASSF2_vs_CYTIP-leukocyte.svg"
            ).move(420, 0),
            SVG(COEF_COMP_DIR / "triangles-c_vs_s-RASSF2_CYTIP.svg")
            .scale(7.00)
            .move(20, 350),
            Text("a)", 0, 30, size=28, weight="bold"),
            SVG(BLOOD_NETWORKS_DIR / "color_bar.svg").scale(3.50).move(270, 410),
        )
        .scale(0.0175)
        .move(0, 0),
        Panel(
            SVG(BLOOD_NETWORKS_DIR / "GIANT-MYOZ1_vs_TNNI2-blood.svg").move(10, 0),
            SVG(
                AUTO_SELECTED_NETWORKS_DIR / "GIANT-MYOZ1_vs_TNNI2-skeletal_muscle.svg"
            ).move(420, 0),
            Text("b)", 0, 30, size=28, weight="bold"),
            SVG(COEF_COMP_DIR / "triangles-p_vs_c-MYOZ1_TNNI2.svg")
            .scale(7.00)
            .move(20, 350),
        )
        .scale(0.0175)
        .move(16, 0),
    ),
    Panel(
        SVG(OUTPUT_FIGURE_DIR / "top_gene_pairs-tissue_count.svg").scale(0.5),
        # cm vs rest
        SVG(COEF_COMP_DIR / "triangles-c_vs_p-c_signif.svg").scale(1.50).move(30, 2),
        Text("+", 52, 10, size=6),
        SVG(COEF_COMP_DIR / "triangles-c_vs_ps-c_signif.svg").scale(1.50).move(60, 2),
        Text("+", 83, 10, size=6),
        SVG(COEF_COMP_DIR / "triangles-c_vs_s-c_signif.svg").scale(1.50).move(90, 2),
        # p vs rest
        SVG(COEF_COMP_DIR / "triangles-p_vs_c-p_signif.svg").scale(1.50).move(30, 63),
        Text("+", 52, 71, size=6),
        SVG(COEF_COMP_DIR / "triangles-p_vs_cs-p_signif.svg").scale(1.50).move(60, 63),
        # another
        SVG(OUTPUT_FIGURE_DIR / "top_gene_pairs-tissue_avg_weight.svg")
        .scale(0.5)
        .move(130, 0),
        Text("c)", 0, 9, size=6, weight="bold"),
    )
    .scale(0.10)
    .move(0, 8),
).save(OUTPUT_FIGURE_DIR / "top_gene_pairs-main.svg")

# %% [markdown] tags=[]
# Compile the manuscript with manubot and make sure the image has a white background and displays properly.

# %% tags=[]
