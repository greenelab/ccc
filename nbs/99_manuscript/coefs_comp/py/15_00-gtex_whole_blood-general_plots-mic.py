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
# It generates different general plots to compare coefficient values from Pearson, Spearman, Clustermatch and Maximal Information Coefficient (MIC), such as their distribution. This notebook focuses on MIC.
#
# In `Settings` below, the data set and other options (such as tissue for GTEx) are specified.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd
from scipy import stats
import seaborn as sns

from clustermatch.plots import plot_histogram, plot_cumulative_histogram, jointplot
from clustermatch import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
DATASET_CONFIG = conf.GTEX
GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"

# %% tags=[]
# this is used for the cumulative histogram
GENE_PAIRS_PERCENT = 0.70

# %%
CLUSTERMATCH_LABEL = "CCC"
PEARSON_LABEL = "Pearson"
SPEARMAN_LABEL = "Spearman"
MIC_LABEL = "MIC"

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
OUTPUT_FIGURE_DIR = COEF_COMP_DIR / f"gtex_{GTEX_TISSUE}" / "mic"
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% tags=[]
COMPARISONS_DIR = DATASET_CONFIG["RESULTS_DIR"] / "comparison_others"
display(COMPARISONS_DIR)

# %% tags=[]
INPUT_FILE = COMPARISONS_DIR / f"gtex_v8_data_{GTEX_TISSUE}-{GENE_SEL_STRATEGY}-all.pkl"
display(INPUT_FILE)

assert INPUT_FILE.exists()

# %% [markdown] tags=[]
# # Data

# %% tags=[]
corrs_df = pd.read_pickle(INPUT_FILE).rename(
    columns={
        "clustermatch": CLUSTERMATCH_LABEL,
        "pearson": PEARSON_LABEL,
        "spearman": SPEARMAN_LABEL,
        "mic": MIC_LABEL,
    }
)

# %% tags=[]
corrs_df.shape

# %% tags=[]
corrs_df.head()

# %% [markdown] tags=[]
# ## Data stats

# %% tags=[]
corrs_df.describe().applymap(str)

# %% tags=[]
# skewness
corrs_df.apply(lambda x: stats.skew(x))

# %% [markdown]
# # MIC subset: all gene pairs

# %%
# this is supposed to be one of the values of column "mic_subset"
mic_subset = "all"

# %% [markdown] tags=[]
# ## Select MIC subset

# %%
df = corrs_df

# %%
df.shape

# %%
df.head()

# %% [markdown] tags=[]
# ## Histogram plot

# %% tags=[]
with sns.plotting_context("talk", font_scale=1.0):
    plot_histogram(df, output_dir=OUTPUT_FIGURE_DIR, fill=False)

# %% [markdown] tags=[]
# The distribution of CCC and MIC are very similar

# %% [markdown] tags=[]
# ## Cumulative histogram plot

# %% [markdown] tags=[]
# I include also a cumulative histogram without specifying `bins`.

# %% tags=[]
with sns.plotting_context("talk", font_scale=1.0):
    plot_cumulative_histogram(df, GENE_PAIRS_PERCENT, output_dir=OUTPUT_FIGURE_DIR)

# %% [markdown] tags=[]
# ## Joint plots comparing each coefficient

# %% tags=[]
with sns.plotting_context("talk", font_scale=1.0):
    jointplot(
        data=df,
        x=PEARSON_LABEL,
        y=MIC_LABEL,
        add_corr_coefs=False,
        output_dir=OUTPUT_FIGURE_DIR,
    )

# %% tags=[]
with sns.plotting_context("talk", font_scale=1.0):
    x, y = SPEARMAN_LABEL, MIC_LABEL

    g = jointplot(
        data=df,
        x=x,
        y=y,
        add_corr_coefs=False,
    )

    sns.despine(ax=g.ax_joint, left=True)
    g.ax_joint.set_yticks([])
    g.ax_joint.set_ylabel(None)

    g.savefig(
        OUTPUT_FIGURE_DIR / f"dist-{x.lower()}_vs_{y.lower()}.svg",
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
    )

# %% tags=[]
with sns.plotting_context("talk", font_scale=1.0):
    x, y = CLUSTERMATCH_LABEL, MIC_LABEL

    g = jointplot(
        data=df,
        x=x,
        y=y,
        add_corr_coefs=False,
    )

    sns.despine(ax=g.ax_joint, left=True)
    g.ax_joint.set_yticks([])
    g.ax_joint.set_ylabel(None)

    g.savefig(
        OUTPUT_FIGURE_DIR / f"dist-{x.lower()}_vs_{y.lower()}.svg",
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
    )

# %% [markdown] tags=[]
# ## Compute correlations

# %% [markdown] tags=[]
# These are the correlation between the correlation values (!). The idea is to see how coefficient match.

# %%
df.corr()

# %%
df.corr("spearman")

# %% [markdown] tags=[]
# # Create final figure

# %%
from svgutils.compose import Figure, SVG, Panel, Text

# %%
Figure(
    "64.371cm",
    "42.766cm",
    # white background
    Panel(
        SVG(COEF_COMP_DIR / "white_background.svg"),
    )
    .scale(0.5)
    .move(0, 0),
    Panel(
        SVG(OUTPUT_FIGURE_DIR / "dist-histograms.svg").scale(0.05),
        Text("a)", 0.2, 1, size=0.9, weight="bold"),
    ),
    Panel(
        SVG(OUTPUT_FIGURE_DIR / "dist-cum_histograms.svg").scale(0.05),
        Text("b)", 0.2, 1, size=0.9, weight="bold"),
    ).move(32, 0),
    Panel(
        SVG(OUTPUT_FIGURE_DIR / "dist-pearson_vs_mic.svg").scale(0.0595),
        Panel(
            SVG(OUTPUT_FIGURE_DIR / "dist-spearman_vs_mic.svg")
            .scale(0.0595)
            .move(21.5, 0)
        ),
        Panel(SVG(OUTPUT_FIGURE_DIR / "dist-ccc_vs_mic.svg").scale(0.0595).move(46, 0)),
        Text("c)", 0.2, 1, size=0.9, weight="bold"),
    ).move(0, 22),
).save(OUTPUT_FIGURE_DIR / "dist-main.svg")

# %% [markdown]
# Compile the manuscript with manubot and make sure the image has a white background and displays properly.

# %%
