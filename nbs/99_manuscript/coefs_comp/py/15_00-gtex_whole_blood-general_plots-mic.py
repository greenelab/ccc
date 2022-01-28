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

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None and conf.MANUSCRIPT["BASE_DIR"].exists()
), "Manuscript dir not set"

# %% tags=[]
OUTPUT_FIGURE_DIR = (
    conf.MANUSCRIPT["FIGURES_DIR"] / "coefs_comp" / f"gtex_{GTEX_TISSUE}" / "mic"
)
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
corrs_df = pd.read_pickle(INPUT_FILE)

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
corrs_df.drop(columns=["mic_subset"]).apply(lambda x: stats.skew(x))

# %% [markdown]
# # MIC subset: all gene pairs

# %%
# this is supposed to be one of the values of column "mic_subset"
mic_subset = "all"

# %% [markdown] tags=[]
# ## Select MIC subset

# %%
df = corrs_df[corrs_df["mic_subset"].isin((mic_subset,))].drop(columns=["mic_subset"])

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
# **UPDATE**
#
# Coefficients' values distribute very differently. Clustermatch is skewed to the left, whereas Pearson and specially Spearman seem more uniform.

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
        x="pearson",
        y="mic",
        add_corr_coefs=False,
        output_dir=OUTPUT_FIGURE_DIR,
    )

# %% tags=[]
with sns.plotting_context("talk", font_scale=1.0):
    x, y = "spearman", "mic"

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
        OUTPUT_FIGURE_DIR / f"dist-{x}_vs_{y}.svg",
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
    )

# %% tags=[]
with sns.plotting_context("talk", font_scale=1.0):
    x, y = "clustermatch", "mic"

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
        OUTPUT_FIGURE_DIR / f"dist-{x}_vs_{y}.svg",
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

# %%
