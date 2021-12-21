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
# It generates different general plots to compare coefficient values from Pearson, Spearman and Clustermatch, such as their distribution.
#
# In `Settings` below, the data set and other options (such as tissue for GTEx) are specified.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from clustermatch.coef import cm
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
OUTPUT_FIGURE_DIR = conf.MANUSCRIPT["FIGURES_DIR"] / "coefs_comp" / f"gtex_{GTEX_TISSUE}"
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% tags=[]
INPUT_CORR_FILE_TEMPLATE = (
    DATASET_CONFIG["SIMILARITY_MATRICES_DIR"]
    / DATASET_CONFIG["SIMILARITY_MATRIX_FILENAME_TEMPLATE"]
)
display(INPUT_CORR_FILE_TEMPLATE)

# %% tags=[]
INPUT_FILE = DATASET_CONFIG["SIMILARITY_MATRICES_DIR"] / str(
    INPUT_CORR_FILE_TEMPLATE
).format(
    tissue=GTEX_TISSUE,
    gene_sel_strategy=GENE_SEL_STRATEGY,
    corr_method="all",
)
display(INPUT_FILE)

assert INPUT_FILE.exists()

# %% [markdown] tags=[]
# # Data

# %%
df = pd.read_pickle(INPUT_FILE)

# %%
df.shape

# %%
df.head()

# %% [markdown] tags=[]
# ## Data stats

# %%
df.describe().applymap(str)

# %%
# skewness
df.apply(lambda x: stats.skew(x))

# %% [markdown]
# # Histogram plot

# %%
with sns.plotting_context("talk", font_scale=1.0):
    fig, ax = plt.subplots(figsize=(10, 7))

    ax = sns.histplot(
        data=df,
        stat="density",
        bins=100,
        common_bins=True,
        common_norm=False,
        kde=True,
        ax=ax,
    )
    sns.despine(ax=ax)

    ax.set_xticks(np.linspace(0, 1, 10 + 1))

    plt.savefig(
        OUTPUT_FIGURE_DIR / "dist-histograms.svg",
        bbox_inches="tight",
        facecolor="white",
    )

# %% [markdown]
# Coefficients' values distribute very differently. Clustermatch is skewed to the left, whereas Pearson and specially Spearman seem more uniform.

# %% [markdown]
# # Cumulative histogram plot

# %% [markdown]
# I include also a cumulative histogram without specifying `bins`.

# %%
_percent = 0.70
_coef_at_percent = df.quantile(_percent)
display(_coef_at_percent)

# %%
with sns.plotting_context("talk", font_scale=1.1):
    fig, ax = plt.subplots(figsize=(10, 7))

    ax = sns.histplot(
        data=df,
        element="step",
        fill=False,
        stat="percent",
        common_norm=False,
        cumulative=True,
        legend=False,
        ax=ax,
    )
    sns.despine(ax=ax)

    ax.set_xticks(np.linspace(0, 1, 10 + 1))
    ax.set_yticks(np.linspace(0, 100, 10 + 1))
    
    ax.set_ylabel("Percent of gene pairs")

    x_lim = ax.get_xlim()
    ax.hlines(
        y=_percent * 100,
        xmin=x_lim[0],
        xmax=_coef_at_percent["spearman"],
        color="gray",
        linestyle="dotted",
    )
    ax.vlines(
        x=_coef_at_percent["clustermatch"],
        ymin=0,
        ymax=_percent * 100,
        color="gray",
        linestyle="dotted",
    )
    ax.vlines(
        x=_coef_at_percent["pearson"],
        ymin=0,
        ymax=_percent * 100,
        color="gray",
        linestyle="dotted",
    )
    ax.vlines(
        x=_coef_at_percent["spearman"],
        ymin=0,
        ymax=_percent * 100,
        color="gray",
        linestyle="dotted",
    )

    ax.set_xlim(x_lim)

    plt.savefig(
        OUTPUT_FIGURE_DIR / "dist-cum_histograms.svg",
        bbox_inches="tight",
        facecolor="white",
    )

# %% [markdown]
# # Joint plots comparing each coefficient

# %%
import matplotlib as mpl
from matplotlib.colors import LogNorm
from seaborn.distributions import _freedman_diaconis_bins


# %%
def jointplot(data, x, y, bins=None):
    """
    Function based on Seaborn's jointplot, but without marginal plots.
    """

    # compute correlations
    x_values = df[x].to_numpy()
    y_values = df[y].to_numpy()
    r = stats.pearsonr(x_values, y_values)[0]
    rs = stats.spearmanr(x_values, y_values)[0]
    c = cm(x_values, y_values)
    xy_corr = {
        "pearson": r,
        "spearman": rs,
        "clustermatch": c,
    }
    
    grid = sns.JointGrid(
        data=data,
        x=x,
        y=y,
    )

    color = "C0"
    color_rgb = mpl.colors.colorConverter.to_rgb(color)
    colors = [sns.utils.set_hls_values(color_rgb, l=l) for l in np.linspace(1, 0, 12)]
    cmap = sns.palettes.blend_palette(colors, as_cmap=True)

    x_bins = min(_freedman_diaconis_bins(grid.x), 50)
    y_bins = min(_freedman_diaconis_bins(grid.y), 50)
    gridsize = int(np.mean([x_bins, y_bins]))

    joint_kws = {
        "bins": bins,
    }

    joint_kws.setdefault("gridsize", gridsize)
    joint_kws.setdefault("cmap", cmap)
    joint_kws.setdefault("rasterized", True)

    grid.plot_joint(
        plt.hexbin,
        **joint_kws,
    )

    # remove marginal axes
    grid.ax_marg_x.set_visible(False)
    grid.ax_marg_y.set_visible(False)
    
    # add text box for the statistics
    ax = grid.ax_joint
    corr_vals = f"$r$ = {r:.2f}\n" f"$r_s$ = {rs:.2f}\n" f"$c$ = {c:.2f}"
    bbox = dict(boxstyle="round", fc="white", ec="black", alpha=0.15)
    ax.text(
        0.25,
        0.80,
        corr_vals,
        fontsize=12,
        bbox=bbox,
        transform=ax.transAxes,
        horizontalalignment="right",
    )

    return grid


# %%
with sns.plotting_context("talk", font_scale=1.0):
    g = jointplot(
        data=df,
        x="pearson",
        y="clustermatch",
        bins="log",
    )

    g.savefig(
        OUTPUT_FIGURE_DIR / "dist-clustermatch_vs_pearson.svg",
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
    )

# %%
with sns.plotting_context("talk", font_scale=1.0):
    g = jointplot(
        data=df,
        x="spearman",
        y="clustermatch",
        bins="log",
    )

    sns.despine(ax=g.ax_joint, left=True)
    g.ax_joint.set_yticks([])
    g.ax_joint.set_ylabel(None)

    g.savefig(
        OUTPUT_FIGURE_DIR / "dist-clustermatch_vs_spearman.svg",
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
    )

# %%
with sns.plotting_context("talk", font_scale=1.0):
    g = jointplot(
        data=df,
        x="spearman",
        y="pearson",
        bins="log",
    )

    g.savefig(
        OUTPUT_FIGURE_DIR / "dist-spearman_vs_pearson.svg",
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
    )

# %%
