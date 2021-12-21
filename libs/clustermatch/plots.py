"""
Contains some plotting functions to compare a set of correlation coefficients.
These functions are intended to be used within Jupyter notebooks.

TODO i should add the Seaborn's license here
"""

from pathlib import Path

import numpy as np
from scipy import stats
import pandas as pd
from IPython.display import display
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from seaborn.distributions import _freedman_diaconis_bins

from clustermatch.coef import cm


def plot_histogram(
    data: pd.DataFrame, figsize: tuple = (10, 7), output_dir: Path = None
):
    """
    TODO

    Args:
        data: a dataframe with gene pairs in rows and coefficient values in columns.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax = sns.histplot(
        data=data,
        stat="density",
        bins=100,
        common_bins=True,
        common_norm=False,
        kde=True,
        ax=ax,
    )
    sns.despine(ax=ax)

    ax.set_xticks(np.linspace(0, 1, 10 + 1))

    if output_dir is not None:
        plt.savefig(
            output_dir / "dist-histograms.svg",
            bbox_inches="tight",
            facecolor="white",
        )

    return fig, ax


def plot_cumulative_histogram(
    data: pd.DataFrame,
    gene_pairs_percent: float = None,
    figsize: tuple = (10, 7),
    output_dir: Path = None,
):
    """
    TODO

    data, figsize and output_dir are the same as in plot_histogram
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax = sns.histplot(
        data=data,
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

    if gene_pairs_percent is not None:
        coef_at_percent = data.quantile(gene_pairs_percent)

        # show values so it is saved in the notebook
        display(coef_at_percent)

        x_lim = ax.get_xlim()
        ax.hlines(
            y=gene_pairs_percent * 100,
            xmin=x_lim[0],
            xmax=coef_at_percent["spearman"],
            color="gray",
            linestyle="dotted",
        )
        ax.vlines(
            x=coef_at_percent["clustermatch"],
            ymin=0,
            ymax=gene_pairs_percent * 100,
            color="gray",
            linestyle="dotted",
        )
        ax.vlines(
            x=coef_at_percent["pearson"],
            ymin=0,
            ymax=gene_pairs_percent * 100,
            color="gray",
            linestyle="dotted",
        )
        ax.vlines(
            x=coef_at_percent["spearman"],
            ymin=0,
            ymax=gene_pairs_percent * 100,
            color="gray",
            linestyle="dotted",
        )

        ax.set_xlim(x_lim)

    if output_dir is not None:
        plt.savefig(
            output_dir / "dist-cum_histograms.svg",
            bbox_inches="tight",
            facecolor="white",
        )

    return fig, ax


def jointplot(data: pd.DataFrame, x: str, y: str, bins="log", output_dir: Path = None):
    """
    TODO
    Function based on Seaborn's jointplot, but without marginal plots.

    Args:
        data: same as in plot_histogram
        x, y: name of column in data (it is the name of a correlation coefficient)
    """

    # compute correlations
    x_values = data[x].to_numpy()
    y_values = data[y].to_numpy()
    r = stats.pearsonr(x_values, y_values)[0]
    rs = stats.spearmanr(x_values, y_values)[0]
    c = cm(x_values, y_values)

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

    if output_dir is not None:
        plt.savefig(
            output_dir / f"dist-{x}_vs_{y}.svg",
            bbox_inches="tight",
            dpi=300,
            facecolor="white",
        )

    return grid
