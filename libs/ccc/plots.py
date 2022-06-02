"""
Contains some plotting functions to compare a set of correlation coefficients.
These functions are intended to be used within Jupyter notebooks. The idea is
to make them very specific, not for general use, so many parameters (such as
`bins` for histograms) are fixed.

Some code (indicated in each function) is based on seaborns's code base
(https://github.com/mwaskom/seaborn/), for which the copyright notice and
license are shown below.

Copyright (c) 2012-2021, Michael L. Waskom
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the project nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from pathlib import Path

import numpy as np
from scipy import stats
import pandas as pd
from IPython.display import display
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.distributions import _freedman_diaconis_bins
from upsetplot import UpSet

from ccc.coef import ccc
from ccc.utils import human_format


def plot_histogram(
    data: pd.DataFrame,
    figsize: tuple = (10, 7),
    output_dir: Path = None,
    **kwargs,
):
    """
    It plots, in the same figure, the histograms of all the columns in the
    given dataframe. The function is mainly used to plot the distribution
    of correlation coefficients (columns) across gene pairs (rows). It sets
    some visual parameters to general the final figures.

    Args:
        data: a dataframe with gene pairs in rows and coefficient values
        in columns.
        figsize: figure's size.
        output_dir: if not None, the figure will be saved in this directory.
          The file name is "dist-histograms.svg"
        **kwargs: other parameter passed to seaborn.histplot.

    Returns:
        The figure and axis objects.
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
        **kwargs,
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
    Very similar to plot_histogram, but this plots the cummulative histogram
    instead.

    Args:
        data: same as in plot_histogram.
        figsize: same as in plot_histogram
        output_dir: same as in plot_histogram. The file name is
          "dist-cum_histograms.svg"

    Returns:
        The figure and axis objects.
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
        display(coef_at_percent.sort_values())

        x_lim = ax.get_xlim()
        hline_x_max = coef_at_percent.max()

        ax.hlines(
            y=gene_pairs_percent * 100,
            xmin=x_lim[0],
            xmax=hline_x_max,
            color="gray",
            linestyle="dotted",
        )

        for method_name in coef_at_percent.index:
            ax.vlines(
                x=coef_at_percent[method_name],
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


def jointplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    bins: str = "log",
    add_corr_coefs: bool = True,
    output_dir: Path = None,
):
    """
    It mimics some part of the functionality of seaborn's jointplot function,
    but without marginal plots. This function is based on function jointplot in
    https://github.com/mwaskom/seaborn/blob/v0.11/seaborn/axisgrid.py

    Args:
        data: same as in plot_histogram
        x: name of a correlation method name (should be a column of data).
        y: name of a correlation method name (should be a column of data).
        bins: bins parameter passed to function seaborn.plot_joint
        add_corr_coefs: if True, the correlation coefficient of x and y is added
          in a text box using pearson, spearman and ccc.
        output_dir: if given, the output directory where the figure will be
          saved. The file name is "dist-{x}_vs_{y}.svg".

    Returns:
        A seaborn.JointGrid instance.
    """

    # compute correlations
    x_values = data[x].to_numpy()
    y_values = data[y].to_numpy()

    grid = sns.JointGrid(
        data=data,
        x=x,
        y=y,
    )

    color = "C0"
    color_rgb = mpl.colors.colorConverter.to_rgb(color)
    colors = [sns.utils.set_hls_values(color_rgb, l=i) for i in np.linspace(1, 0, 12)]
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
    if add_corr_coefs:
        # compute correlations
        r = stats.pearsonr(x_values, y_values)[0]
        rs = stats.spearmanr(x_values, y_values)[0]
        c = ccc(x_values, y_values)

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
            output_dir / f"dist-{x.lower()}_vs_{y.lower()}.svg",
            bbox_inches="tight",
            dpi=300,
            facecolor="white",
        )

    return grid


class MyUpSet(UpSet):
    """
    This class tweaks the UpSet class (used to create an UpSet plot) to show
    numbers in a short format. It assumes that both numbers and percentages
    are shown, and only at the top (no left/right position of labels is
    supported).
    """

    def _label_sizes(self, ax, rects, where):
        def make_args(val):
            fmt_num = human_format(val)
            fmt_perc = "%.1f%%" % (100 * val / self.total)
            return f"{fmt_num}\n{fmt_perc}"

        margin = 0.01 * abs(np.diff(ax.get_ylim()))
        for rect in rects:
            height = rect.get_height() + rect.get_y()
            ax.text(
                rect.get_x() + rect.get_width() * 0.5,
                height + margin,
                make_args(height),
                ha="center",
                va="bottom",
            )
