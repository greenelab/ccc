import sys
import pytest

if sys.platform.startswith("win"):
    pytest.skip("Skipping plot tests on Windows", allow_module_level=True)

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ccc.plots import (
    plot_histogram,
    plot_cumulative_histogram,
    jointplot,
    MyUpSet,
)


def test_plots_plot_histogram_simple():
    df = pd.DataFrame(data=np.random.rand(100, 3), columns=["coef1", "coef2", "coef3"])

    f, ax = plot_histogram(df)
    assert f is not None
    assert ax is not None

    assert hasattr(ax, "lines")
    assert len(ax.lines) == 3


def test_plots_plot_histogram_simple_more_columns():
    df = pd.DataFrame(
        data=np.random.rand(100, 5),
        columns=["coef1", "coef2", "coef3", "coef4", "coef5"],
    )

    f, ax = plot_histogram(df)
    assert f is not None
    assert ax is not None

    assert hasattr(ax, "lines")
    assert len(ax.lines) == 5


def test_plots_plot_histogram_save():
    df = pd.DataFrame(data=np.random.rand(100, 3), columns=["coef1", "coef2", "coef3"])

    with tempfile.TemporaryDirectory() as dirpath:
        dirpath = Path(dirpath)
        f, ax = plot_histogram(df, output_dir=dirpath)
        assert f is not None
        assert ax is not None

        assert (dirpath / "dist-histograms.svg").exists()


def test_plots_plot_cumulative_histogram_simple():
    df = pd.DataFrame(
        data=np.random.rand(100, 3), columns=["pearson", "spearman", "ccc"]
    )

    f, ax = plot_cumulative_histogram(df)
    assert f is not None
    assert ax is not None

    assert hasattr(ax, "lines")
    assert len(ax.lines) == 3

    assert hasattr(ax, "collections")
    assert len(ax.collections) == 0


def test_plots_plot_cumulative_histogram_with_mic():
    df = pd.DataFrame(
        data=np.random.rand(100, 4),
        columns=["pearson", "spearman", "ccc", "mic"],
    )

    f, ax = plot_cumulative_histogram(df)
    assert f is not None
    assert ax is not None

    assert hasattr(ax, "lines")
    assert len(ax.lines) == 4

    assert hasattr(ax, "collections")
    assert len(ax.collections) == 0


def test_plots_plot_cumulative_histogram_with_gene_percent():
    df = pd.DataFrame(
        data=np.random.rand(100, 3), columns=["pearson", "spearman", "ccc"]
    )

    f, ax = plot_cumulative_histogram(df, gene_pairs_percent=0.70)
    assert f is not None
    assert ax is not None

    assert hasattr(ax, "lines")
    assert len(ax.lines) == 3

    assert hasattr(ax, "collections")
    assert len(ax.collections) == 4


def test_plots_plot_cumulative_histogram_with_gene_percent_with_mic():
    df = pd.DataFrame(
        data=np.random.rand(100, 4),
        columns=["pearson", "spearman", "ccc", "mic"],
    )

    f, ax = plot_cumulative_histogram(df, gene_pairs_percent=0.70)
    assert f is not None
    assert ax is not None

    assert hasattr(ax, "lines")
    assert len(ax.lines) == 4

    assert hasattr(ax, "collections")
    assert len(ax.collections) == 5


def test_plots_plot_cumulative_histogram_save():
    df = pd.DataFrame(
        data=np.random.rand(100, 3), columns=["pearson", "spearman", "ccc"]
    )

    with tempfile.TemporaryDirectory() as dirpath:
        dirpath = Path(dirpath)
        f, ax = plot_cumulative_histogram(df, output_dir=dirpath)
        assert f is not None
        assert ax is not None

        assert (dirpath / "dist-cum_histograms.svg").exists()


def test_plots_jointplot_simple():
    df = pd.DataFrame(
        data=np.random.rand(100, 3), columns=["pearson", "spearman", "ccc"]
    )

    grid = jointplot(df, x="pearson", y="spearman")
    assert grid is not None
    assert hasattr(grid, "ax_joint")
    assert hasattr(grid, "ax_marg_x")
    assert hasattr(grid, "ax_marg_y")

    assert hasattr(grid.ax_joint, "texts")
    assert len(grid.ax_joint.texts) == 1


def test_plots_jointplot_simple_add_corr_is_false():
    df = pd.DataFrame(
        data=np.random.rand(100, 3), columns=["pearson", "spearman", "ccc"]
    )

    grid = jointplot(df, x="pearson", y="spearman", add_corr_coefs=False)
    assert grid is not None
    assert hasattr(grid, "ax_joint")
    assert hasattr(grid, "ax_marg_x")
    assert hasattr(grid, "ax_marg_y")

    assert hasattr(grid.ax_joint, "texts")
    assert len(grid.ax_joint.texts) == 0


def test_plots_jointplot_save():
    df = pd.DataFrame(
        data=np.random.rand(100, 3), columns=["pearson", "spearman", "ccc"]
    )

    with tempfile.TemporaryDirectory() as dirpath:
        dirpath = Path(dirpath)

        grid = jointplot(df, x="pearson", y="spearman", output_dir=dirpath)
        assert grid is not None

        assert (dirpath / "dist-pearson_vs_spearman.svg").exists()


def test_MyUpSet():
    np.random.seed(0)

    n = 10000
    df = pd.DataFrame(data=np.random.rand(n, 3), columns=["pearson", "spearman", "ccc"])

    # only two intersection groups with more than 1k pairs, forcing to use
    # the human readable numbers in MyUpSet
    intersection_group0 = np.random.randint(0, 2, size=n).astype(bool)
    intersection_group1 = ~intersection_group0

    new_cols = {
        "Spearman (high)": intersection_group0,
        "Spearman (low)": intersection_group1,
        "Pearson (high)": intersection_group0,
        "Pearson (low)": intersection_group1,
        "CCC (high)": intersection_group0,
        "CCC (low)": intersection_group1,
    }

    df = df.assign(**new_cols)
    df = df.set_index(list(new_cols.keys()), drop=False)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 5))
    g = MyUpSet(
        df,
        show_counts=True,
        sort_categories_by=None,
        sort_by=None,
        show_percentages=True,
        element_size=None,
    ).plot(fig)

    assert g is not None
    assert "intersections" in g

    ax = g["intersections"]
    assert hasattr(ax, "texts")
    assert len(ax.texts) == 2
    ax_text = ax.texts[0]._text
    assert "4.97K\n" in ax_text, ax_text
