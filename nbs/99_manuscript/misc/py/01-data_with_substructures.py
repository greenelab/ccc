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
# This notebooks analyzes CCC in the presence of data with substructures. The analyses are focused on the Reviewer 2's comment:
#
# ```
# Consider a scenario where there are two distinct clusters in the data. If the data has a clear cluster, the CCC will always be 1. This may not be an expected result, especially if we know there are substructures in the data, such as different cell types. Will the CCC method fail in such datasets?
# ```

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd

from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import make_blobs

from ccc import conf
from ccc.coef import ccc
from ccc.methods import mic

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
RANDOM_STATE = 123

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
OUTPUT_FIGURE_DIR = conf.MANUSCRIPT["FIGURES_DIR"] / "misc" / "data_with_substructures"
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% [markdown] tags=[]
# # Diagonal

# %% tags=[]
n_samples = 5000

centers = [(-2.5, -2.5), (2.5, 2.5)]
X, y = make_blobs(
    n_samples=n_samples, centers=centers, shuffle=True, random_state=RANDOM_STATE
)

# %% tags=[]
data = pd.DataFrame(X).rename(columns={0: "x", 1: "y"})
data = data.assign(dataset="Diagonal")

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.8):
    plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=data, x="x", y="y")
    plt.axis("equal")
    sns.despine()

# %% tags=[]
datasets_df = data

# %% [markdown] tags=[]
# # Horizontal

# %% tags=[]
n_samples = 5000

centers = [(-2.5, 0), (2.5, 0)]
X, y = make_blobs(
    n_samples=n_samples, centers=centers, shuffle=True, random_state=RANDOM_STATE
)

# %% tags=[]
data = pd.DataFrame(X).rename(columns={0: "x", 1: "y"})
data = data.assign(dataset="Horizontal")

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.8):
    plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=data, x="x", y="y")
    plt.axis("equal")
    sns.despine()

# %% tags=[]
datasets_df = datasets_df.append(
    data,
    ignore_index=True,
)

# %% [markdown] tags=[]
# # Vertical

# %% tags=[]
n_samples = 5000

centers = [(0, -2.5), (0, 2.5)]
X, y = make_blobs(
    n_samples=n_samples, centers=centers, shuffle=True, random_state=RANDOM_STATE
)

# %% tags=[]
data = pd.DataFrame(X).rename(columns={0: "x", 1: "y"})
data = data.assign(dataset="Vertical")

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.8):
    plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=data, x="x", y="y")
    plt.axis("equal")
    sns.despine()

# %% tags=[]
datasets_df = datasets_df.append(
    data,
    ignore_index=True,
)

# %% [markdown] tags=[]
# # Understand how CCC divides samples

# %% [markdown] tags=[]
# ## Prepare datasets

# %% tags=[]
datasets = {
    idx: df.drop(columns="dataset") for idx, df in datasets_df.groupby("dataset")
}


# %% [markdown] tags=[]
# ## Plot

# %% tags=[]
def get_cm_line_points(x, y, max_parts, parts):
    """
    Given two data vectors (x and y) and the max_parts and parts
    returned from calling cm, this function returns two arrays with
    scalars to draw the lines that separates clusters in x and y.
    """
    # get the ccc partitions that maximize the coefficient
    x_max_part = parts[0][max_parts[0]]
    x_unique_k = {}
    for k in np.unique(x_max_part):
        data = x[x_max_part == k]
        x_unique_k[k] = data.min(), data.max()
    x_unique_k = sorted(x_unique_k.items(), key=lambda x: x[1][0])

    y_max_part = parts[1][max_parts[1]]
    y_unique_k = {}
    for k in np.unique(y_max_part):
        data = y[y_max_part == k]
        y_unique_k[k] = data.min(), data.max()
    y_unique_k = sorted(y_unique_k.items(), key=lambda x: x[1][0])

    x_line_points, y_line_points = [], []

    for idx in range(len(x_unique_k) - 1):
        k, (k_min, k_max) = x_unique_k[idx]
        nk, (nk_min, nk_max) = x_unique_k[idx + 1]

        x_line_points.append((k_max + nk_min) / 2.0)

    for idx in range(len(y_unique_k) - 1):
        k, (k_min, k_max) = y_unique_k[idx]
        nk, (nk_min, nk_max) = y_unique_k[idx + 1]

        y_line_points.append((k_max + nk_min) / 2.0)

    return x_line_points, y_line_points


# %% tags=[]
with sns.plotting_context("paper", font_scale=1.8):
    g = sns.FacetGrid(
        data=datasets_df,
        col="dataset",
        col_order=[
            "Diagonal",
            "Horizontal",
            "Vertical",
        ],
        col_wrap=3,
        height=5,
    )
    g.map(sns.scatterplot, "x", "y", s=50, alpha=1)
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    for ds, ax in g.axes_dict.items():
        df = datasets[ds].to_numpy()
        x, y = df[:, 0], df[:, 1]

        # pearson and spearman
        r = pearsonr(x, y)[0]
        rs = spearmanr(x, y)[0]

        # ccc
        c, max_parts, parts = ccc(x, y, return_parts=True)
        c = ccc(x, y)
        m = mic(x, y)

        x_line_points, y_line_points = get_cm_line_points(x, y, max_parts, parts)
        for yp in y_line_points:
            ax.hlines(y=yp, xmin=x.min(), xmax=x.max(), color="r", alpha=0.90)

        for xp in x_line_points:
            ax.vlines(x=xp, ymin=y.min(), ymax=y.max(), color="r", alpha=0.90)

        # add text box for the statistics
        stats = f"$CCC$ = {c:.2f}\n $MIC$ = {m:.2f}"
        bbox = dict(boxstyle="round", fc="white", ec="black", alpha=0.75)
        ax.text(
            0.95,
            0.90,
            stats,
            fontsize=14,
            bbox=bbox,
            transform=ax.transAxes,
            horizontalalignment="right",
        )

    plt.savefig(
        OUTPUT_FIGURE_DIR / "clusters.png",
        # rasterized=True,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

# %% tags=[]
