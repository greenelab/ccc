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
# Reads $k_{\mathrm{max}}$ analyses and plot results.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from ccc import conf
from ccc.coef import ccc

# %% [markdown] tags=[]
# # Settings

# %% tags=[]

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None and conf.MANUSCRIPT["BASE_DIR"].exists()
), "Manuscript dir not set"

# %% tags=[]
INPUT_DIR = conf.RESULTS_DIR / "k_max_test"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
display(INPUT_DIR)

# %% tags=[]
INPUT_FILE = INPUT_DIR / "k_max-results.pkl"
display(INPUT_FILE)

# %% tags=[]
OUTPUT_FIGURE_DIR = conf.MANUSCRIPT["FIGURES_DIR"] / "misc"
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% [markdown] tags=[]
# # Load data

# %% tags=[]
results = pd.read_pickle(INPUT_FILE)

# %% tags=[]
results.shape

# %% tags=[]
results.head()

# %% [markdown] tags=[]
# # Stats

# %% tags=[]
results.describe()

# %% [markdown] tags=[]
# # Change data types

# %% tags=[]
results.dtypes

# %% tags=[]
results["data_size"] = results["data_size"].astype(int)
results["k_max"] = results["k_max"].astype(int)

# %% tags=[]
results.dtypes

# %% [markdown] tags=[]
# # Plot

# %% tags=[]
sns.set(rc={"figure.figsize": (12, 8)})

# %% tags=[]
with sns.axes_style("whitegrid"), sns.plotting_context("poster"):
    # fig, axes = plt.subplots(1, 2, figsize=(24, 8))

    common_params = {
        "data": results,
        "x": "k_max_as_n_fraction",
        "y": "coef",
        "hue": "data_size",
        "style": "data_size",
        "markers": True,
        "dashes": False,
    }

    ax = sns.lineplot(
        # ax=axes[0],
        **common_params,
    )
    ax.set(xlim=(0, 1), ylim=(0, 1.0))
    ax.set(ylabel="CCC", xlabel="$k_{\mathrm{max}}$ as fraction of $n$")
    ax.xaxis.grid(False)
    # sns.move_legend(ax, 0, title="$n$")
    ax.get_legend().set_title("$n$")

    plt.savefig(
        OUTPUT_FIGURE_DIR / "constant_baseline-k_max.svg",
        bbox_inches="tight",
        facecolor="white",
    )

# %% tags=[]
with sns.axes_style("whitegrid"), sns.plotting_context("poster"):
    fig, axes = plt.subplots(1, 2, figsize=(24, 8))

    common_params = {
        "data": results,
        "x": "k_max_as_n_fraction",
        "y": "coef",
        "hue": "data_size",
        "style": "data_size",
        "markers": True,
        "dashes": False,
    }

    ax = sns.lineplot(
        ax=axes[0],
        **common_params,
    )
    ax.set(xlim=(0, 1), ylim=(0, 1.0))
    ax.get_legend().set_title("$n$")

    ax = sns.lineplot(
        ax=axes[1],
        legend=False,
        **common_params,
    )
    ax.set(xlim=(0, 1), ylim=(0, 0.12))

    for ax in axes:
        ax.set(ylabel="CCC", xlabel="$k_{\mathrm{max}}$ as fraction of $n$")
        ax.xaxis.grid(False)

    plt.savefig(
        OUTPUT_FIGURE_DIR / "constant_baseline-k_max-ccc_scaled.svg",
        bbox_inches="tight",
        facecolor="white",
    )

# %% tags=[]
