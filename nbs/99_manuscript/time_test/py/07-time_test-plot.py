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
# # Modules loading

# %% tags=[]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from clustermatch import conf

# %% [markdown] tags=[]
# # Settings

# %%
INPUT_FILENAME_TEMPLATE = "time_test"

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None and conf.MANUSCRIPT["BASE_DIR"].exists()
), "Manuscript dir not set"

# %% tags=[]
OUTPUT_FIGURE_DIR = conf.MANUSCRIPT["FIGURES_DIR"] / "coefs_comp" / "time_test"
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% tags=[]
INPUT_DIR = conf.RESULTS_DIR / "time_test"
display(INPUT_DIR)

# %% [markdown] tags=[]
# # Load results

# %%
time_results = pd.read_pickle(INPUT_DIR / f"{INPUT_FILENAME_TEMPLATE}.pkl")

# %%
time_results.shape

# %%
time_results.head()

# %%
time_results["method"].unique()

# %% [markdown] tags=[]
# # Processing

# %%
time_results = time_results.replace(
    {
        "method": {
            "p-1": "Pearson (1 core)",
            "s-1": "Spearman (1 core)",
            "cm-1": "CCC (1 core)",
            "mic-1": "MIC (1 core)",
            "p-3": "Pearson (3 cores)",
            "s-3": "Spearman (3 cores)",
            "cm-3": "CCC (3 cores)",
            "mic-3": "MIC (3 cores)",
        }
    }
)

# %%
time_results.shape

# %%
time_results.head()

# %% [markdown]
# # Run numbers

# %%
plot_data = time_results  # [time_results["data_size"] >= 500]

# %%
run_numbers = plot_data.groupby(["data_size", "method"])["time"].describe()
display(run_numbers)

# %% [markdown]
# # Plot

# %%
hue_order = None  # ["CCC", "MIC", "Pearson", "Spearman"]

# %%
deep_colors = sns.color_palette("deep")
display(deep_colors)


# %%
# colors = {
#     "CCC": deep_colors[0],
#     "Pearson": deep_colors[1],
#     "Spearman": deep_colors[2],
# }

# %%
def format_data_size(x):
    if x < 1000:
        return f"{int(x)}"
    elif x < 1000000:
        return f"{int(x/1000)}k"

    return f"{int(x/1000000)}m"


plot_data = plot_data.assign(data_size=plot_data["data_size"].apply(format_data_size))

# %% [markdown]
# ## First analysis

# %% [markdown]
# Here I take a look if using more than 1 core benefits methods.

# %%
with sns.plotting_context("paper", font_scale=1.5):
    ax = sns.pointplot(
        data=plot_data,
        x="data_size",
        y="time",
        hue="method",
        hue_order=hue_order,
        palette=deep_colors,
        legend=False,
    )
    sns.despine()
    plt.legend(loc="best")
    plt.xlabel("Number of measured objects")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    # plt.savefig(
    #     OUTPUT_FIGURE_DIR / f"{INPUT_FILENAME_TEMPLATE}.svg",
    #     bbox_inches="tight",
    #     facecolor="white",
    # )
    # ax.set_yscale('log')

# %%
with sns.plotting_context("paper", font_scale=1.5):
    ax = sns.pointplot(
        data=plot_data,
        x="data_size",
        y="time",
        hue="method",
        hue_order=hue_order,
        palette=deep_colors,
        legend=False,
    )
    sns.despine()
    plt.legend([], [], frameon=False)
    plt.xlabel("Number of measured objects")
    plt.ylabel("Time (seconds) in log scale")
    plt.tight_layout()
    # plt.savefig(
    #     OUTPUT_FIGURE_DIR / f"{INPUT_FILENAME_TEMPLATE}-log.svg",
    #     bbox_inches="tight",
    #     facecolor="white",
    # )
    ax.set_yscale("log")

# %% [markdown]
# Only CCC is really taking advantage of more than 1 core, so I'll remove the rest below.

# %% [markdown]
# # Final analysis

# %%
# select runs with 3 cores for the other methods
plot_data = plot_data.replace(
    {
        "method": {
            "Pearson (3 cores)": "Pearson",
            "Spearman (3 cores)": "Spearman",
            "MIC (3 cores)": "MIC",
        }
    }
)

# %%
plot_data["method"].unique()

# %%
hue_order = ["MIC", "CCC (1 core)", "CCC (3 cores)", "Spearman", "Pearson"]

# %%
with sns.plotting_context("paper", font_scale=1.5):
    ax = sns.pointplot(
        data=plot_data,
        x="data_size",
        y="time",
        hue="method",
        hue_order=hue_order,
        palette=deep_colors,
        legend=False,
    )
    sns.despine()
    plt.legend(loc="best")
    plt.xlabel("Number of measured objects")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_FIGURE_DIR / f"{INPUT_FILENAME_TEMPLATE}.svg",
        bbox_inches="tight",
        facecolor="white",
    )

# %%
with sns.plotting_context("paper", font_scale=1.5):
    ax = sns.pointplot(
        data=plot_data,
        x="data_size",
        y="time",
        hue="method",
        hue_order=hue_order,
        palette=deep_colors,
        legend=False,
    )
    sns.despine()
    plt.legend([], [], frameon=False)
    plt.xlabel("Number of measured objects")
    plt.ylabel("Time (seconds) in log scale")
    plt.tight_layout()
    ax.set_yscale("log")
    plt.savefig(
        OUTPUT_FIGURE_DIR / f"{INPUT_FILENAME_TEMPLATE}-log.svg",
        bbox_inches="tight",
        facecolor="white",
    )

# %% [markdown] tags=[]
# # Create final figure

# %%
from svgutils.compose import Figure, SVG, Panel, Text

# %%
Figure(
    "434.7513cm",
    "135.00382cm",
    SVG(OUTPUT_FIGURE_DIR / "time_test.svg").scale(0.5),
    SVG(OUTPUT_FIGURE_DIR / "time_test-log.svg").scale(0.5).move(220, 0),
).save(OUTPUT_FIGURE_DIR / "time_test-main.svg")

# %% [markdown]
# Now open the file, reside to fit drawing to page, and add a white rectangle to the background.

# %% [markdown]
# I think it's important to open the file with Inkscape and save it, just to make sure the content is right.
# Because sometimes Inkscape crashed when opening it.

# %%
