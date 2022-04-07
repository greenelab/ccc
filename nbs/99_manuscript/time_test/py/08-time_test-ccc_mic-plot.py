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
INPUT_FILENAME_TEMPLATE = "time_test-ccc_mic"

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None and conf.MANUSCRIPT["BASE_DIR"].exists()
), "Manuscript dir not set"

# %% tags=[]
OUTPUT_FIGURE_DIR = (
    conf.MANUSCRIPT["FIGURES_DIR"] / "coefs_comp" / "time_test"
)
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% tags=[]
INPUT_DIR = (
    conf.RESULTS_DIR / "time_test"
)
display(INPUT_DIR)

# %% [markdown] tags=[]
# # Load results

# %%
time_results = pd.read_pickle(INPUT_DIR / f"{INPUT_FILENAME_TEMPLATE}.pkl")

# %%
time_results.shape

# %%
time_results.head()

# %% [markdown] tags=[]
# # Processing

# %%
time_results = time_results.replace(
    {
        "method": {
            "p": "Pearson",
            "s": "Spearman",
            "cm": "CCC",
            "mic": "MIC",
        }
    }
)

# %%
time_results.shape

# %%
time_results.head()

# %%
time_results.groupby(["data_size", "method"])["time"].describe()

# %% [markdown]
# # Plot

# %%
hue_order = ["CCC", "MIC"]

# %%
deep_colors = sns.color_palette("deep")
display(deep_colors)

# %%
colors = {
    "CCC": deep_colors[0],
    "MIC": deep_colors[3],
}

# %%
plot_data = time_results[time_results["data_size"] >= 500]

# %%
# plot_data = plot_data.assign(
#     data_size=plot_data["data_size"].apply(lambda x: f"{int(x/1000)}k" if x < 1000000 else f"{int(x/1000000)}m")
# )

# %%
with sns.plotting_context('paper', font_scale=1.5):
    ax = sns.pointplot(
        data=plot_data,
        x='data_size',
        y='time',
        hue='method',
        hue_order=hue_order,
        palette=colors,
        legend=False
    )
    sns.despine()
    plt.legend(loc='best')
    plt.xlabel('Number of measured objects')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig(
        OUTPUT_FIGURE_DIR / f'{INPUT_FILENAME_TEMPLATE}.svg',
        bbox_inches="tight",
        facecolor="white",
    )
    # ax.set_yscale('log')

# %%
with sns.plotting_context('paper', font_scale=1.5):
    ax = sns.pointplot(
        data=plot_data,
        x='data_size',
        y='time',
        hue='method',
        hue_order=hue_order,
        palette=colors,
        legend=False,
    )
    sns.despine()
    plt.legend([],[], frameon=False)
    plt.xlabel('Number of measured objects')
    plt.ylabel('Time (seconds) in log scale')
    plt.tight_layout()
    plt.savefig(
        OUTPUT_FIGURE_DIR / f'{INPUT_FILENAME_TEMPLATE}-log.svg',
        bbox_inches="tight",
        facecolor="white",
    )
    ax.set_yscale('log')

# %%
