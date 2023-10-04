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
# Make plots to show the computational complexity results comparing all coefficients.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ccc import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
INPUT_FILENAME_TEMPLATE = "time_test"

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
OUTPUT_FIGURE_DIR = COEF_COMP_DIR / "time_test"
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% tags=[]
INPUT_DIR = conf.RESULTS_DIR / "time_test"
display(INPUT_DIR)

# %% [markdown] tags=[]
# # Load results

# %% tags=[]
time_results = pd.read_pickle(INPUT_DIR / f"{INPUT_FILENAME_TEMPLATE}.pkl")

# %% tags=[]
time_results.shape

# %% tags=[]
time_results.head()

# %% tags=[]
time_results["method"].unique()

# %% [markdown] tags=[]
# # Processing

# %% tags=[]
time_results = time_results.replace(
    {
        "method": {
            "p-1": "Pearson (1 core)",
            "s-1": "Spearman (1 core)",
            "cm-1": "CCC (1 core)",
            "mic-1": "MIC (1 core)",
            "mic_e-1": "MICe (1 core)",
            "p-3": "Pearson (3 cores)",
            "s-3": "Spearman (3 cores)",
            "cm-3": "CCC (3 cores)",
            "mic-3": "MIC (3 cores)",
            "mic_e-3": "MICe (3 cores)",
            "p-6": "Pearson (6 cores)",
            "s-6": "Spearman (6 cores)",
            "cm-6": "CCC (6 cores)",
            "mic-6": "MIC (6 cores)",
            "mic_e-6": "MICe (6 cores)",
        }
    }
)

# %% tags=[]
time_results.shape

# %% tags=[]
time_results.head()

# %% [markdown] tags=[]
# # Run numbers

# %% tags=[]
plot_data = time_results

# %% tags=[]
run_numbers = (
    plot_data[
        plot_data["method"].str.contains("1 core", regex=False)
        | plot_data["method"].str.contains("CCC (3 cores)", regex=False)
        | plot_data["method"].str.contains("CCC (6 cores)", regex=False)
    ]
    .groupby(["data_size", "method"])["time"]
    .describe()
)
display(run_numbers)

# %% tags=[]
# this is necessary to make sure we did not mix results when running the time test notebooks
# that could happen if the notebooks are run separately without running them all together
assert run_numbers["count"].unique().shape[0] == 2

# %% [markdown] tags=[]
# # Plot

# %% tags=[]
hue_order = sorted(time_results["method"].unique())

# %% tags=[]
hue_order

# %% tags=[]
deep_colors = sns.color_palette("Paired")
display(deep_colors)


# %% tags=[]
def format_data_size(x):
    if x < 1000:
        return f"{int(x)}"
    elif x < 1000000:
        return f"{int(x/1000)}k"

    return f"{int(x/1000000)}m"


plot_data = plot_data.assign(data_size=plot_data["data_size"].apply(format_data_size))

# %% [markdown] tags=[]
# ## First analysis

# %% [markdown] tags=[]
# Here I take a look if using more than 1 core benefits methods.

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.5):
    g = sns.catplot(
        kind="point",
        data=plot_data,
        x="data_size",
        y="time",
        hue="method",
        hue_order=hue_order,
        palette=deep_colors,
        height=5,
        aspect=1.4,
    )

    plt.xlabel("Number of objects")
    plt.ylabel("Time (seconds)")

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.5):
    g = sns.catplot(
        kind="point",
        data=plot_data,
        x="data_size",
        y="time",
        hue="method",
        hue_order=hue_order,
        palette=deep_colors,
        height=5,
        aspect=1.4,
    )

    plt.xlabel("Number of objects")
    plt.ylabel("Time (seconds) in log scale")

    g.ax.set_yscale("log")

# %% [markdown] tags=[]
# Only CCC is really taking advantage of more than 1 core, so I'll remove the rest below.

# %% [markdown] tags=[]
# # Final analysis

# %% tags=[]
# select runs with 3 cores for the other methods
plot_data = plot_data.replace(
    {
        "method": {
            "Pearson (6 cores)": "Pearson",
            "Spearman (6 cores)": "Spearman",
            "MIC (6 cores)": "MIC",
            "MICe (6 cores)": r"$\mathregular{MIC_e}$",
        }
    }
)

# %% tags=[]
plot_data["method"].unique()

# %% tags=[]
hue_order = [
    "MIC",
    r"$\mathregular{MIC_e}$",
    "CCC (1 core)",
    "CCC (3 cores)",
    "Spearman",
    "Pearson",
]

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.5):
    g = sns.catplot(
        kind="point",
        data=plot_data,
        x="data_size",
        y="time",
        hue="method",
        hue_order=hue_order,
        palette=deep_colors,
        height=5,
        aspect=1.1,
        legend=False,
    )

    plt.legend(loc="best")
    plt.xlabel("Number of objects")
    plt.ylabel("Time (seconds)")

    plt.savefig(
        OUTPUT_FIGURE_DIR / f"{INPUT_FILENAME_TEMPLATE}.svg",
        bbox_inches="tight",
        facecolor="white",
    )

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.5):
    g = sns.catplot(
        kind="point",
        data=plot_data,
        x="data_size",
        y="time",
        hue="method",
        hue_order=hue_order,
        palette=deep_colors,
        height=5,
        aspect=1.1,
        legend=False,
    )

    plt.legend(loc="best", fontsize="small", framealpha=0.5)
    plt.xlabel("Number of objects")
    plt.ylabel("Time (seconds) in log scale")
    g.ax.set_yscale("log")

    plt.savefig(
        OUTPUT_FIGURE_DIR / f"{INPUT_FILENAME_TEMPLATE}-log.svg",
        bbox_inches="tight",
        facecolor="white",
    )

# %% [markdown] tags=[]
# # Create final figure

# %% tags=[]
from svgutils.compose import Figure, SVG

# %% tags=[]
Figure(
    "19.79335cm",
    "17.09335cm",
    # white background
    SVG(COEF_COMP_DIR / "white_background.svg").scale(0.5).move(0, 0),
    # SVG(OUTPUT_FIGURE_DIR / "time_test.svg").scale(0.05),
    SVG(OUTPUT_FIGURE_DIR / "time_test-log.svg").scale(0.05),
).save(OUTPUT_FIGURE_DIR / "time_test-main.svg")

# %% [markdown] tags=[]
# Compile the manuscript with manubot and make sure the image has a white background and displays properly.

# %% tags=[]
