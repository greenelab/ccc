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
# # Modules

# %% [markdown] tags=[]
# Make sure we are not using multiple cores for this comparison.

# %% tags=[]
# # %env NUMBA_NUM_THREADS=1
# # %env MKL_NUM_THREADS=1
# # %env OPEN_BLAS_NUM_THREADS=1
# # %env NUMEXPR_NUM_THREADS=1
# # %env OMP_NUM_THREADS=1

# %% tags=[]
from timeit import timeit

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

from clustermatch import conf
from clustermatch.coef import cm
from clustermatch.methods import mic, distcorr

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
N_REPS = 10
N_SAMPLES_LIST = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]

# %%
np.random.seed(0)

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None and conf.MANUSCRIPT["BASE_DIR"].exists()
), "Manuscript dir not set"

# %% tags=[]
OUTPUT_FIGURE_DIR = conf.MANUSCRIPT["FIGURES_DIR"] / "intro"
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% [markdown] tags=[]
# # Setup code

# %% tags=[]
x = np.random.rand(100)
y = np.random.rand(100)

# %% tags=[]
cm(x, y)

# %% tags=[]
mic(x, y)

# %% tags=[]
distcorr(x, y)


# %% tags=[]
def run_timeit(corr_func, x, y):
    results = []
    for i in range(N_REPS):
        results.append(timeit(lambda: corr_func(x, y), number=1))
    return results


# %% [markdown] tags=[]
# # Run

# %% tags=[]
n_samples_list = []

cm_times = []
mic_times = []
distcorr_times = []

# %% tags=[]
for i, n_samples in enumerate(N_SAMPLES_LIST):
    x = np.random.rand(n_samples)
    y = np.random.rand(n_samples)

    cm_times.extend(run_timeit(cm, x, y))
    distcorr_times.extend(run_timeit(distcorr, x, y))

    if n_samples <= 10000:
        mic_times.extend(run_timeit(mic, x, y))
    else:
        mic_times.extend([np.nan] * N_REPS)

    n_samples_list.extend([n_samples] * N_REPS)

# %% tags=[]
res = pd.DataFrame(
    {
        "n_samples": n_samples_list,
        "cm": cm_times,
        "dcor": distcorr_times,
        "mic": mic_times,
    }
)

# %% tags=[]
res.shape

# %% tags=[]
res.head()

# %% tags=[]
res = pd.melt(res, id_vars=["n_samples"], var_name="method", value_name="time")

# %% tags=[]
res.head()

# %% [markdown] tags=[]
# # Plot

# %% [markdown] tags=[]
# ## Point plot

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.3):
    g = sns.catplot(
        data=res,
        x="n_samples",
        y="time",
        hue="method",
        kind="point",
        height=5,
        aspect=1.5,
        legend_out=False,
    )
    g.ax.set_xlabel("Number of samples")
    g.ax.set_ylabel("Average time (seconds)")
    g.legend.set_title("Method")

    plt.savefig(
        OUTPUT_FIGURE_DIR / "time-point_plot.svg",
        # rasterized=True,
        # dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

# %% [markdown] tags=[]
# ## Line plot

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.3):
    res_thin = res[~res["method"].isin(("mic",))]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax = sns.lineplot(data=res_thin, x="n_samples", y="time", hue="method", legend=True)
    sns.despine()
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Average time (seconds)")
    ax.legend_.set_title("Method")

    plt.savefig(
        OUTPUT_FIGURE_DIR / "time-line_plot.svg",
        # rasterized=True,
        # dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

# %% tags=[]
