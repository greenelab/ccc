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

# %% [markdown]
# Make sure we are not using multiple cores for this comparison.

# %%
# # %env NUMBA_NUM_THREADS=1
# # %env MKL_NUM_THREADS=1
# # %env OPEN_BLAS_NUM_THREADS=1
# # %env NUMEXPR_NUM_THREADS=1
# # %env OMP_NUM_THREADS=1

# %% tags=[]
from timeit import timeit
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.preprocessing import minmax_scale

from clustermatch import conf
from clustermatch.coef import cm
from clustermatch.methods import mic, distcorr

# %% [markdown] tags=[]
# # Settings

# %%
N_REPS = 10
N_SAMPLES_LIST = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]

# %% [markdown] tags=[]
# # Paths

# %%
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None and conf.MANUSCRIPT["BASE_DIR"].exists()
), "Manuscript dir not set"

# %% tags=[]
OUTPUT_FIGURE_DIR = conf.MANUSCRIPT["FIGURES_DIR"] / "intro"
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% [markdown]
# # Setup code

# %%
x = np.random.rand(100)
y = np.random.rand(100)

# %%
cm(x, y)

# %%
mic(x, y)

# %%
distcorr(x, y)


# %%
def run_timeit(corr_func, x, y, **kwargs):
    return timeit(lambda: corr_func(x, y), **kwargs)


# %% [markdown]
# # Run

# %%
n_samples_list = []

cm_times = []
mic_times = []
distcorr_times = []

# %%
for i, n_samples in enumerate(N_SAMPLES_LIST):
    x = np.random.rand(n_samples)
    y = np.random.rand(n_samples)

    n_samples_list.append(n_samples)

    cm_times.append(run_timeit(cm, x, y, number=N_REPS))
    distcorr_times.append(run_timeit(distcorr, x, y, number=N_REPS))

    if n_samples <= 1000:
        mic_times.append(run_timeit(mic, x, y, number=N_REPS))
    else:
        mic_times.append(np.nan)

# %%
plt.title("Distance covariance performance comparison")
plt.xlabel("Number of samples")
plt.ylabel("Time (seconds)")
plt.plot(N_SAMPLES_LIST, cm_times, label="cm")
plt.plot(N_SAMPLES_LIST, mic_times, label="mic")
plt.plot(N_SAMPLES_LIST, distcorr_times, label="distcorr")
plt.legend()
plt.show()

# %%

# %% [markdown]
# Multiple CPU, cdist_parts parallel attempt #1
# * it's a weird, almost no improvement

# %%
plt.title("Distance covariance performance comparison")
plt.xlabel("Number of samples")
plt.ylabel("Time (seconds)")
plt.plot(N_SAMPLES_LIST, cm_times, label="cm")
plt.plot(N_SAMPLES_LIST, mic_times, label="mic")
plt.plot(N_SAMPLES_LIST, distcorr_times, label="distcorr")
plt.legend()
plt.show()

# %%

# %% [markdown]
# Results with multiple CPU enabled, current clustermatch versino with no extra optimization (should be the same as single CPU):

# %%
plt.title("Distance covariance performance comparison")
plt.xlabel("Number of samples")
plt.ylabel("Time (seconds)")
plt.plot(N_SAMPLES_LIST, cm_times, label="cm")
plt.plot(N_SAMPLES_LIST, mic_times, label="mic")
plt.plot(N_SAMPLES_LIST, distcorr_times, label="distcorr")
plt.legend()
plt.show()

# %%

# %% [markdown]
# Results with only one CPU core:

# %%
plt.title("Distance covariance performance comparison")
plt.xlabel("Number of samples")
plt.ylabel("Time (seconds)")
plt.plot(N_SAMPLES_LIST, cm_times, label="cm")
plt.plot(N_SAMPLES_LIST, mic_times, label="mic")
plt.plot(N_SAMPLES_LIST, distcorr_times, label="distcorr")
plt.legend()
plt.show()

# %%
