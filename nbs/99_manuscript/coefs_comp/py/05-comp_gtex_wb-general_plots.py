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

# %% tags=[]
import pandas as pd
import numpy as np

# from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.preprocessing import minmax_scale

from clustermatch import conf

# from clustermatch.coef import cm

# %% [markdown] tags=[]
# # Settings

# %%
DATASET_CONFIG = conf.GTEX
GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None and conf.MANUSCRIPT["BASE_DIR"].exists()
), "Manuscript dir not set"

# %% tags=[]
OUTPUT_FIGURE_DIR = conf.MANUSCRIPT["FIGURES_DIR"] / "coefs_comp"
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_FIGURE_DIR)

# %% tags=[]
INPUT_CORR_FILE_TEMPLATE = (
    DATASET_CONFIG["SIMILARITY_MATRICES_DIR"]
    / DATASET_CONFIG["SIMILARITY_MATRIX_FILENAME_TEMPLATE"]
)
display(INPUT_CORR_FILE_TEMPLATE)

# %% tags=[]
INPUT_FILE = DATASET_CONFIG["SIMILARITY_MATRICES_DIR"] / str(
    INPUT_CORR_FILE_TEMPLATE
).format(
    tissue=GTEX_TISSUE,
    gene_sel_strategy=GENE_SEL_STRATEGY,
    corr_method="all",
)
display(INPUT_FILE)

assert INPUT_FILE.exists()

# %% [markdown] tags=[]
# # Data

# %%
df = pd.read_pickle(INPUT_FILE)

# %%
df.shape

# %%
df.head()

# %% [markdown]
# # QQ-plot

# %%
QUANTILES = np.linspace(0, 1, 10000)
display(QUANTILES[:10])
display(QUANTILES[-10:])

# %%
quantiles_df = pd.DataFrame(
    {
        "clustermatch": df["clustermatch"].quantile(QUANTILES).to_numpy(),
        "pearson": df["pearson"].quantile(QUANTILES).to_numpy(),
        "spearman": df["spearman"].quantile(QUANTILES).to_numpy(),
    }
)

# %%
quantiles_df.shape

# %%
quantiles_df.head()

# %%
quantiles_df.describe()

# %%
fig, ax = plt.subplots(figsize=(10, 8))

sns.scatterplot(
    data=quantiles_df,
    x="pearson",
    y="clustermatch",
    label="vs Pearson (abs)",
    ax=ax,
)

sns.scatterplot(
    data=quantiles_df,
    x="spearman",
    y="clustermatch",
    label="vs Spearman (abs)",
    ax=ax,
)

ax.set_xlabel(None)
# if "_k2" in "clustermatch:
# ax.set_ylabel("clustermatch (linear)")

min_val = min((quantiles_df.iloc[:, 0].min(), quantiles_df.iloc[:, 1].min()))
max_val = max((quantiles_df.iloc[:, 0].max(), quantiles_df.iloc[:, 1].max()))
ax.plot([min_val, max_val], [min_val, max_val], "k", linewidth=0.5)

# ax.set_title(f"{ENRICH_FUNC} ({PERFORMANCE_MEASURE})")

# %% [markdown]
# Pearson/Spearman values seem to be more variable in the range (0, 1) than clustermatch

# %% [markdown]
# # Density plot

# %%
with sns.plotting_context("talk", font_scale=1.1):
    fig, ax = plt.subplots(figsize=(10, 7))

    for method in [x for x in df.columns if not x.endswith("_std")]:
        sns.distplot(x=df[method], hist=True, kde=False, label=method, ax=ax)

    plt.legend()

# %% [markdown]
# The distributions are very different. Clustermatch is skewed to the left, whereas pearson and (specially) spearman seem more uniform.

# %% [markdown]
# ## Joint plots comparing each coefficient

# %%
sns.jointplot(
    data=df,
    x="pearson",
    y="clustermatch",
    kind="hex",
    bins="log",
)

# %%
sns.jointplot(
    data=df,
    x="spearman",
    y="clustermatch",
    kind="hex",
    bins="log",
)

# %%
sns.jointplot(
    data=df,
    x="spearman",
    y="pearson",
    kind="hex",
    bins="log",
)

# %% [markdown]
# ## Correlations between coefficient values

# %%
df.corr()

# %%
df.corr("spearman")

# %%
