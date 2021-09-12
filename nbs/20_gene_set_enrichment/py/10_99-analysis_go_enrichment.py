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
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown] tags=[]
# # Settings

# %%
QUANTILES = np.linspace(0, 1, 10000)
display(QUANTILES[:10])
display(QUANTILES[-10:])

# %% tags=[]
BASE_FOLDER = Path("..", "base").resolve()
# BASE_FOLDER = Path("base").resolve()

assert BASE_FOLDER.exists()

display(BASE_FOLDER)

# %% tags=[]
OUTPUT_DIR_CM = Path(BASE_FOLDER, "results", "clustermatch", "enrichment").resolve()
display(OUTPUT_DIR_CM)
assert OUTPUT_DIR_CM.exists()

# %% tags=[]
OUTPUT_DIR_PE = Path(BASE_FOLDER, "results", "pearson", "enrichment").resolve()
display(OUTPUT_DIR_PE)
assert OUTPUT_DIR_PE.exists()

# %% [markdown]
# # Load enrichment results

# %%
all_files = list(OUTPUT_DIR_CM.glob("*.pkl")) + list(OUTPUT_DIR_PE.glob("*.pkl"))

# %%
display(len(all_files))
assert len(all_files) == int(2 * 126)

# %%
all_results = []

for f_full in all_files:
#     print(f)
    
    f = f_full.name
    
    fsplit = str(f).split("-")
    k = int(fsplit[0].split("_")[1])
    method = fsplit[1]
    gene_sets = fsplit[2]
    gene_sets_ont = fsplit[3]
    results_type = fsplit[4].split(".")[0]
    
    data = pd.read_pickle(f_full)
    
    data = data.assign(**{
        "k": k,
        "method": method,
        "gene_sets": gene_sets,
        "ont": gene_sets_ont,
        "results_type": results_type,
    })
    
    data["Cluster"] = data["Cluster"].astype("category")
    data["k"] = data["k"].astype("category")
    data["method"] = data["method"].astype("category")
    data["gene_sets"] = data["gene_sets"].astype("category")
    data["ont"] = data["ont"].astype("category")
    data["results_type"] = data["results_type"].astype("category")
    
    all_results.append(data)

# %%
len(all_results)

# %%
all_results_df = pd.concat(all_results, ignore_index=True)

# %%
all_results_df.shape

# %%
all_results_df.head()

# %% [markdown]
# # QQ-plot of $p$-values TESTING

# %%
# def get_quantiles(data):
#     return pd.Series(-np.log10(data["p.adjust"])).quantile(QUANTILES).rename("quantile")

# %%
# plot_df = all_results_df.groupby(["method", "k", "ont", "results_type"]).apply(get_quantiles).stack().rename("p.adjust").reset_index()

# %%
# plot_df.shape

# %%
# plot_df.head()

# %%
# fig, ax = plt.subplots(figsize=(10, 8))

# sns.scatterplot(
#     data=plot_df[plot_df["ont"] == "BP"],
#     x="quantile",
#     y="p.adjust",
#     hue="method",
#     ax=ax,
# )

# # ax.set_xlabel(None)
# # ax.set_ylabel(None)

# # min_val = min((quantiles_df.iloc[:, 0].min(), quantiles_df.iloc[:, 1].min()))
# # max_val = max((quantiles_df.iloc[:, 0].max(), quantiles_df.iloc[:, 1].max()))
# # ax.plot([min_val, max_val], [min_val, max_val], "k", linewidth=0.5)

# # plt.legend()

# %% [markdown]
# # QQ-plot of $p$-values (full set)

# %%
_df_common = all_results_df[all_results_df["results_type"] == "full"]
_clustermatch_values = _df_common[_df_common["method"] == "clustermatch"]["p.adjust"]
_pearson_values = _df_common[_df_common["method"] == "pearson"]["p.adjust"]

# %%
# quantiles_df = pd.DataFrame(
#     {
#         "clustermatch": -np.log10(_clustermatch_values.quantile(QUANTILES)),
#         "pearson": -np.log10(_pearson_values.quantile(QUANTILES)),
#     }
# )

# %%
quantiles_df = pd.DataFrame(
    {
        "clustermatch": pd.Series(-np.log10(_clustermatch_values)).quantile(QUANTILES),
        "pearson": pd.Series(-np.log10(_pearson_values)).quantile(QUANTILES),
    }
)

# %%
quantiles_df.shape

# %%
quantiles_df.head()

# %%
fig, ax = plt.subplots(figsize=(10, 8))

sns.scatterplot(
    data=quantiles_df,
    x="pearson",
    y="clustermatch",
    ax=ax,
)

# ax.set_xlabel(None)
# ax.set_ylabel(None)

min_val = min((quantiles_df.iloc[:, 0].min(), quantiles_df.iloc[:, 1].min()))
max_val = max((quantiles_df.iloc[:, 0].max(), quantiles_df.iloc[:, 1].max()))
ax.plot([min_val, max_val], [min_val, max_val], "k", linewidth=0.5)

ax.set_title("Gene Ontology (full)")

# plt.legend()

# %% [markdown]
# # QQ-plot of $p$-values (simplified set)

# %%
_df_common = all_results_df[all_results_df["results_type"] == "simplified"]
_clustermatch_values = _df_common[_df_common["method"] == "clustermatch"]["p.adjust"]
_pearson_values = _df_common[_df_common["method"] == "pearson"]["p.adjust"]

# %%
# quantiles_df = pd.DataFrame(
#     {
#         "clustermatch": -np.log10(_clustermatch_values.quantile(QUANTILES)),
#         "pearson": -np.log10(_pearson_values.quantile(QUANTILES)),
#     }
# )

# %%
quantiles_df = pd.DataFrame(
    {
        "clustermatch": pd.Series(-np.log10(_clustermatch_values)).quantile(QUANTILES),
        "pearson": pd.Series(-np.log10(_pearson_values)).quantile(QUANTILES),
    }
)

# %%
# quantiles_df[quantiles_df > 5] = np.nan

# %%
quantiles_df.shape

# %%
quantiles_df.head()

# %%
fig, ax = plt.subplots(figsize=(10, 8))

sns.scatterplot(
    data=quantiles_df,
    x="pearson",
    y="clustermatch",
    ax=ax,
)

# ax.set_xlabel(None)
# ax.set_ylabel(None)

min_val = min((quantiles_df.iloc[:, 0].min(), quantiles_df.iloc[:, 1].min()))
max_val = max((quantiles_df.iloc[:, 0].max(), quantiles_df.iloc[:, 1].max()))
ax.plot([min_val, max_val], [min_val, max_val], "k", linewidth=0.5)

ax.set_title("Gene Ontology (simplified)")

# plt.legend()

# %% [markdown]
# # QQ-plot of $p$-values (simplified set + BP)

# %%
_df_common = all_results_df[
    (all_results_df["results_type"] == "simplified")
    & (all_results_df["ont"] == "BP")
]
_clustermatch_values = _df_common[_df_common["method"] == "clustermatch"]["p.adjust"]
_pearson_values = _df_common[_df_common["method"] == "pearson"]["p.adjust"]

# %%
quantiles_df = pd.DataFrame(
    {
        "clustermatch": pd.Series(-np.log10(_clustermatch_values)).quantile(QUANTILES),
        "pearson": pd.Series(-np.log10(_pearson_values)).quantile(QUANTILES),
    }
)

# %%
# quantiles_df[quantiles_df > 5] = np.nan

# %%
quantiles_df.shape

# %%
quantiles_df.head()

# %%
fig, ax = plt.subplots(figsize=(10, 8))

sns.scatterplot(
    data=quantiles_df,
    x="pearson",
    y="clustermatch",
    ax=ax,
)

# ax.set_xlabel(None)
# ax.set_ylabel(None)

min_val = min((quantiles_df.iloc[:, 0].min(), quantiles_df.iloc[:, 1].min()))
max_val = max((quantiles_df.iloc[:, 0].max(), quantiles_df.iloc[:, 1].max()))
ax.plot([min_val, max_val], [min_val, max_val], "k", linewidth=0.5)

ax.set_title("Gene Ontology (simplified + BP)")

# plt.legend()

# %% [markdown]
# # QQ-plot of $p$-values (simplified set + CC)

# %%
_df_common = all_results_df[
    (all_results_df["results_type"] == "simplified")
    & (all_results_df["ont"] == "CC")
]
_clustermatch_values = _df_common[_df_common["method"] == "clustermatch"]["p.adjust"]
_pearson_values = _df_common[_df_common["method"] == "pearson"]["p.adjust"]

# %%
quantiles_df = pd.DataFrame(
    {
        "clustermatch": pd.Series(-np.log10(_clustermatch_values)).quantile(QUANTILES),
        "pearson": pd.Series(-np.log10(_pearson_values)).quantile(QUANTILES),
    }
)

# %%
# quantiles_df[quantiles_df > 5] = np.nan

# %%
quantiles_df.shape

# %%
quantiles_df.head()

# %%
fig, ax = plt.subplots(figsize=(10, 8))

sns.scatterplot(
    data=quantiles_df,
    x="pearson",
    y="clustermatch",
    ax=ax,
)

# ax.set_xlabel(None)
# ax.set_ylabel(None)

min_val = min((quantiles_df.iloc[:, 0].min(), quantiles_df.iloc[:, 1].min()))
max_val = max((quantiles_df.iloc[:, 0].max(), quantiles_df.iloc[:, 1].max()))
ax.plot([min_val, max_val], [min_val, max_val], "k", linewidth=0.5)

ax.set_title("Gene Ontology (simplified + CC)")

# plt.legend()

# %% [markdown]
# # QQ-plot of $p$-values (simplified set + MF)

# %%
_df_common = all_results_df[
    (all_results_df["results_type"] == "simplified")
    & (all_results_df["ont"] == "MF")
]
_clustermatch_values = _df_common[_df_common["method"] == "clustermatch"]["p.adjust"]
_pearson_values = _df_common[_df_common["method"] == "pearson"]["p.adjust"]

# %%
quantiles_df = pd.DataFrame(
    {
        "clustermatch": pd.Series(-np.log10(_clustermatch_values)).quantile(QUANTILES),
        "pearson": pd.Series(-np.log10(_pearson_values)).quantile(QUANTILES),
    }
)

# %%
# quantiles_df[quantiles_df > 5] = np.nan

# %%
quantiles_df.shape

# %%
quantiles_df.head()

# %%
fig, ax = plt.subplots(figsize=(10, 8))

sns.scatterplot(
    data=quantiles_df,
    x="pearson",
    y="clustermatch",
    ax=ax,
)

# ax.set_xlabel(None)
# ax.set_ylabel(None)

min_val = min((quantiles_df.iloc[:, 0].min(), quantiles_df.iloc[:, 1].min()))
max_val = max((quantiles_df.iloc[:, 0].max(), quantiles_df.iloc[:, 1].max()))
ax.plot([min_val, max_val], [min_val, max_val], "k", linewidth=0.5)

ax.set_title("Gene Ontology (simplified + MF)")

# plt.legend()

# %% [markdown]
# # Plot unique GO terms

# %%
plot_df = all_results_df.groupby(["method", "k", "results_type"])['ID'].nunique().rename("count").reset_index()

# %%
plot_df.shape

# %%
plot_df.head()

# %%
# fig, ax = plt.subplots(figsize=(10, 8))

sns.catplot(
    data=plot_df,
    x="k",
    y="count",
    hue="method",
    col="results_type",
#     ax=ax,
)

# ax.set_xlabel(None)
# ax.set_ylabel(None)

# min_val = min((quantiles_df.iloc[:, 0].min(), quantiles_df.iloc[:, 1].min()))
# max_val = max((quantiles_df.iloc[:, 0].max(), quantiles_df.iloc[:, 1].max()))
# ax.plot([min_val, max_val], [min_val, max_val], "k", linewidth=0.5)

# plt.legend()

# %% [markdown]
# # Intersection of terms

# %%
from upsetplot import UpSet

# %%
# UpSet?

# %%
_df_common = all_results_df[
    (all_results_df["results_type"] == "simplified")
]

# %%
plot_df = pd.DataFrame({
    "clustermatch": _df_common["method"] == "clustermatch",
    "clustermatch": _df_common["method"] == "pearson",
})

# %%
plot_df

# %%
plot_df.unstack()

# %%
plot(plot_df)

# %%
df = all_results_df[all_results_df["results_type"] == "full"][["method", "ont"]]

# %%
df.shape

# %%
df.head()

# %%
