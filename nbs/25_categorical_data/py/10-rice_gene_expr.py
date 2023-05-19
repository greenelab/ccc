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

# %%
from scipy.spatial.distance import squareform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ccc.coef import ccc
from ccc import conf

# %% [markdown]
# # Settings

# %%
INPUT_DIR = conf.DATA_DIR / "rice"

# %% [markdown]
# # Read data

# %%
data = pd.read_csv(INPUT_DIR / "GSE74793_processed-data.txt.gz", sep="\t")

# %%
data.shape

# %%
data.head()

# %% [markdown]
# # Read metadata

# %% [markdown]
# ## Explore

# %%
metadata = pd.read_csv(INPUT_DIR / "test.tsv", sep="\t")

# %%
metadata.shape

# %%
metadata.head()

# %%
sample_cultivars = (
    metadata["genotype"].str.extract("^(.+)\s+\(").rename(columns={0: "cultivar"})
)

# %%
sample_cultivars

# %%
sample_cultivars.value_counts()

# %%
sample_cultivars.isna().sum()

# %% [markdown]
# ## Process metadata

# %%
metadata = pd.read_csv(
    INPUT_DIR / "test.tsv",
    sep="\t",
    usecols=["run_accession", "experiment_title", "genotype", "treatment", "time"],
)

# %%
sample_cultivars = (
    metadata["genotype"].str.extract("^(.+)\s+\(").rename(columns={0: "cultivar"})
)

# %%
sample_times = (
    metadata["time"]
    .str.extract("^(\d+) min")
    .rename(columns={0: "time"})
    .astype({"time": float})
)

# %%
sample_genotypes = (
    metadata["genotype"].str.extract(", (\S+)\)").rename(columns={0: "genotype"})
)

# %%
sample_ids = (
    metadata["experiment_title"]
    .str.extract("^GSM.+: (\S+); ")
    .rename(columns={0: "id"})
)

# %%
full_metadata = (
    sample_cultivars.join(sample_ids)
    .join(sample_genotypes)
    .join(sample_times)
    .join(metadata[["treatment"]])
)

# %%
full_metadata = full_metadata.set_index("id")

# %%
full_metadata.shape

# %%
full_metadata.head()

# %%
full_metadata.isna().any()

# %% [markdown]
# # Plot

# %%
plot_data = data.T

# %%
plot_data

# %%
plot_data = plot_data.join(full_metadata, how="inner")

# %%
plot_data.shape

# %%
plot_data.head()

# %%
sns.scatterplot(data=plot_data, x="LOC_Os08g35305", y="LOC_Os07g43460", hue="genotype")

# %%
plot_data[plot_data["LOC_Os08g35305"] < 2].shape

# %%
plot_data[plot_data["LOC_Os08g35305"] < 2]

# %%
plot_data[plot_data["LOC_Os08g35305"] < 2]["cultivar"].value_counts()

# %%
sns.scatterplot(data=plot_data, x="LOC_Os05g38530", y="LOC_Os04g01740", hue="treatment")

# %% [markdown]
# # Compute CCC

# %%
data_subset = plot_data["LOC_Os08g35305 LOC_Os07g43460 genotype".split()]

# %%
ccc_corrs = ccc(data_subset)

# %%
ccc_corrs = squareform(ccc_corrs)

# %%
ccc_corrs = pd.DataFrame(
    ccc_corrs, index=data_subset.columns.tolist(), columns=data_subset.columns.tolist()
)

# %%
ccc_corrs

# %%
