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
# It combines all coefficient values in one tissue (see `Settings` below) into a single dataframe for easier processing later.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
import numpy as np
import pandas as pd

from clustermatch import conf

# %% [markdown] tags=[]
# # Settings

# %%
DATASET_CONFIG = conf.GTEX
GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_GENE_EXPR_DATA_FILE = (
    DATASET_CONFIG["GENE_SELECTION_DIR"]
    / f"gtex_v8_data_{GTEX_TISSUE}-{GENE_SEL_STRATEGY}.pkl"
)
display(INPUT_GENE_EXPR_DATA_FILE)

assert INPUT_GENE_EXPR_DATA_FILE.exists()

# %% tags=[]
INPUT_CORR_FILE_TEMPLATE = (
    DATASET_CONFIG["SIMILARITY_MATRICES_DIR"]
    / DATASET_CONFIG["SIMILARITY_MATRIX_FILENAME_TEMPLATE"]
)
display(INPUT_CORR_FILE_TEMPLATE)

# %% tags=[]
OUTPUT_FILE = DATASET_CONFIG["SIMILARITY_MATRICES_DIR"] / str(
    INPUT_CORR_FILE_TEMPLATE
).format(
    tissue=GTEX_TISSUE,
    gene_sel_strategy=GENE_SEL_STRATEGY,
    corr_method="all",
)
display(OUTPUT_FILE)

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## Gene Ensembl ID -> Symbol mapping

# %%
gene_map = pd.read_pickle(
    DATASET_CONFIG["DATA_DIR"] / "gtex_gene_id_symbol_mappings.pkl"
)

# %%
gene_map = gene_map.set_index("gene_ens_id")["gene_symbol"].to_dict()

# %%
assert gene_map["ENSG00000145309.5"] == "CABS1"

# %% [markdown] tags=[]
# ## Gene expression

# %%
data = pd.read_pickle(INPUT_GENE_EXPR_DATA_FILE)

# %%
data.shape

# %%
data.head()

# %% [markdown] tags=[]
# ## Clustermatch

# %%
clustermatch_df = pd.read_pickle(
    str(INPUT_CORR_FILE_TEMPLATE).format(
        tissue=GTEX_TISSUE,
        gene_sel_strategy=GENE_SEL_STRATEGY,
        corr_method="clustermatch",
    )
)

# %%
clustermatch_df.shape

# %%
clustermatch_df.head()

# %%
assert data.index.equals(clustermatch_df.index)

# %% [markdown] tags=[]
# ## Pearson

# %%
pearson_df = pd.read_pickle(
    str(INPUT_CORR_FILE_TEMPLATE).format(
        tissue=GTEX_TISSUE,
        gene_sel_strategy=GENE_SEL_STRATEGY,
        corr_method="pearson",
    )
)

# %%
pearson_df.shape

# %%
pearson_df.head()

# %%
assert data.index.equals(pearson_df.index)

# %% [markdown] tags=[]
# ## Spearman

# %%
spearman_df = pd.read_pickle(
    str(INPUT_CORR_FILE_TEMPLATE).format(
        tissue=GTEX_TISSUE,
        gene_sel_strategy=GENE_SEL_STRATEGY,
        corr_method="spearman",
    )
)

# %%
spearman_df.shape

# %%
spearman_df.head()

# %%
assert data.index.equals(spearman_df.index)


# %% [markdown] tags=[]
# ## Merge

# %%
def get_upper_triag(data, k=1):
    mask = np.triu(np.ones(data.shape), k=k).astype(bool)
    return data.where(mask)


# %%
# # make sure genes match
# clustermatch_df = clustermatch_df.loc[pearson_df.index, pearson_df.columns]

# %%
clustermatch_df = get_upper_triag(clustermatch_df)

# %%
clustermatch_df = clustermatch_df.unstack().rename_axis((None, None)).dropna()

# %%
clustermatch_df.shape

# %%
clustermatch_df.head()

# %%
pearson_df = get_upper_triag(pearson_df)

# %%
# make pearson abs
pearson_df = pearson_df.unstack().rename_axis((None, None)).dropna().abs()

# %%
pearson_df.shape

# %%
pearson_df.head()

# %%
assert clustermatch_df.index.equals(pearson_df.index)

# %%
spearman_df = get_upper_triag(spearman_df)

# %%
# make spearman abs
spearman_df = spearman_df.unstack().rename_axis((None, None)).dropna().abs()

# %%
spearman_df.shape

# %%
spearman_df.head()

# %%
assert clustermatch_df.index.equals(spearman_df.index)

# %%
df = pd.DataFrame(
    {
        "clustermatch": clustermatch_df,
        "pearson": pearson_df,
        "spearman": spearman_df,
    }
).sort_index()

# %%
df.shape

# %%
df.head()

# %% [markdown]
# # Save

# %%
df.to_pickle(OUTPUT_FILE)

# %%
