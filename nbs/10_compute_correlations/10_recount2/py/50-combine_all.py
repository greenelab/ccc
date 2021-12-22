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

# %% tags=["parameters"]
DATASET_CONFIG = conf.RECOUNT2
# GTEX_TISSUE = None
GENE_SEL_STRATEGY = "var_pc_log2"

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_GENE_EXPR_DATA_FILE = conf.RECOUNT2["DATA_FILE"]
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
    corr_method="all",
)
display(OUTPUT_FILE)

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## Gene Ensembl ID -> Symbol mapping

# %% tags=[]
gene_map = pd.read_pickle(
    DATASET_CONFIG["DATA_DIR"] / "gtex_gene_id_symbol_mappings.pkl"
)

# %% tags=[]
gene_map = gene_map.set_index("gene_ens_id")["gene_symbol"].to_dict()

# %% tags=[]
assert gene_map["ENSG00000145309.5"] == "CABS1"

# %% [markdown] tags=[]
# ## Gene expression

# %% tags=[]
data = pd.read_pickle(INPUT_GENE_EXPR_DATA_FILE)

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% [markdown] tags=[]
# ## Clustermatch

# %% tags=[]
clustermatch_df = pd.read_pickle(
    str(INPUT_CORR_FILE_TEMPLATE).format(
        corr_method="clustermatch_k2to5",
    )
)

# %% tags=[]
clustermatch_df.shape

# %% tags=[]
clustermatch_df.head()

# %% tags=[]
assert data.index.equals(clustermatch_df.index)

# %% [markdown] tags=[]
# ## Pearson

# %% tags=[]
pearson_df = pd.read_pickle(
    str(INPUT_CORR_FILE_TEMPLATE).format(
        corr_method="pearson",
    )
)

# %% tags=[]
pearson_df.shape

# %% tags=[]
pearson_df.head()

# %% tags=[]
assert data.index.equals(pearson_df.index)

# %% [markdown] tags=[]
# ## Spearman

# %% tags=[]
spearman_df = pd.read_pickle(
    str(INPUT_CORR_FILE_TEMPLATE).format(
        corr_method="spearman",
    )
)

# %% tags=[]
spearman_df.shape

# %% tags=[]
spearman_df.head()

# %% tags=[]
assert data.index.equals(spearman_df.index)


# %% [markdown] tags=[]
# ## Merge

# %% tags=[]
def get_upper_triag(data, k=1):
    mask = np.triu(np.ones(data.shape), k=k).astype(bool)
    return data.where(mask)


# %% tags=[]
# # make sure genes match
# clustermatch_df = clustermatch_df.loc[pearson_df.index, pearson_df.columns]

# %% tags=[]
clustermatch_df = get_upper_triag(clustermatch_df)

# %% tags=[]
clustermatch_df = clustermatch_df.unstack().rename_axis((None, None)).dropna()

# %% tags=[]
clustermatch_df.shape

# %% tags=[]
clustermatch_df.head()

# %% tags=[]
pearson_df = get_upper_triag(pearson_df)

# %% tags=[]
# make pearson abs
pearson_df = pearson_df.unstack().rename_axis((None, None)).dropna().abs()

# %% tags=[]
pearson_df.shape

# %% tags=[]
pearson_df.head()

# %% tags=[]
assert clustermatch_df.index.equals(pearson_df.index)

# %% tags=[]
spearman_df = get_upper_triag(spearman_df)

# %% tags=[]
# make spearman abs
spearman_df = spearman_df.unstack().rename_axis((None, None)).dropna().abs()

# %% tags=[]
spearman_df.shape

# %% tags=[]
spearman_df.head()

# %% tags=[]
assert clustermatch_df.index.equals(spearman_df.index)

# %% tags=[]
df = pd.DataFrame(
    {
        "clustermatch": clustermatch_df,
        "pearson": pearson_df,
        "spearman": spearman_df,
    }
).sort_index()

# %% tags=[]
assert not df.isna().any().any()

# %% tags=[]
df.shape

# %% tags=[]
df.head()

# %% [markdown] tags=[]
# # Save

# %% tags=[]
df.to_pickle(OUTPUT_FILE)

# %% tags=[]
