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
# Creates **Supplementary File 1**.
#
# *Description*: Classification and correlations of gene pairs used in Figure 3a (top 5,000 most variable genes in GTEx v8 whole blood). P-values are only included for a subset of gene pairs, as explained in the Methods section of the manuscript.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from ccc import conf

# %% tags=[]
readRDS = ro.r["readRDS"]

# %% tags=[]
saveRDS = ro.r["saveRDS"]

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
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
INPUT_GENE_PAIRS_INTERSECTIONS_FILE = (
    DATASET_CONFIG["GENE_PAIR_INTERSECTIONS"]
    / f"gene_pair_intersections-gtex_v8-{GTEX_TISSUE}-{GENE_SEL_STRATEGY}.pkl"
)
display(INPUT_GENE_PAIRS_INTERSECTIONS_FILE)

assert INPUT_GENE_PAIRS_INTERSECTIONS_FILE.exists()

# %% tags=[]
INPUT_PVALUES_FILE = (
    DATASET_CONFIG["GENE_PAIR_INTERSECTIONS"]
    / "pvalues"
    / "gene_pair-samples-pvalues-fdr.pkl"
)
display(INPUT_PVALUES_FILE)
assert INPUT_PVALUES_FILE.exists()

# %% tags=[]
OUTPUT_DIR = conf.MANUSCRIPT["SUPPLEMENTARY_MATERIAL_DIR"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% tags=[]
OUTPUT_FILENAME = "Supplementary_File_01-Gene_pair_intersections"

# %% [markdown] tags=[]
# # Data

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
# ## Gene pairs intersection

# %% tags=[]
gene_pair_intersections = (
    pd.read_pickle(INPUT_GENE_PAIRS_INTERSECTIONS_FILE)
    .rename_axis(("gene0_id", "gene1_id"))
    .sort_index()
)

# %% tags=[]
gene_pair_intersections.shape

# %% tags=[]
gene_pair_intersections.head()

# %% [markdown] tags=[]
# ## p-values

# %% tags=[]
df_pvalues = (
    pd.read_pickle(INPUT_PVALUES_FILE)
    .rename_axis(("gene0_id", "gene1_id"))
    .sort_index()
)

# %% tags=[]
df_pvalues.shape

# %% tags=[]
df_pvalues.head()

# %% tags=[]
# remove duplicated gene pairs
df_pvalues = df_pvalues[~df_pvalues.index.duplicated(keep="first")]

# %% [markdown] tags=[]
# **Note**: Here the "group" column specifies the categories in Figure 3a, followed by `top_[coef]`, where for the same category I sorted gene pairs by `coef`. This allows me, for instance, to take the gene pairs where Pearson is high and CCC is low, and sort by any of those coefficient values.

# %% [markdown] tags=[]
# # Combine data

# %% tags=[]
df_full = gene_pair_intersections.rename(
    columns={
        "ccc": "ccc_coef",
        "pearson": "pearson_coef",
        "spearman": "spearman_coef",
    }
).join(df_pvalues["ccc_fdr pearson_fdr spearman_fdr".split()], how="left")

# %% tags=[]
assert gene_pair_intersections.shape[0] == df_full.shape[0]
display(df_full.shape)

# %% tags=[]
df_full.head()

# %% [markdown] tags=[]
# ## Add gene symbols

# %% tags=[]
df_full = df_full.assign(
    gene0_symbol=df_full.apply(lambda x: gene_map[x.name[0]], axis=1),
    gene1_symbol=df_full.apply(lambda x: gene_map[x.name[1]], axis=1),
)

# %% tags=[]
df_full.shape

# %% tags=[]
# reorder columns
col_name = "gene1_symbol"
col = df_full.pop(col_name)
df_full.insert(0, col_name, col)

col_name = "gene0_symbol"
col = df_full.pop(col_name)
df_full.insert(0, col_name, col)

# %% tags=[]
df_full.head()

# %% [markdown] tags=[]
# ## Optimize DataFrame dtypes

# %% tags=[]
df_full_orig = df_full

# %% tags=[]
display(df_full.memory_usage())
display(f"{df_full.memory_usage().sum():,}")

# %% [markdown] tags=[]
# ### Remove MultiIndex

# %% [markdown] tags=[]
# A MultiIndex is not necessary for a supplementary file.

# %% tags=[]
display(df_full.index.dtype)
display(df_full.index)

# %% tags=[]
df_full = df_full.reset_index()

# %% tags=[]
df_full.head()

# %% tags=[]
display(df_full.index.dtype)
display(df_full.index)

# %% tags=[]
display(df_full.memory_usage())
display(f"{df_full.memory_usage().sum():,}")

# %% [markdown] tags=[]
# ### Downcast dtypes

# %% tags=[]
df_full.dtypes

# %% tags=[]
# categorical values
for _col in ("gene0_id", "gene1_id", "gene0_symbol", "gene1_symbol"):
    df_full[_col] = df_full[_col].astype("category")

# %% tags=[]
df_full.dtypes

# %% tags=[]
display(df_full.memory_usage())
display(f"{df_full.memory_usage().sum():,}")

# %% tags=[]
# float
for _col in ("ccc_coef", "pearson_coef", "spearman_coef"):
    df_full[_col] = pd.to_numeric(df_full[_col], downcast="float")

# %% tags=[]
df_full.dtypes

# %% tags=[]
display(df_full.memory_usage())
display(f"{df_full.memory_usage().sum():,}")

# %% [markdown] tags=[]
# ### Check results

# %% tags=[]
df_full.shape

# %% tags=[]
df_full.head()

# %% tags=[]
# testing
pd.testing.assert_frame_equal(
    df_full_orig.reset_index(),
    df_full,
    check_categorical=False,
    check_dtype=False,
)

# %% tags=[]
del df_full_orig

# %% [markdown] tags=[]
# # Save

# %% tags=[]
data = df_full

# %% tags=[]
# reset index to avoid problems with MultiIndex in Pandas
if isinstance(data.index, pd.MultiIndex):
    display("MultiIndex")
    data = data.reset_index()

# %% [markdown] tags=[]
# ## Pickle

# %% tags=[]
data.to_pickle(OUTPUT_DIR / f"{OUTPUT_FILENAME}.pkl.gz")

# %% [markdown] tags=[]
# ## RDS

# %% tags=[]
output_file = OUTPUT_DIR / f"{OUTPUT_FILENAME}.rds"
display(output_file)

# %% tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    data_r = ro.conversion.py2rpy(data)

# %% tags=[]
data_r

# %% tags=[]
saveRDS(data_r, str(output_file))

# %% tags=[]
# testing: load the rds file again
data_r = readRDS(str(output_file))

# %% tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    data_again = ro.conversion.rpy2py(data_r)
    data_again.index = data_again.index.astype(int)

# %% tags=[]
data_again.shape

# %% tags=[]
data_again.head()

# %% tags=[]
# testing
pd.testing.assert_frame_equal(
    data,
    data_again,
    check_dtype=False,
)

# %% [markdown] tags=[]
# ## Text

# %% tags=[]
# tsv format
output_file = OUTPUT_DIR / f"{OUTPUT_FILENAME}.tsv.gz"
display(output_file)

# %% tags=[]
data.to_csv(output_file, sep="\t", index=False, float_format="%.5e")

# %% tags=[]
# testing
data2 = data.copy()
data2.index = list(range(0, data2.shape[0]))

data_again = pd.read_csv(output_file, sep="\t")
data_again.index = list(data_again.index)

# %% tags=[]
data_again.shape

# %% tags=[]
data_again.head()

# %% tags=[]
# testing
pd.testing.assert_frame_equal(
    data2,
    data_again,
    check_categorical=False,
    check_dtype=False,
)

# %% tags=[]
