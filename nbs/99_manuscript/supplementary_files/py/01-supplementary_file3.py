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
# Creates **Supplementary File 3**.
#
# *Description*: Correlations and p-values of a subset of gene pairs across all tissues in GTEx v8.

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

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None and conf.MANUSCRIPT["BASE_DIR"].exists()
), "Manuscript dir not set"

# %% tags=[]
INPUT_DIR = conf.GTEX["RESULTS_DIR"] / "other_tissues"
display(INPUT_DIR)

# %% tags=[]
OUTPUT_DIR = conf.MANUSCRIPT["SUPPLEMENTARY_MATERIAL_DIR"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% tags=[]
OUTPUT_FILENAME = "Supplementary_File_03-Gene_pairs_correlations_all_GTEx_tissues"

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
# # List of dataframes to combine

# %% tags=[]
df_list = []

# %% [markdown] tags=[]
# # KDM6A - UTY

# %% tags=[]
gene0_id, gene1_id = "ENSG00000147050.14", "ENSG00000183878.15"
gene0_symbol, gene1_symbol = "KDM6A", "UTY"

assert gene_map[gene0_id] == gene0_symbol
assert gene_map[gene1_id] == gene1_symbol

# %% tags=[]
GENE_PAIR_INPUT_DIR = INPUT_DIR / f"{gene0_symbol.lower()}_vs_{gene1_symbol.lower()}"
display(GENE_PAIR_INPUT_DIR)

# %% [markdown] tags=[]
# ## Correlation values

# %% tags=[]
res_all = pd.read_pickle(GENE_PAIR_INPUT_DIR / "coef_values.pkl").rename(
    columns={
        "cm": "ccc_coef",
        "pearson": "pearson_coef",
        "spearman": "spearman_coef",
    }
)

# %% tags=[]
res_all.shape

# %% tags=[]
res_all.head()

# %% [markdown] tags=[]
# ## P-values

# %% tags=[]
res_pval_all = pd.read_pickle(GENE_PAIR_INPUT_DIR / "coef_pvalues.pkl").rename(
    columns={
        "cm": "ccc_pvalue",
        "pearson": "pearson_pvalue",
        "spearman": "spearman_pvalue",
    }
)

# %% tags=[]
res_pval_all.shape

# %% tags=[]
res_pval_all.head()

# %% [markdown] tags=[]
# ## Combine

# %% tags=[]
df = res_all.join(res_pval_all, how="inner").rename_axis("tissue").reset_index()
assert df.shape[0] == res_all.shape[0]
assert df.shape[0] == res_pval_all.shape[0]

# %% tags=[]
df.insert(0, "gene0_id", gene0_id)
df.insert(1, "gene1_id", gene1_id)
df.insert(2, "gene0_symbol", gene0_symbol)
df.insert(3, "gene1_symbol", gene1_symbol)

# %% tags=[]
df.shape

# %% tags=[]
df.head()

# %% tags=[]
df_list.append(df)

# %% [markdown] tags=[]
# # KDM6A - DDX3Y

# %% tags=[]
gene0_id, gene1_id = "ENSG00000147050.14", "ENSG00000067048.16"
gene0_symbol, gene1_symbol = "KDM6A", "DDX3Y"

assert gene_map[gene0_id] == gene0_symbol
assert gene_map[gene1_id] == gene1_symbol

# %% tags=[]
GENE_PAIR_INPUT_DIR = INPUT_DIR / f"{gene0_symbol.lower()}_vs_{gene1_symbol.lower()}"
display(GENE_PAIR_INPUT_DIR)

# %% [markdown] tags=[]
# ## Correlation values

# %% tags=[]
res_all = pd.read_pickle(GENE_PAIR_INPUT_DIR / "coef_values.pkl").rename(
    columns={
        "cm": "ccc_coef",
        "pearson": "pearson_coef",
        "spearman": "spearman_coef",
    }
)

# %% tags=[]
res_all.shape

# %% tags=[]
res_all.head()

# %% [markdown] tags=[]
# ## P-values

# %% tags=[]
res_pval_all = pd.read_pickle(GENE_PAIR_INPUT_DIR / "coef_pvalues.pkl").rename(
    columns={
        "cm": "ccc_pvalue",
        "pearson": "pearson_pvalue",
        "spearman": "spearman_pvalue",
    }
)

# %% tags=[]
res_pval_all.shape

# %% tags=[]
res_pval_all.head()

# %% [markdown] tags=[]
# ## Combine

# %% tags=[]
df = res_all.join(res_pval_all, how="inner").rename_axis("tissue").reset_index()
assert df.shape[0] == res_all.shape[0]
assert df.shape[0] == res_pval_all.shape[0]

# %% tags=[]
df.insert(0, "gene0_id", gene0_id)
df.insert(1, "gene1_id", gene1_id)
df.insert(2, "gene0_symbol", gene0_symbol)
df.insert(3, "gene1_symbol", gene1_symbol)

# %% tags=[]
df.shape

# %% tags=[]
df.head()

# %% tags=[]
df_list.append(df)

# %% [markdown] tags=[]
# # Combine

# %% tags=[]
df_final = pd.concat(df_list, ignore_index=True, axis=0)

# %% tags=[]
assert df_final.shape[0] == sum(d.shape[0] for d in df_list)
for d in df_list:
    assert df_final.shape[1] == d.shape[1]
display(df_final.shape)

# %% tags=[]
df_final

# %% [markdown] tags=[]
# # Save

# %% tags=[]
data = df_final

# %% tags=[]
display(data.index.dtype)
display(data.index)

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
output_file = OUTPUT_DIR / f"{OUTPUT_FILENAME}.tsv"
display(output_file)

# %% tags=[]
data.to_csv(output_file, sep="\t", index=False, float_format="%.5e")

# %% tags=[]
# testing
data2 = data  # .copy()
# data2.index = list(range(0, data2.shape[0]))

data_again = pd.read_csv(output_file, sep="\t", index_col=None)
# data_again.index = data_again.index.map(lambda x: f"{x:.2f}")

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
