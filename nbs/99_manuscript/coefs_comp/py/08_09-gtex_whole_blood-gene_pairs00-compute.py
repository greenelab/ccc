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
# It selects one gene pair (see `Settings` below) and computes the correlation coefficients and p-values across all the tissues in GTEx.
# We do this to check whether one pattern found in whole blood also replicates in other tissues.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd

from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from ccc import conf
from ccc.coef import ccc

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
# this gene pair was originally found with ccc on whole blood
# interesting: https://clincancerres.aacrjournals.org/content/26/21/5567.figures-only
gene0_id, gene1_id = "ENSG00000147050.14", "ENSG00000183878.15"
gene0_symbol, gene1_symbol = "KDM6A", "UTY"

CCC_PVALUE_N_PERMS = 1000000

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
TISSUE_DIR = conf.GTEX["DATA_DIR"] / "data_by_tissue"
assert TISSUE_DIR.exists()

# %% tags=[]
OUTPUT_DIR = (
    conf.GTEX["RESULTS_DIR"]
    / "other_tissues"
    / f"{gene0_symbol.lower()}_vs_{gene1_symbol.lower()}"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Data

# %% [markdown] tags=[]
# ## GTEx metadata

# %% tags=[]
gtex_metadata = pd.read_pickle(conf.GTEX["DATA_DIR"] / "gtex_v8-sample_metadata.pkl")

# %% tags=[]
gtex_metadata.shape

# %% tags=[]
gtex_metadata.head()

# %% [markdown] tags=[]
# ## Gene Ensembl ID -> Symbol mapping

# %% tags=[]
gene_map = pd.read_pickle(conf.GTEX["DATA_DIR"] / "gtex_gene_id_symbol_mappings.pkl")

# %% tags=[]
gene_map = gene_map.set_index("gene_ens_id")["gene_symbol"].to_dict()

# %% tags=[]
assert gene_map["ENSG00000145309.5"] == "CABS1"

# %% tags=[]
assert gene_map[gene0_id] == gene0_symbol
assert gene_map[gene1_id] == gene1_symbol

# %% [markdown] tags=[]
# # Compute correlation on all tissues

# %% tags=[]
res_all = pd.DataFrame(
    {
        f.stem.split("_data_")[1]: {
            "cm": ccc(data[gene0_id], data[gene1_id]),
            "pearson": pearsonr(data[gene0_id], data[gene1_id])[0],
            "spearman": spearmanr(data[gene0_id], data[gene1_id])[0],
        }
        for f in TISSUE_DIR.glob("*.pkl")
        if (data := pd.read_pickle(f).T[[gene0_id, gene1_id]].dropna()) is not None
        and data.shape[0] > 10
    }
).T

# %% tags=[]
res_all.shape

# %% tags=[]
res_all.head()

# %% tags=[]
res_all.sort_values("cm")

# %% tags=[]
res_all.sort_values("pearson")

# %% tags=[]
res_all.sort_values("spearman")

# %% [markdown] tags=[]
# # Compute p-values on all tissues

# %% tags=[]
res_pval_all = pd.DataFrame(
    {
        f.stem.split("_data_")[1]: {
            "cm": ccc(
                data[gene0_id],
                data[gene1_id],
                pvalue_n_perms=CCC_PVALUE_N_PERMS,
                n_jobs=conf.GENERAL["N_JOBS"],
            )[1],
            "pearson": pearsonr(data[gene0_id], data[gene1_id])[1],
            "spearman": spearmanr(data[gene0_id], data[gene1_id])[1],
        }
        for f in TISSUE_DIR.glob("*.pkl")
        if (data := pd.read_pickle(f).T[[gene0_id, gene1_id]].dropna()) is not None
        and data.shape[0] > 10
    }
).T

# %% tags=[]
res_pval_all.shape

# %% tags=[]
res_pval_all.head()

# %% tags=[]
res_pval_all.sort_values("cm")

# %% tags=[]
res_pval_all.sort_values("pearson")

# %% tags=[]
res_pval_all.sort_values("spearman")

# %% [markdown] tags=[]
# # Save

# %% [markdown] tags=[]
# ## Coefficient values

# %% tags=[]
res_all.to_pickle(OUTPUT_DIR / "coef_values.pkl")

# %% [markdown] tags=[]
# ## Coefficient p-values

# %% tags=[]
res_pval_all.to_pickle(OUTPUT_DIR / "coef_pvalues.pkl")

# %% tags=[]
