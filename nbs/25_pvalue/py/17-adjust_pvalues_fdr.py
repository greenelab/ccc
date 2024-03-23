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
# It reads the pvalues generated previously and adjust them using FDR.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ccc import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
DATASET_CONFIG = conf.GTEX

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
OUTPUT_DIR = DATASET_CONFIG["GENE_PAIR_INTERSECTIONS"] / "pvalues"
assert OUTPUT_DIR.exists()

# %% tags=[]
OUTPUT_DIR

# %% tags=[]
INPUT_PVALUES_FILE = OUTPUT_DIR / "gene_pair-samples-pvalues.pkl"
display(INPUT_PVALUES_FILE)
assert INPUT_PVALUES_FILE.exists()

# %% [markdown] tags=[]
# # Load pvalues

# %% tags=[]
pvalues = pd.read_pickle(INPUT_PVALUES_FILE).sort_index()

# %% tags=[]
pvalues.shape

# %% tags=[]
pvalues.head()

# %% [markdown] tags=[]
# # Get set of unique gene pairs

# %% tags=[]
pvalues = pvalues.set_index(["gene0", "gene1"])

# %% tags=[]
assert not pvalues.index.is_unique

# %% tags=[]
pvalues.loc[pvalues.index.duplicated(keep=False)].sort_index()

# %% tags=[]
pvalues_nodup = pvalues.loc[~pvalues.index.duplicated(keep="first"), :]

# %% tags=[]
pvalues_nodup.shape

# %% [markdown] tags=[]
# # Adjust p-values for all methods

# %% tags=[]
adj_pvals = multipletests(pvalues_nodup["pearson_pvalue"], alpha=0.05, method="fdr_bh")

# %% tags=[]
adj_pvals[1].shape

# %% tags=[]
adj_pvals

# %% tags=[]
for coef in ("ccc", "pearson", "spearman"):
    pval_col_name = f"{coef}_pvalue"
    fdr_col_name = f"{coef}_fdr"
    print(f"{pval_col_name} - {fdr_col_name}")

    adj_pvals = multipletests(pvalues_nodup[pval_col_name], alpha=0.05, method="fdr_bh")
    pvalues_nodup = pvalues_nodup.assign(**{fdr_col_name: adj_pvals[1]})

# %% tags=[]
pvalues_nodup.shape

# %% tags=[]
# reorder columns
_tmp = (
    pvalues_nodup.rename(columns={"group": "agroup"})
    .sort_index(axis="columns")
    .rename(columns={"agroup": "group"})
)
display(_tmp.head())

# %% tags=[]
pvalues_nodup = _tmp

# %% tags=[]
pvalues_nodup.shape

# %% tags=[]
pvalues_nodup.head()

# %% [markdown] tags=[]
# # Reassign adjusted pvalues to original file

# %% tags=[]
pvalues.shape

# %% tags=[]
pvalues = pvalues.assign(
    **{
        (col := f"{coef}_fdr"): pvalues_nodup[col]
        for coef in ("ccc", "pearson", "spearman")
    }
)
pvalues = pvalues[pvalues_nodup.columns]

# %% tags=[]
pvalues.shape

# %% tags=[]
pvalues.head()

# %% tags=[]
# Make sure duplicated gene pairs have the same pvalues/values
pvalues.loc[pvalues.index.duplicated(keep=False)].sort_index()


# %% tags=[]
def _assert_same_values(x):
    for coef in ("ccc", "pearson", "spearman"):
        assert x[f"{coef}"].unique().shape[0] == 1
        assert x[f"{coef}_fdr"].unique().shape[0] == 1

        # for CCC, the pvalue column is computed via permutations, so we don't expect to be all the same
        if coef == "ccc":
            assert x[f"{coef}_pvalue"].unique().shape[0] >= 1, x
        else:
            assert x[f"{coef}_pvalue"].unique().shape[0] == 1, x


# %% tags=[]
pvalues.loc[pvalues.index.duplicated(keep=False)].groupby(["gene0", "gene1"]).apply(
    _assert_same_values
)
print("values seem correct")

# %% [markdown] tags=[]
# # Save

# %% tags=[]
INPUT_PVALUES_FILE.parent

# %% tags=[]
INPUT_PVALUES_FILE.stem

# %% tags=[]
INPUT_PVALUES_FILE.suffix

# %% tags=[]
output_file = (
    INPUT_PVALUES_FILE.parent
    / f"{INPUT_PVALUES_FILE.stem}-fdr{INPUT_PVALUES_FILE.suffix}"
)
display(output_file)

# %% tags=[]
pvalues.to_pickle(output_file)

# %% tags=[]
