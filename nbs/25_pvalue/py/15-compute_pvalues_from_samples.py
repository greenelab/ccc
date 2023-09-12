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
# TODO

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import pandas as pd
from concurrent.futures import as_completed, ProcessPoolExecutor

from ccc.coef import ccc
from ccc import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
DATASET_CONFIG = conf.GTEX
GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"

PVALUE_N_PERMS = 100000

RANDOM_STATE = np.random.RandomState(0)

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_GENE_EXPR_FILE = (
    DATASET_CONFIG["GENE_SELECTION_DIR"]
    / f"gtex_v8_data_{GTEX_TISSUE}-{GENE_SEL_STRATEGY}.pkl"
)
display(INPUT_GENE_EXPR_FILE)

assert INPUT_GENE_EXPR_FILE.exists()

# %% tags=[]
INPUT_GENE_PAIRS_INTERSECTIONS_FILE = (
    DATASET_CONFIG["GENE_PAIR_INTERSECTIONS"]
    / f"gene_pair_intersections-gtex_v8-{GTEX_TISSUE}-{GENE_SEL_STRATEGY}.pkl"
)
display(INPUT_GENE_PAIRS_INTERSECTIONS_FILE)

assert INPUT_GENE_PAIRS_INTERSECTIONS_FILE.exists()

# %% tags=[]
OUTPUT_DIR = DATASET_CONFIG["GENE_PAIR_INTERSECTIONS"] / "pvalues"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=[]
OUTPUT_DIR

# %% [markdown] tags=[]
# # Load gene expression data

# %%
data = pd.read_pickle(INPUT_GENE_EXPR_FILE).sort_index()

# %%
data.shape

# %%
data.head()

# %% [markdown] tags=[]
# # Load gene pairs samples

# %%
output_file = OUTPUT_DIR / "gene_pair-samples.pkl"

# %%
gene_pair_samples = pd.read_pickle(output_file)

# %%
len(gene_pair_samples)

# %%
sorted(gene_pair_samples.keys())

# %%
gene_pair_samples["all_high"].head()

# %%
[i for i in gene_pair_samples["all_high"].head(10).index]

# %% [markdown]
# # Compute pvalues on sampled gene pairs

# %% tags=[]
output_file = OUTPUT_DIR / "gene_pair-samples-pvalues.pkl"


# %% tags=[]
def corr_single(x, y):
    ccc_val, ccc_pval = ccc(x, y, pvalue_n_perms=PVALUE_N_PERMS, n_jobs=1)
    p_val, p_pval = stats.pearsonr(x, y)
    s_val, s_pval = stats.spearmanr(x, y)
    
    return ccc_val, ccc_pval, p_val, p_pval, s_val, s_pval


# %%
results = []

with ProcessPoolExecutor(max_workers=conf.GENERAL["N_JOBS"]) as executor:
    tasks = {
        executor.submit(corr_single, data.loc[gene0], data.loc[gene1]): (gene0, gene1, k)
        for k, v in gene_pair_samples.items()
        for gene0, gene1 in gene_pair_samples[k].index
    }

    for t_idx, t in enumerate(as_completed(tasks)):
        gene0, gene1, k = tasks[t]
        ccc_val, ccc_pval, p_val, p_pval, s_val, s_pval = t.result()

        results.append({
            "gene0": gene0,
            "gene1": gene1,
            "group": k,
            "ccc": ccc_val,
            "ccc_pvalue": ccc_pval,
            "pearson": p_val,
            "pearson_pvalue": p_pval,
            "spearman": s_val,
            "spearman_pvalue": s_pval,
        })
        
        if t_idx % 10:
            _df = pd.DataFrame(results)
            _df["group"] = _df["group"].astype("category")
            _df.to_pickle(output_file)

# %% tags=[]
len(results)

# %% tags=[]
results_df = pd.DataFrame(results)
results_df["group"] = results_df["group"].astype("category")

# %%
results_df.shape

# %%
results_df.head()

# %% [markdown] tags=[]
# # Save

# %% tags=[]
results_df.to_pickle(output_file)

# %% tags=[]
