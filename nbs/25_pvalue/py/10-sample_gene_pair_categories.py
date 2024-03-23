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
# It sample gene pairs from the categories in Figure 3b.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import pandas as pd

from ccc import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
DATASET_CONFIG = conf.GTEX
GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"
N_MAX_SAMPLES_PER_CATEGORY = 500

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
# # Load gene pairs intersection

# %% tags=[]
df_plot = pd.read_pickle(INPUT_GENE_PAIRS_INTERSECTIONS_FILE)

# %% tags=[]
df_plot.shape

# %% tags=[]
df_plot.head()

# %% tags=[]
df_plot[
    df_plot["Spearman (high)"]
    & df_plot["Pearson (low)"]
    & (~df_plot["Clustermatch (high)"])
    & (~df_plot["Clustermatch (low)"])
]

# %% [markdown] tags=[]
# # Select gene pairs from each category in Figure 3b

# %% tags=[]
gene_pair_cats = {}

# %% tags=[]
gene_pair_cats["all_high"] = df_plot[
    df_plot["Clustermatch (high)"]
    & df_plot["Spearman (high)"]
    & df_plot["Pearson (high)"]
]
display(gene_pair_cats["all_high"].shape)

# %% tags=[]
gene_pair_cats["all_low"] = df_plot[
    df_plot["Clustermatch (low)"] & df_plot["Spearman (low)"] & df_plot["Pearson (low)"]
]
display(gene_pair_cats["all_low"].shape)

# %% tags=[]
gene_pair_cats["ccc_spearman_high_and_pearson_low"] = df_plot[
    df_plot["Clustermatch (high)"]
    & df_plot["Spearman (high)"]
    & df_plot["Pearson (low)"]
]
display(gene_pair_cats["ccc_spearman_high_and_pearson_low"].shape)

# %% tags=[]
gene_pair_cats["ccc_high_and_pearson_low"] = df_plot[
    df_plot["Clustermatch (high)"]
    & (~df_plot["Spearman (high)"])
    & (~df_plot["Spearman (low)"])
    & df_plot["Pearson (low)"]
]
display(gene_pair_cats["ccc_high_and_pearson_low"].shape)

# %% tags=[]
gene_pair_cats["ccc_high_and_spearman_low"] = df_plot[
    df_plot["Clustermatch (high)"]
    & df_plot["Spearman (low)"]
    & (~df_plot["Pearson (low)"])
    & (~df_plot["Pearson (high)"])
]
display(gene_pair_cats["ccc_high_and_spearman_low"].shape)

# %% tags=[]
gene_pair_cats["ccc_high_and_spearman_pearson_low"] = df_plot[
    df_plot["Clustermatch (high)"]
    & df_plot["Spearman (low)"]
    & df_plot["Pearson (low)"]
]
display(gene_pair_cats["ccc_high_and_spearman_pearson_low"].shape)

# %% tags=[]
gene_pair_cats["pearson_high_and_ccc_low"] = df_plot[
    df_plot["Clustermatch (low)"]
    & (~df_plot["Spearman (low)"])
    & (~df_plot["Spearman (high)"])
    & df_plot["Pearson (high)"]
]
display(gene_pair_cats["pearson_high_and_ccc_low"].shape)

# %% tags=[]
gene_pair_cats["pearson_high_and_ccc_spearman_low"] = df_plot[
    df_plot["Clustermatch (low)"]
    & df_plot["Spearman (low)"]
    & df_plot["Pearson (high)"]
]
display(gene_pair_cats["pearson_high_and_ccc_spearman_low"].shape)

# %% tags=[]
assert len(gene_pair_cats) == 8

# %% [markdown] tags=[]
# # Sample gene pairs

# %% [markdown] tags=[]
# Here I take all the categories defined above (keys in dictionaries) and I create three subcategories for each, where I take the top genes prioritized by the three coefficients.

# %% tags=[]
# prepare weights for sampling, where I will put zeros on already sampled gene pairs
gene_pairs_weights = (
    df_plot.drop(columns=df_plot.columns[:-1])
    .rename(columns={df_plot.columns[-1]: "weight"})
    .assign(weight=1.0)
    .squeeze()
    .sort_index()
)

# %% tags=[]
gene_pairs_weights

# %% tags=[]
_tmp = df_plot.sample(n=10, replace=False, weights=gene_pairs_weights)
assert _tmp.shape[0] == 10

display(_tmp)

# %% tags=[]
gene_pair_samples = {}

for k, v in gene_pair_cats.items():
    # sample at most 100 gene pairs
    df = gene_pair_cats[k]

    n = min(N_MAX_SAMPLES_PER_CATEGORY, df.shape[0])

    for coef in ("ccc", "pearson", "spearman", "random"):
        if coef == "random":
            new_k = f"{k}-{coef}"

            # do not sample if all gene pairs were already sampled
            df_weights = gene_pairs_weights.loc[df.index]
            if (df_weights > 0).sum() < n:
                display(f"  WARNING: {new_k}: none selected")
                continue

            sample_n = df.sample(
                n=n,
                replace=False,
                random_state=RANDOM_STATE,
                weights=gene_pairs_weights,
            )

            # do not sample these gene pairs again
            gene_pairs_weights.loc[sample_n.index] = 0.0

            gene_pair_samples[new_k] = sample_n

            display(f"{new_k}: {gene_pair_samples[new_k].shape}")

            continue

        df_coef = df.sort_values(coef, ascending=False)
        sample_n = df_coef.head(n)

        # when taking the top gene pairs by a coefficient, I do not remove repeated ones

        # do not sample these gene pairs again
        gene_pairs_weights.loc[sample_n.index] = 0.0

        new_k = f"{k}-top_{coef}"
        gene_pair_samples[new_k] = sample_n

        display(f"{new_k}: {gene_pair_samples[new_k].shape}")

# %% [markdown] tags=[]
# # Include gene pairs mentioned in the paper

# %% tags=[]
selected_gene_pairs = [
    # ('SDS', 'IFNG')
    ("ENSG00000135094.10", "ENSG00000111537.4"),
    # ('APOC1', 'JUN')
    ("ENSG00000130208.9", "ENSG00000177606.6"),
    # ('CCL18', 'ZDHHC12')
    ("ENSG00000275385.1", "ENSG00000160446.18"),
    # ('KDM6A', 'UTY')
    ("ENSG00000147050.14", "ENSG00000183878.15"),
    # ('CYTIP', 'RASSF2')
    ("ENSG00000115165.9", "ENSG00000101265.15"),
    # ('KLHL21', 'AC068580.6')
    ("ENSG00000162413.16", "ENSG00000235027.1"),
    # ('TNNI2', 'MYOZ1')
    ("ENSG00000130598.15", "ENSG00000177791.11"),
    # ('TPM2', 'PYGM')
    ("ENSG00000198467.13", "ENSG00000068976.13"),
]

# %% tags=[]
gene_pair_samples["selected_in_manuscript"] = df_plot.loc[selected_gene_pairs]
display(gene_pair_samples["selected_in_manuscript"].shape)

# %% tags=[]
gene_pair_samples["selected_in_manuscript"]

# %% [markdown] tags=[]
# # Include a random sample across the entire dataset

# %% [markdown] tags=[]
# This includes all possible gene pairs from the top 5k genes initially selected, not the filtered list derived from the intersections.

# %% [markdown] tags=[]
# ## Load all correlations

# %% tags=[]
INPUT_CORR_FILE_TEMPLATE = (
    DATASET_CONFIG["SIMILARITY_MATRICES_DIR"]
    / DATASET_CONFIG["SIMILARITY_MATRIX_FILENAME_TEMPLATE"]
)
display(INPUT_CORR_FILE_TEMPLATE)

# %% tags=[]
INPUT_CORR_FILE = DATASET_CONFIG["SIMILARITY_MATRICES_DIR"] / str(
    INPUT_CORR_FILE_TEMPLATE
).format(
    tissue=GTEX_TISSUE,
    gene_sel_strategy=GENE_SEL_STRATEGY,
    corr_method="all",
)
display(INPUT_CORR_FILE)

# %% tags=[]
df = pd.read_pickle(INPUT_CORR_FILE)

# %% tags=[]
df.shape

# %% tags=[]
df.head()

# %% [markdown] tags=[]
# ## Select 2n here (double)

# %% tags=[]
sample_n = df.sample(n=int(n * 2), replace=False, random_state=RANDOM_STATE)

new_k = f"entire_dataset-random"
gene_pair_samples[new_k] = sample_n

# %% tags=[]
gene_pair_samples[new_k].shape

# %% tags=[]
gene_pair_samples[new_k]

# %% [markdown] tags=[]
# # Save

# %% tags=[]
output_file = OUTPUT_DIR / "gene_pair-samples.pkl"

# %% tags=[]
pd.to_pickle(gene_pair_samples, output_file)

# %% tags=[]
