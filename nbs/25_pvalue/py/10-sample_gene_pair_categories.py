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

# %% tags=[]
gene_pair_samples = {}

for k, v in gene_pair_cats.items():
    # sample at most 100 gene pairs
    df = gene_pair_cats[k]
    n = min(100, df.shape[0])
    sample_n = df.sample(n=n, replace=False, random_state=RANDOM_STATE)
    # sample_fraq = gene_pair_cats[k].sample(fraq=replace=False)

    gene_pair_samples[k] = sample_n

    display(f"{k}: {gene_pair_samples[k].shape}")

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
# # Save

# %% tags=[]
output_file = OUTPUT_DIR / "gene_pair-samples.pkl"

# %% tags=[]
pd.to_pickle(gene_pair_samples, output_file)

# %% tags=[]
