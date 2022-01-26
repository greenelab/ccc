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
# It samples all the universe of gene pairs within the top genes initially selected (5,000 genes with maximum variance).

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd
import numpy as np

from clustermatch import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
DATASET_CONFIG = conf.GTEX
GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"

# %% tags=[]
# amount of gene pairs to sample
SAMPLE_SIZE = 10000

# number of samples to take
N_SAMPLES = 10

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_FILE = (
    DATASET_CONFIG["GENE_SELECTION_DIR"]
    / f"gtex_v8_data_{GTEX_TISSUE}-{GENE_SEL_STRATEGY}.pkl"
)
display(INPUT_FILE)

assert INPUT_FILE.exists()

# %% tags=[]
OUTPUT_DIR = INPUT_FILE.parent / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% tags=[]
OUTPUT_FILE_TEMPLATE = str(
    OUTPUT_DIR / (f"{INPUT_FILE.stem}-gene_pairs-sample_" + "{sample_id}" + ".pkl")
)

display(OUTPUT_FILE_TEMPLATE)

# %% [markdown] tags=[]
# # Data

# %% [markdown] tags=[]
# ## Genes IDs universe

# %% tags=[]
genes_ids = pd.read_pickle(INPUT_FILE).index.tolist()

# %% tags=[]
len(genes_ids)

# %% tags=[]
genes_ids[:10]

# %% [markdown] tags=[]
# # Create list of gene pairs

# %%
gene_pairs = []

for i in range(len(genes_ids) - 1):
    for j in range(i + 1, len(genes_ids)):
        gene_pairs.append((genes_ids[i], genes_ids[j]))

gene_pairs_df = pd.DataFrame(data=gene_pairs, columns=["gene0", "gene1"])

# %%
assert gene_pairs_df.shape[0] == len(genes_ids) * (len(genes_ids) - 1) / 2
display(gene_pairs_df.shape)

# %%
gene_pairs_df.shape

# %%
gene_pairs_df.head()

# %% [markdown] tags=[]
# # Create samples and save

# %%
for sample_id in range(N_SAMPLES):
    data_sample = gene_pairs_df.sample(n=SAMPLE_SIZE, random_state=sample_id)

    output_filepath = OUTPUT_FILE_TEMPLATE.format(sample_id=sample_id)
    display(output_filepath)

    data_sample.to_pickle(output_filepath)

# %% tags=[]
display(data_sample.shape)
display(data_sample.head())

# %%
