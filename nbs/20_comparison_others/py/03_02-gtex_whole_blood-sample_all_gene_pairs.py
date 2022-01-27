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
SAMPLE_SIZE = 33000

# number of samples to take
N_SAMPLES = 1

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_CORR_FILE_TEMPLATE = (
    DATASET_CONFIG["SIMILARITY_MATRICES_DIR"]
    / DATASET_CONFIG["SIMILARITY_MATRIX_FILENAME_TEMPLATE"]
)
display(INPUT_CORR_FILE_TEMPLATE)

# %% tags=[]
INPUT_FILE = DATASET_CONFIG["SIMILARITY_MATRICES_DIR"] / str(
    INPUT_CORR_FILE_TEMPLATE
).format(
    tissue=GTEX_TISSUE,
    gene_sel_strategy=GENE_SEL_STRATEGY,
    corr_method="all",
)
display(INPUT_FILE)

# %% tags=[]
# INPUT_FILE = (
#     DATASET_CONFIG["GENE_SELECTION_DIR"]
#     / f"gtex_v8_data_{GTEX_TISSUE}-{GENE_SEL_STRATEGY}.pkl"
# )
# display(INPUT_FILE)

# assert INPUT_FILE.exists()

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
# ## Gene pairs universe

# %% tags=[]
gene_pairs_df = (
    pd.read_pickle(INPUT_FILE).index.rename(("gene0", "gene1")).to_frame(index=False)
)

# %% tags=[]
gene_pairs_df.shape

# %% tags=[]
gene_pairs_df.head()

# %% [markdown] tags=[]
# # Create samples and save

# %% tags=[]
for sample_id in range(N_SAMPLES):
    data_sample = gene_pairs_df.sample(n=SAMPLE_SIZE, random_state=sample_id)

    output_filepath = OUTPUT_FILE_TEMPLATE.format(sample_id=sample_id)
    display(output_filepath)

    data_sample.to_pickle(output_filepath)

# %% tags=[]
display(data_sample.dtypes)
display(data_sample.shape)
display(data_sample.head())

# %% tags=[]
