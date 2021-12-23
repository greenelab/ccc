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
# It uses the same strategy for GTEx (`00-gtex_v8-split_by_tissue.ipynb`) to select the top variable genes in recount2.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import numpy as np
import pandas as pd
from tqdm import tqdm

from clustermatch import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
N_TOP_GENES_MAX_VARIANCE = 5000

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_FILE_DIR = conf.RECOUNT2FULL["DATA_DIR"] / "recount2_rpkm.pkl"
display(INPUT_FILE_DIR)

# %% tags=[]
OUTPUT_DIR = conf.RECOUNT2FULL["GENE_SELECTION_DIR"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Select top genes

# %% [markdown] tags=[]
# Based on the previous findings, I select genes with both strategies `var_raw` and `var_pc_log2`.
#
# Then I save, for both, the raw data (note that I only use the strategies to select genes, not to log-transform the data).

# %% tags=[]
input_files = sorted(
    [
        INPUT_FILE_DIR,
    ]
)

display(input_files[:5])

# %% [markdown] tags=[]
# ## Run

# %% tags=[]
pbar = tqdm(input_files, ncols=100)

for data_file in pbar:
    pbar.set_description(data_file.stem)

    data = pd.read_pickle(data_file)

    # var_raw
    top_genes_var = (
        data.var(axis=1).sort_values(ascending=False).head(N_TOP_GENES_MAX_VARIANCE)
    )
    selected_data = data.loc[top_genes_var.index]

    output_filename = f"{data_file.stem}-var_raw.pkl"
    selected_data.to_pickle(path=OUTPUT_DIR / output_filename)

    # var_pc_log2
    log2_tissue_data = np.log2(data + 1)

    top_genes_var = (
        log2_tissue_data.var(axis=1)
        .sort_values(ascending=False)
        .head(N_TOP_GENES_MAX_VARIANCE)
    )
    # save the same raw data, but with genes selected by var_pc_log2
    selected_data = data.loc[top_genes_var.index]

    output_filename = f"{data_file.stem}-var_pc_log2.pkl"
    selected_data.to_pickle(path=OUTPUT_DIR / output_filename)

# %% tags=[]
