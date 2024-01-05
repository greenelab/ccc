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
# Creates **Supplementary File 2**.
#
# *Description*: Percentiles for Pearson, Spearman and CCC computed on Supplementary File 1.

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
OUTPUT_DIR = conf.MANUSCRIPT["SUPPLEMENTARY_MATERIAL_DIR"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% tags=[]
OUTPUT_FILENAME = "Supplementary_File_02-Coefficients_percentiles_GTEx_whole_blood"

# %% [markdown] tags=[]
# # Data

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
# # Compute percentiles

# %% tags=[]
percentiles = (
    gene_pair_intersections[["ccc", "pearson", "spearman"]]
    .quantile(np.arange(0.00, 1.01, 0.01))
    .rename_axis("percentile")
)

# %% tags=[]
# convert index to string
percentiles.index = percentiles.index.map(lambda x: f"{x:.2f}")
display(percentiles.index)

# %% tags=[]
with pd.option_context("display.max_rows", None):
    display(percentiles)

# %% [markdown] tags=[]
# # Save

# %% tags=[]
data = percentiles

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
    # data_again.index = data_again.index.astype(int)

# %% tags=[]
data_again.shape

# %% tags=[]
data_again.head()

# %% tags=[]
# testing
pd.testing.assert_frame_equal(
    data,
    data_again.rename_axis("percentile"),
    check_dtype=False,
)

# %% [markdown] tags=[]
# ## Text

# %% tags=[]
# tsv format
output_file = OUTPUT_DIR / f"{OUTPUT_FILENAME}.tsv"
display(output_file)

# %% tags=[]
data.to_csv(output_file, sep="\t", index=True, float_format="%.5e")

# %% tags=[]
# testing
data2 = data  # .copy()
# data2.index = list(range(0, data2.shape[0]))

data_again = pd.read_csv(output_file, sep="\t", index_col="percentile")
data_again.index = data_again.index.map(lambda x: f"{x:.2f}")

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
