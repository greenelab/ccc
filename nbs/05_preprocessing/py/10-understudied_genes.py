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
# This notebook preprocess data about understudied genes from this article: https://doi.org/10.1371/journal.pbio.2006643

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd

from ccc import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
OUTPUT_DIR = conf.UNDERSTUDIED_GENES_ARTICLE["DATA_DIR"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## S1 Data

# %% tags=[]
s1_data_1a = pd.read_excel(
    conf.UNDERSTUDIED_GENES_ARTICLE["S1_DATA_FILE"], sheet_name="1A"
)

# %% tags=[]
s1_data_1a.shape

# %% tags=[]
s1_data_1a.head()

# %% tags=[]
s1_data_1a = s1_data_1a.assign(diff=s1_data_1a["predicted"] - s1_data_1a["target"])

# %% tags=[]
s1_data_1a = s1_data_1a.set_index("gene_ncbi")

# %% tags=[]
s1_data_1a.head()

# %% tags=[]
assert s1_data_1a.index.is_unique

# %% [markdown] tags=[]
# ## S3 Table

# %% tags=[]
s3_table = pd.read_excel(conf.UNDERSTUDIED_GENES_ARTICLE["S3_TABLE_FILE"])

# %% tags=[]
s3_table.shape

# %% tags=[]
s3_table.head()

# %% tags=[]
s3_table = s3_table.set_index("gene_ncbi")

# %% tags=[]
s3_table.head()

# %% tags=[]
assert s3_table.index.is_unique

# %% [markdown] tags=[]
# ## Get gene symbol to Entrez ID map

# %% tags=[]
gene_id_map = (
    s3_table.reset_index()[["symbol_ncbi", "gene_ncbi"]]
    .set_index("symbol_ncbi")
    .squeeze()
    .to_dict()
)

# %% tags=[]
assert gene_id_map["SDS"] == 10993

# %% [markdown] tags=[]
# # Save

# %% tags=[]
s1_data_1a.to_pickle(OUTPUT_DIR / "s1_data_1a.pkl")

# %% tags=[]
s3_table.to_pickle(OUTPUT_DIR / "s3_table.pkl")

# %% [markdown] tags=[]
# # Test with some specific genes

# %% [markdown] tags=[]
# ## SDS

# %% tags=[]
_gene_symbol = "SDS"
_gene_id = gene_id_map[_gene_symbol]
display(_gene_id)

# %% tags=[]
s1_data_1a.loc[_gene_id]

# %% [markdown] tags=[]
# Predicted is higher than observed.

# %% tags=[]
s3_table.loc[_gene_id]

# %% [markdown] tags=[]
# Only 15 papers. A search in PubMed of this genes gives... it's hard to find this gene in PubMed since it overlaps with other concepts.
# But [another paper](https://doi.org/10.7554/eLife.93429.1) also identifies this gene with only 8 publications (see their data in [this GitHub repo](https://github.com/amarallab/fmug_analysis) and specifically [here](https://github.com/amarallab/fmug_analysis/blob/main/data/main_table_with_subject_counts_221116.csv)).

# %% [markdown] tags=[]
# ## ZDHHC12

# %% tags=[]
_gene_symbol = "ZDHHC12"
_gene_id = gene_id_map[_gene_symbol]
display(_gene_id)

# %% tags=[]
s1_data_1a.loc[_gene_id]

# %% [markdown] tags=[]
# Predicted is higher than observed.

# %% tags=[]
s3_table.loc[_gene_id]

# %% [markdown] tags=[]
# Only 11 papers.

# %% [markdown] tags=[]
# ## PRSS36

# %% tags=[]
_gene_symbol = "PRSS36"
_gene_id = gene_id_map[_gene_symbol]
display(_gene_id)

# %% tags=[]
s1_data_1a.loc[_gene_id]

# %% [markdown] tags=[]
# Predicted is higher than observed.

# %% tags=[]
s3_table.loc[_gene_id]

# %% [markdown] tags=[]
# Only 5 papers.

# %% [markdown] tags=[]
# ## CYTIP

# %% tags=[]
_gene_symbol = "CYTIP"
_gene_id = gene_id_map[_gene_symbol]
display(_gene_id)

# %% tags=[]
s1_data_1a.loc[_gene_id]

# %% tags=[]
s3_table.loc[_gene_id]

# %% tags=[]
