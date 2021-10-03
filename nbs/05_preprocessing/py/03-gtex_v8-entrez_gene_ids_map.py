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
# TODO

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd

# from tqdm import tqdm
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

pandas2ri.activate()

# from clustermatch.utils import simplify_string
from clustermatch import conf

# %% tags=[]
clusterProfiler = importr("clusterProfiler")

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
# OUTPUT_DIR = conf.GTEX["DATA_DIR"] / "data_by_tissue"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Data loading

# %% tags=[]
input_filename = conf.GTEX["DATA_DIR"] / "gtex_gene_id_symbol_mappings.pkl"
display(input_filename)

# %% tags=[]
data = pd.read_pickle(input_filename)

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% [markdown] tags=[]
# # Add Ensembl IDs without version

# %% tags=[]
data = data.rename(columns={"gene_ens_id": "gene_ens_id_v"})

# %% tags=[]
data = data.assign(gene_ens_id=data["gene_ens_id_v"].apply(lambda x: x.split(".")[0]))

# %% tags=[]
data = data[["gene_ens_id_v", "gene_ens_id", "gene_symbol"]]

# %% tags=[]
data.head()

# %% tags=[]
assert data.shape[0] == data.drop_duplicates().shape[0]

# %% [markdown] tags=[]
# # Add Entrez Gene IDs

# %% tags=[]
assert data["gene_ens_id_v"].is_unique

# %% tags=[]
data["gene_ens_id"].is_unique

# %% tags=[]
data["gene_symbol"].is_unique

# %% [markdown] tags=[]
# Gene Ensembl IDs (without version) and gene symbols by their own are not unique.

# %% tags=[]
gene_ens_ids = data["gene_ens_id"].unique().tolist()
display(len(gene_ens_ids))
display(gene_ens_ids[:5])

# %% tags=[]
entrez_gene_ids = clusterProfiler.bitr(
    gene_ens_ids,
    fromType="ENSEMBL",
    toType="ENTREZID",
    OrgDb="org.Hs.eg.db",
    drop=True,
)

# %% tags=[]
entrez_gene_ids.shape

# %% tags=[]
entrez_gene_ids.head()

# %% tags=[]
assert entrez_gene_ids.shape[0] == entrez_gene_ids.drop_duplicates().shape[0]

# %% tags=[]
entrez_gene_ids["ENSEMBL"].drop_duplicates().shape

# %% tags=[]
entrez_gene_ids["ENTREZID"].drop_duplicates().shape

# %% tags=[]
entrez_gene_ids = entrez_gene_ids.rename(
    columns={
        "ENSEMBL": "ensembl_id",
        "ENTREZID": "entrez_id",
    }
)

# %% [markdown] tags=[]
# # Merge

# %% tags=[]
gene_mappings = pd.merge(
    data[["gene_ens_id_v", "gene_ens_id"]],
    entrez_gene_ids,
    left_on="gene_ens_id",
    right_on="ensembl_id",
    how="inner",
)[["gene_ens_id_v", "ensembl_id", "entrez_id"]]

# %% tags=[]
gene_mappings.shape

# %% tags=[]
gene_mappings.head()

# %% [markdown] tags=[]
# # Save

# %% tags=[]
output_filename = conf.GTEX["DATA_DIR"] / "gtex_entrez_gene_ids_mappings.pkl"
display(output_filename)

# %% tags=[]
gene_mappings.to_pickle(output_filename)

# %% tags=[]
