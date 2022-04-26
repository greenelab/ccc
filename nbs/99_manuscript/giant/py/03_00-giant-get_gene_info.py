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
# It gets all the gene pairs prioritized by different correlation coefficients and writes a file with gene ID mappings (symbols and Entrez IDs).

# %% [markdown] tags=[]
# # Modules

# %%
# %load_ext rpy2.ipython

# %% tags=[]
import pandas as pd

from clustermatch import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
DATASET_CONFIG = conf.GTEX

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_DIR = DATASET_CONFIG["GENE_PAIR_INTERSECTIONS"]
display(INPUT_DIR)

assert INPUT_DIR.exists()

# %% tags=[]
OUTPUT_DIR = conf.GIANT["RESULTS_DIR"] / "intersection_genes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Get gene entrez ids

# %%
genes = set()

# %%
data = pd.read_pickle(INPUT_DIR / "clustermatch_vs_pearson.pkl")
_tmp0 = set(data.index.get_level_values(0))
_tmp1 = set(data.index.get_level_values(1))
genes.update(_tmp0.union(_tmp1))
display(len(genes))

# %%
data = pd.read_pickle(INPUT_DIR / "clustermatch_vs_pearson_spearman.pkl")
_tmp0 = set(data.index.get_level_values(0))
_tmp1 = set(data.index.get_level_values(1))
genes.update(_tmp0.union(_tmp1))
display(len(genes))

# %%
data = pd.read_pickle(INPUT_DIR / "clustermatch_vs_spearman.pkl")
_tmp0 = set(data.index.get_level_values(0))
_tmp1 = set(data.index.get_level_values(1))
genes.update(_tmp0.union(_tmp1))
display(len(genes))

# %%
data = pd.read_pickle(INPUT_DIR / "pearson_vs_clustermatch.pkl")
_tmp0 = set(data.index.get_level_values(0))
_tmp1 = set(data.index.get_level_values(1))
genes.update(_tmp0.union(_tmp1))
display(len(genes))

# %%
data = pd.read_pickle(INPUT_DIR / "pearson_vs_clustermatch_spearman.pkl")
_tmp0 = set(data.index.get_level_values(0))
_tmp1 = set(data.index.get_level_values(1))
genes.update(_tmp0.union(_tmp1))
display(len(genes))

# %%
genes = list(genes)
assert not pd.Series(genes).isna().any()

# %% magic_args="-i genes -o symbol_to_entrezid" language="R"
# library(org.Hs.eg.db)
# hs <- org.Hs.eg.db
#
# symbol_to_entrezid <- select(hs,
#        keys = unlist(genes),
#        columns = c("ENTREZID", "SYMBOL"),
#        keytype = "SYMBOL")

# %%
symbol_to_entrezid.shape

# %%
assert symbol_to_entrezid.shape[0] == len(genes)

# %%
symbol_to_entrezid.head()

# %%
symbol_to_entrezid.isna().any().any()

# %%
symbol_to_entrezid = symbol_to_entrezid.dropna()

# %%
symbol_to_entrezid.shape

# %%
assert symbol_to_entrezid[symbol_to_entrezid["SYMBOL"] == "IFNG"].shape[0] == 1
assert symbol_to_entrezid[symbol_to_entrezid["SYMBOL"] == "RASSF2"].shape[0] == 1

# %% [markdown]
# # Save

# %%
symbol_to_entrezid.to_pickle(OUTPUT_DIR / "gene_map-symbol_to_entrezid.pkl")

# %%
