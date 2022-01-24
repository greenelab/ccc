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
# It reads GTEx v8 metadata on samples and subjects and writes a file with that info.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import re

import pandas as pd

from clustermatch import conf

# %% [markdown] tags=[]
# # Settings

# %% [markdown] tags=[]
# # Paths

# %%
TISSUE_DIR = conf.GTEX["DATA_DIR"] / "data_by_tissue"
assert TISSUE_DIR.exists()

# %%
OUTPUT_DIR = conf.GTEX["DATA_DIR"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Data

# %% [markdown] tags=[]
# ## GTEx samples info

# %%
assert conf.GTEX["SAMPLE_ATTRS_FILE"].exists(), "Sample files does not exist"

# %%
gtex_samples = pd.read_csv(
    conf.GTEX["SAMPLE_ATTRS_FILE"],
    sep="\t",
    index_col="SAMPID",
)

# %%
display(gtex_samples.shape)
assert gtex_samples.index.is_unique

# %%
gtex_samples.head()

# %% [markdown] tags=[]
# ## GTEx subject phenotypes

# %%
assert conf.GTEX["SUBJECTS_ATTRS_FILE"].exists(), "Subject files does not exist"

# %%
gtex_phenotypes = pd.read_csv(
    conf.GTEX["SUBJECTS_ATTRS_FILE"],
    sep="\t",
)

# %%
gtex_phenotypes.shape

# %%
gtex_phenotypes.head()

# %% [markdown] tags=[]
# ## GTEx gene expression sample

# %%
pd.read_pickle(next(TISSUE_DIR.glob("*.pkl"))).head()

# %% [markdown] tags=[]
# ## Gene Ensembl ID -> Symbol mapping

# %%
gene_map = pd.read_pickle(conf.GTEX["DATA_DIR"] / "gtex_gene_id_symbol_mappings.pkl")

# %%
gene_map = gene_map.set_index("gene_ens_id")["gene_symbol"].to_dict()

# %%
assert gene_map["ENSG00000145309.5"] == "CABS1"

# %% [markdown]
# # Get GTEx sample metadata

# %%
gtex_samples_ids = gtex_samples.index.to_list()
display(gtex_samples_ids[:5])

# %%
gtex_samples_ids = pd.Series(gtex_samples_ids).rename("SAMPID")

# %%
gtex_samples_ids

# %%
gtex_subjects_ids = gtex_samples_ids.str.extract(
    r"([\w\d]+\-[\w\d]+)", flags=re.IGNORECASE, expand=True
)[0].rename("SUBJID")

# %%
gtex_subjects_ids

# %%
gtex_metadata = pd.concat([gtex_samples_ids, gtex_subjects_ids], axis=1)

# %%
gtex_metadata

# %%
gtex_phenotypes

# %%
gtex_metadata = pd.merge(gtex_metadata, gtex_phenotypes).set_index("SAMPID")

# %%
gtex_metadata

# %%
gtex_metadata = pd.merge(gtex_metadata, gtex_samples, left_index=True, right_index=True)

# %%
gtex_metadata = gtex_metadata.replace(
    {
        "SEX": {
            1: "Male",
            2: "Female",
        }
    }
)

# %%
gtex_metadata = gtex_metadata.sort_index()

# %%
gtex_metadata.head()

# %% [markdown]
# # Testing

# %%
gtex_metadata.shape

# %%
assert not gtex_metadata["SUBJID"].isna().any()

# %%
assert not gtex_metadata["SMTS"].isna().any()
assert not gtex_metadata["SMTSD"].isna().any()

# %%
assert not gtex_metadata["SEX"].isna().any()
assert gtex_metadata["SEX"].unique().shape[0] == 2
assert set(gtex_metadata["SEX"].unique()) == {"Female", "Male"}

# %% [markdown]
# # Save

# %%
output_filename = OUTPUT_DIR / "gtex_v8-sample_metadata.pkl"
display(output_filename)

# %%
gtex_metadata.to_pickle(output_filename)

# %%
