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

# %% tags=[]
TISSUE_DIR = conf.GTEX["DATA_DIR"] / "data_by_tissue"
assert TISSUE_DIR.exists()

# %% tags=[]
OUTPUT_DIR = conf.GTEX["DATA_DIR"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] tags=[]
# # Data

# %% [markdown] tags=[]
# ## GTEx samples info

# %% tags=[]
assert conf.GTEX["SAMPLE_ATTRS_FILE"].exists(), "Sample files does not exist"

# %% tags=[]
gtex_samples = pd.read_csv(
    conf.GTEX["SAMPLE_ATTRS_FILE"],
    sep="\t",
    index_col="SAMPID",
)

# %% tags=[]
display(gtex_samples.shape)
assert gtex_samples.index.is_unique

# %% tags=[]
gtex_samples.head()

# %% [markdown] tags=[]
# ## GTEx subject phenotypes

# %% tags=[]
assert conf.GTEX["SUBJECTS_ATTRS_FILE"].exists(), "Subject files does not exist"

# %% tags=[]
gtex_phenotypes = pd.read_csv(
    conf.GTEX["SUBJECTS_ATTRS_FILE"],
    sep="\t",
)

# %% tags=[]
gtex_phenotypes.shape

# %% tags=[]
gtex_phenotypes.head()

# %% [markdown] tags=[]
# ## GTEx gene expression sample

# %% tags=[]
pd.read_pickle(next(TISSUE_DIR.glob("*.pkl"))).head()

# %% [markdown] tags=[]
# # Get GTEx sample metadata

# %% tags=[]
gtex_samples_ids = gtex_samples.index.to_list()
display(gtex_samples_ids[:5])

# %% tags=[]
gtex_samples_ids = pd.Series(gtex_samples_ids).rename("SAMPID")

# %% tags=[]
gtex_samples_ids

# %% tags=[]
gtex_subjects_ids = gtex_samples_ids.str.extract(
    r"([\w\d]+\-[\w\d]+)", flags=re.IGNORECASE, expand=True
)[0].rename("SUBJID")

# %% tags=[]
gtex_subjects_ids

# %% tags=[]
gtex_metadata = pd.concat([gtex_samples_ids, gtex_subjects_ids], axis=1)

# %% tags=[]
gtex_metadata

# %% tags=[]
gtex_phenotypes

# %% tags=[]
gtex_metadata = pd.merge(gtex_metadata, gtex_phenotypes).set_index("SAMPID")

# %% tags=[]
gtex_metadata

# %% tags=[]
gtex_metadata = pd.merge(gtex_metadata, gtex_samples, left_index=True, right_index=True)

# %% tags=[]
gtex_metadata = gtex_metadata.replace(
    {
        "SEX": {
            1: "Male",
            2: "Female",
        }
    }
)

# %% tags=[]
gtex_metadata = gtex_metadata.sort_index()

# %% tags=[]
gtex_metadata.head()

# %% [markdown] tags=[]
# # Testing

# %% tags=[]
gtex_metadata.shape

# %% tags=[]
assert not gtex_metadata["SUBJID"].isna().any()

# %% tags=[]
assert not gtex_metadata["SMTS"].isna().any()
assert not gtex_metadata["SMTSD"].isna().any()

# %% tags=[]
assert not gtex_metadata["SEX"].isna().any()
assert gtex_metadata["SEX"].unique().shape[0] == 2
assert set(gtex_metadata["SEX"].unique()) == {"Female", "Male"}

# %% [markdown] tags=[]
# # Save

# %% tags=[]
output_filename = OUTPUT_DIR / "gtex_v8-sample_metadata.pkl"
display(output_filename)

# %% tags=[]
gtex_metadata.to_pickle(output_filename)

# %% tags=[]
