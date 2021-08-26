# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Description

# %% [markdown]
# It splits gene expression data from GTEx v8 by tissue and saves a gene id/symbol mapping file.

# %% [markdown]
# # Modules

# %%
import pickle
from pathlib import Path

import pandas as pd
from tqdm.notebook import tqdm

from clustermatch.utils import simplify_string
from clustermatch import conf

# %% [markdown]
# # Settings

# %%
OUTPUT_DIR = conf.GTEX["BASE_DIR"] / "data_by_tissue"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% [markdown]
# # Data loading

# %% [markdown]
# ## GTEx v8

# %% [markdown]
# ### Sample metadata

# %%
gtex_sample_attrs = pd.read_csv(
    conf.GTEX["SAMPLE_ATTRS_FILE"],
    sep="\t",
)

# %%
gtex_sample_attrs.shape

# %%
gtex_sample_attrs.head()

# %% [markdown]
# # Get tissues names

# %%
gtex_tissues = gtex_sample_attrs["SMTSD"].unique()
display(len(gtex_tissues))
display(gtex_tissues)

# %% [markdown]
# # Get sample IDs for each tissue

# %%
# first, get all sample IDs with expression data
gtex_all_sample_ids_with_expr_data = set(
    pd.read_csv(
        conf.GTEX["DATA_TPM_GCT_FILE"],
        sep="\t",
        skiprows=2,
        nrows=1,
        usecols=lambda x: x not in ("Name", "Description"),
    ).columns
)

# %%
len(gtex_all_sample_ids_with_expr_data)

# %%
list(gtex_all_sample_ids_with_expr_data)[:10]

# %%
# get sample IDs by tissue
sample_ids_by_tissue = {
    tissue_name: sorted(
        list(
            gtex_all_sample_ids_with_expr_data.intersection(
                set(
                    gtex_sample_attrs[gtex_sample_attrs["SMTSD"] == tissue_name][
                        "SAMPID"
                    ].tolist()
                )
            )
        )
    )
    for tissue_name in gtex_tissues
}

# %%
assert len(gtex_tissues) == len(sample_ids_by_tissue)

# %%
sample_ids_by_tissue["Whole Blood"][:10]

# %%
# all IDs are unique
assert all(
    [
        len(sample_ids_by_tissue[tissue_name])
        == len(set(sample_ids_by_tissue[tissue_name]))
        for tissue_name in sample_ids_by_tissue.keys()
    ]
)

# %% [markdown]
# ## Show sample size by tissue

# %%
tissue_sample_size = pd.DataFrame(
    [{"tissue": k, "sample_size": len(v)} for k, v in sample_ids_by_tissue.items()]
)

# %%
tissue_sample_size = tissue_sample_size.sort_values("sample_size", ascending=False)
display(tissue_sample_size)

# %%
# some testing
_tmp = tissue_sample_size.set_index("tissue").squeeze()
assert _tmp.loc["Muscle - Skeletal"] == 803
assert _tmp.loc["Whole Blood"] == 755
assert _tmp.loc["Skin - Not Sun Exposed (Suprapubic)"] == 604
assert _tmp.loc["Kidney - Medulla"] == 4

# %% [markdown]
# These numbers match those you can find here: https://gtexportal.org/home/tissueSummaryPage#sampleCountsPerTissue

# %% [markdown]
# # Split expression data by tissue

# %%
pbar = tqdm(tissue_sample_size["tissue"])

gene_id_symbol_map_tuples = set()

for tissue_name in pbar:
    pbar.set_description(tissue_name)

    tissue_ids = sample_ids_by_tissue[tissue_name]
    if len(tissue_ids) == 0:
        continue

    tissue_data = pd.read_csv(
        conf.GTEX["DATA_TPM_GCT_FILE"],
        sep="\t",
        skiprows=2,
        usecols=["Name", "Description"] + tissue_ids,
    )

    tissue_data = tissue_data.rename(
        columns={
            "Name": "gene_ens_id",
            "Description": "gene_symbol",
        }
    )

    # add gene id / gene symbol to mapping variable
    gene_id_symbol_map_tuples.update(
        tissue_data[["gene_ens_id", "gene_symbol"]].itertuples(index=False)
    )

    tissue_data = tissue_data.drop(columns=["gene_symbol"]).set_index("gene_ens_id")

    assert tissue_data.index.is_unique
    assert tissue_data.columns.is_unique

    # save
    tissue_name_simple = simplify_string(simplify_string(tissue_name.lower()))
    tissue_data.to_pickle(path=OUTPUT_DIR / f"gtex_v8_data_{tissue_name_simple}.pkl")

# %% [markdown]
# ## Testing

# %%
_tmp = pd.read_pickle(OUTPUT_DIR / "gtex_v8_data_brain_cerebellar_hemisphere.pkl")

# %%
# taken from GTEx webpage (see above)
assert _tmp.shape[1] == 215

# %%
assert "GTEX-11DXY-0011-R11a-SM-DNZZN" in _tmp.columns
assert "GTEX-WL46-0011-R11A-SM-3MJFT" in _tmp.columns
assert "GTEX-ZF28-0011-R11a-SM-4WWEI" in _tmp.columns

# %%
_v = _tmp.loc["ENSG00000223972.5", "GTEX-11DXY-0011-R11a-SM-DNZZN"]
assert _v == 0.04045, _v
_v = _tmp.loc["ENSG00000278267.1", "GTEX-11DXY-0011-R11a-SM-DNZZN"]
assert _v == 0.0, _v

_v = _tmp.loc["ENSG00000233327.10", "GTEX-WL46-0011-R11A-SM-3MJFT"]
assert _v == 146.4000, _v
_v = _tmp.loc["ENSG00000237118.2", "GTEX-WL46-0011-R11A-SM-3MJFT"]
assert _v == 0.3357, _v

_v = _tmp.loc["ENSG00000233327.10", "GTEX-ZF28-0011-R11a-SM-4WWEI"]
assert _v == 30.7200, _v
_v = _tmp.loc["ENSG00000186907.7", "GTEX-ZF28-0011-R11a-SM-4WWEI"]
assert _v == 0.94720, _v

# %% [markdown]
# # Save gene mappings

# %%
list(gene_id_symbol_map_tuples)[:5]

# %%
gene_mappings = pd.DataFrame(gene_id_symbol_map_tuples)

# %%
gene_mappings.shape

# %%
gene_mappings.head()

# %% [markdown]
# ## Save

# %%
output_filename = conf.GTEX["BASE_DIR"] / "gtex_gene_id_symbol_mappings.pkl"
display(output_filename)

# %%
gene_mappings.to_pickle(output_filename)

# %% [markdown]
# ## Testing

# %%
gene_mappings = pd.read_pickle(output_filename)

# %%
# no null
assert gene_mappings.dropna(how="any").shape == gene_mappings.shape

# %%
# no duplicates
assert gene_mappings.drop_duplicates().shape == gene_mappings.shape

# %%
# check gene id and gene symbol lengths (check no empty entries)
_tmp = gene_mappings.copy()
_tmp = _tmp.assign(id_len=gene_mappings["gene_ens_id"].apply(len))
_tmp = _tmp.assign(symbol_len=gene_mappings["gene_symbol"].apply(len))

# %%
_tmp_unique = _tmp["id_len"].unique()
display(_tmp_unique)

# %%
_tmp.drop_duplicates(subset=["id_len"])

# %% [markdown]
# Unique gene id lengths seem to be valid

# %%
assert list(_tmp_unique) == [17, 18, 24, 23]

# %%
_tmp_unique = _tmp["symbol_len"].unique()
display(_tmp_unique)

# %% [markdown]
# No gene symbol is empty, that's good

# %%
assert (_tmp_unique > 0).all()

# %%
assert _tmp_unique.min() == 1
assert _tmp_unique.max() == 19

# %%
# show how different gene symbol's lengths look like
_tmp.drop_duplicates(subset=["symbol_len"]).sort_values("symbol_len")

# %% [markdown]
# Unique gene symbol lengths seem to be valid

# %%
assert gene_mappings["gene_ens_id"].unique().shape[0] == gene_mappings.shape[0]

# %%
# some gene symbols map to multiple gene ids
display(gene_mappings["gene_symbol"].unique().shape)
assert gene_mappings["gene_symbol"].unique().shape[0] < gene_mappings.shape[0]

# %%
# show some duplicated gene symbols
gene_mappings[gene_mappings["gene_symbol"].duplicated(keep=False)].sort_values(
    "gene_symbol"
)

# %%
_tmp = gene_mappings.set_index("gene_ens_id").squeeze()

# %%
assert _tmp.loc["ENSG00000223972.5"] == "DDX11L1"
assert _tmp.loc["ENSG00000243485.5"] == "MIR1302-2HG"
assert _tmp.loc["ENSG00000274059.1"] == "5S_rRNA"  # repeated gene
assert _tmp.loc["ENSG00000275305.1"] == "5S_rRNA"  # repeated gene

# %%
