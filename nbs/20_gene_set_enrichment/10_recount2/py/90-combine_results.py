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
# Combines all gene enrichment results found in input directory.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
import re

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from clustermatch import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
DATASET_CONFIG = conf.RECOUNT2

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_DIR = DATASET_CONFIG["GENE_ENRICHMENT_DIR"]
display(INPUT_DIR)
assert INPUT_DIR.exists()

# %% tags=[]
OUTPUT_FILE = DATASET_CONFIG["GENE_ENRICHMENT_COMBINED_FILE"]
display(OUTPUT_FILE)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# %% [markdown] tags=[]
# # Get data files

# %% tags=[]
filename_pattern = re.compile(DATASET_CONFIG["GENE_ENRICHMENT_FILENAME_PATTERN"])

# %% tags=[]
# get input data files according to Settings
input_files = sorted(
    [
        f
        for f in INPUT_DIR.iterdir()
        if (m := re.search(filename_pattern, str(f))) is not None
        #         and m.group("corr_method") == CORRELATION_METHOD_NAME
    ]
)
display(len(input_files))
display(input_files[:5])

assert len(input_files) > 0

# %% [markdown] tags=[]
# ## Preview data

# %% tags=[]
display(input_files[0])

# %% tags=[]
_tmp_df = pd.read_pickle(input_files[0])

# %% tags=[]
_tmp_df.shape

# %% tags=[]
_tmp_df.sample(n=5, random_state=0)

# %% [markdown] tags=[]
# # Run

# %% [markdown] tags=[]
# ## Read data, convert dtypes, add new metrics

# %% tags=[]
all_results = []

for f_full in tqdm(input_files, ncols=100):
    f_name = f_full.name

    f_data = pd.read_pickle(f_full)
    # f_data = f_data.rename(
    # columns={
    # "Count": "gene_count",
    # "GeneRatio": "gene_ratio",
    # "BgRatio": "bg_ratio",
    # "ID": "go_term_id",
    # "Description": "go_term_desc",
    # "Cluster": "cluster_id",
    # "clustering_n_clusters": "n_clusters",
    # "p.adjust": "fdr_per_file",
    # }
    # )

    # genes in cluster
    f_data = f_data.assign(
        gene_total=f_data["gene_ratio"].apply(lambda x: int(x.split("/")[1]))
    )

    # background genes
    f_data = f_data.assign(
        bg_count=f_data["bg_ratio"].apply(lambda x: int(x.split("/")[0]))
    )
    f_data = f_data.assign(
        bg_total=f_data["bg_ratio"].apply(lambda x: int(x.split("/")[1]))
    )

    # add metadata
    metadata = re.search(filename_pattern, f_name)

    f_data = f_data[
        [
            "n_clusters",
            "cluster_id",
            "go_term_id",
            "go_term_desc",
            "gene_count",
            "gene_total",
            "gene_ratio",
            "bg_count",
            "bg_total",
            "bg_ratio",
            "pvalue",
            "fdr_per_partition",
        ]
    ]

    # f_data["tissue"] = metadata.group("tissue")
    # f_data["gene_sel_strategy"] = metadata.group("gene_sel_strategy")
    f_data["corr_method"] = metadata.group("corr_method")
    f_data["clust_method"] = metadata.group("clust_method")
    f_data["enrich_func"] = metadata.group("enrich_func")
    f_data["enrich_params"] = metadata.group("enrich_params")

    all_results.append(f_data)

# %% tags=[]
df = pd.concat(all_results, ignore_index=True)

# to category dtype
df["cluster_id"] = df["cluster_id"].astype("category")
df["go_term_id"] = df["go_term_id"].astype("category")
df["go_term_desc"] = df["go_term_desc"].astype("category")
# df["tissue"] = df["tissue"].astype("category")
# df["gene_sel_strategy"] = df["gene_sel_strategy"].astype("category")
df["corr_method"] = df["corr_method"].astype("category")
df["clust_method"] = df["clust_method"].astype("category")
df["enrich_func"] = df["enrich_func"].astype("category")
df["enrich_params"] = df["enrich_params"].astype("category")

# convert to int32
df["n_clusters"] = df["n_clusters"].astype("int32")
df["gene_count"] = df["gene_count"].astype("int32")
df["gene_total"] = df["gene_total"].astype("int32")
df["bg_count"] = df["bg_count"].astype("int32")
df["bg_total"] = df["bg_total"].astype("int32")

# convert ratios to numbers
df["gene_ratio"] = df["gene_count"].div(df["gene_total"])
df["bg_ratio"] = df["bg_count"].div(df["bg_total"])

# add other metrics
df["rich_factor"] = df["gene_count"].div(df["bg_count"])
df["fold_enrich"] = df["gene_ratio"].div(df["bg_ratio"])

# %% tags=[]
# adjust for multiple testing across all results
adj_pval = multipletests(df["pvalue"], alpha=0.05, method="fdr_bh")
df = df.assign(fdr=adj_pval[1])

# %% tags=[]
df.shape

# %% tags=[]
display(df.dtypes)
assert df.dtypes.loc["cluster_id"] == "category"

# %% tags=[]
df.sample(n=5)

# %% [markdown] tags=[]
# ## Some stats

# %% tags=[]
display(df["fdr"].describe())
assert df["fdr"].min() > 0.0
assert df["fdr"].max() < 1.0

# %% tags=[]
df["n_clusters"].unique()

# %% tags=[]
# df["tissue"].unique()

# %% tags=[]
# df["gene_sel_strategy"].unique()

# %% tags=[]
df["corr_method"].unique()

# %% tags=[]
df["clust_method"].unique()

# %% tags=[]
df["enrich_params"].unique()

# %% [markdown] tags=[]
# ## Testing

# %% tags=[]
assert not df.isna().any().any()

# %% tags=[]
# test if values are correctly calculated
# _tmp = df[
#    (df.go_term_id == "GO:0035383")
#    & (df.n_clusters == 65)
#    & (df.cluster_id == "C21")
#    # & (df.tissue == "adipose_subcutaneous")
#    # & (df.gene_sel_strategy == "var_pc_log2")
#    & (df.corr_method == "clustermatch")
#    & (df.clust_method == "SpectralClustering")
#    & (df.enrich_func == "enrichGO")
#    & (df.enrich_params == "BP_full")
# ]
# assert _tmp.shape[0] == 1
# _tmp = _tmp.iloc[0]
#
# assert _tmp["gene_count"] == 15
# assert _tmp["gene_total"] == 329
# assert _tmp["gene_ratio"] == 15.0 / 329.0
# assert _tmp["bg_count"] == 34
# assert _tmp["bg_total"] == 3528
# assert _tmp["bg_ratio"] == 34.0 / 3528.0
# assert _tmp["rich_factor"] == 15.0 / 34.0
# assert _tmp["fold_enrich"] == (15.0 / 329.0) / (34.0 / 3528.0)

# %% [markdown] tags=[]
# # Save

# %% tags=[]
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

saveRDS = ro.r["saveRDS"]
readRDS = ro.r["readRDS"]

# %% tags=[]
data = df

# %% [markdown] tags=[]
# ## Pickle

# %% tags=[]
display(OUTPUT_FILE)

# %% tags=[]
data.to_pickle(OUTPUT_FILE)

# %% [markdown] tags=[]
# ## RDS

# %% tags=[]
output_rds_file = OUTPUT_FILE.with_suffix(".rds")
display(output_rds_file)

# %% tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    data_r = ro.conversion.py2rpy(data)

# %% tags=[]
data_r

# %% tags=[]
saveRDS(data_r, str(output_rds_file))

# %% tags=[]
# testing
data_r = readRDS(str(output_rds_file))

# %% tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    data_again = ro.conversion.rpy2py(data_r)

    # convert index to int, otherwise it's converted to string
    data_again.index = data_again.index.astype(int)

# %% tags=[]
data_again.shape

# %% tags=[]
data_again.head()

# %% tags=[]
pd.testing.assert_frame_equal(
    data,
    data_again,
    check_names=False,  # do not check "name" attribute of index and column
    check_exact=True,  # since this is a binary format, it should match exactly
)

# %% [markdown] tags=[]
# ## tsv.gz

# %% tags=[]
output_text_file = OUTPUT_FILE.with_suffix(".tsv.gz")
display(output_text_file)

# %% tags=[]
data.to_csv(output_text_file, sep="\t", index=False, float_format="%.5e")

# %% tags=[]
# testing
data_again = pd.read_csv(output_text_file, sep="\t")  # , index_col=0)

# %% tags=[]
data_again.shape

# %% tags=[]
data_again.head()

# %% tags=[]
pd.testing.assert_frame_equal(
    data,
    data_again,
    check_names=False,  # do not check "name" attribute of index and column
    check_dtype=False,  # do not check dtypes: do not distinguish between int64 and int32, for instance
    check_categorical=False,
    check_exact=False,
    rtol=1e-5,
    atol=5e-5,
)

# %% tags=[]
