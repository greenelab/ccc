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
# TODO/UPDATE: It computes gene enrichment on *all* the clustering results obtained using some correlation measure on GTEx v8 (specified under `Settings` below).

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from clustermatch import conf
from clustermatch.gene_enrich import run_enrich

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
DATASET_CONFIG = conf.GTEX

# %% tags=[]
CORRELATION_METHOD_NAME = "clustermatch_k2"

# %% tags=[]
# GENE_SELECTION_STRATEGY = "var_pc_log2"

# %% tags=[]
# clusterProfiler settings
ENRICH_FUNCTION = "enrichPathway"
ENRICH_PARAMS = "human"

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_DIR = DATASET_CONFIG["CLUSTERING_DIR"]
display(INPUT_DIR)
assert INPUT_DIR.exists()

# %% tags=[]
# this directory has the input data given to the clustering methods
SIMILARITY_MATRICES_DIR = DATASET_CONFIG["SIMILARITY_MATRICES_DIR"]
display(SIMILARITY_MATRICES_DIR)

# %% tags=[]
SIMILARITY_MATRIX_FILENAME_TEMPLATE = DATASET_CONFIG[
    "SIMILARITY_MATRIX_FILENAME_TEMPLATE"
]
display(SIMILARITY_MATRIX_FILENAME_TEMPLATE)

# %% tags=[]
OUTPUT_DIR = DATASET_CONFIG["GENE_ENRICHMENT_DIR"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Get data files

# %% tags=[]
filename_pattern = re.compile(DATASET_CONFIG["CLUSTERING_FILENAME_PATTERN"])

# %% tags=[]
# get input data files according to Settings
input_files = sorted(
    [
        f
        for f in INPUT_DIR.iterdir()
        if (m := re.search(filename_pattern, str(f))) is not None
        and m.group("corr_method") == CORRELATION_METHOD_NAME
    ]
)
display(len(input_files))
display(input_files[:5])

assert len(input_files) > 0

# %% [markdown] tags=[]
# # Preview input data

# %% [markdown] tags=[]
# ## Clustering results

# %% tags=[]
tmp = pd.read_pickle(input_files[0])

# %% tags=[]
tmp.shape

# %% tags=[]
tmp.head()

# %% [markdown] tags=[]
# ## Similarity matrices (input to clustering methods)

# %% tags=[]
similarity_matrix_filename = SIMILARITY_MATRIX_FILENAME_TEMPLATE.format(
    tissue="adipose_subcutaneous",
    gene_sel_strategy="var_pc_log2",
    corr_method=CORRELATION_METHOD_NAME.split("_")[0]
    if not CORRELATION_METHOD_NAME.startswith("clustermatch")
    else CORRELATION_METHOD_NAME,
)
display(similarity_matrix_filename)

# %% tags=[]
tmp = pd.read_pickle(SIMILARITY_MATRICES_DIR / similarity_matrix_filename)

# %% tags=[]
tmp.shape

# %% tags=[]
tmp.head()

# %% [markdown] tags=[]
# ### Convert Ensembl Gene IDs to Entrez IDs

# %% tags=[]
input_filename = conf.GTEX["DATA_DIR"] / "gtex_entrez_gene_ids_mappings.pkl"
display(input_filename)
assert input_filename.exists()

# %% tags=[]
gene_ids_mappings = pd.read_pickle(input_filename)

# %% tags=[]
gene_ids_mappings.shape

# %% tags=[]
gene_ids_mappings.head()

# %% tags=[]
gene_id_maps = gene_ids_mappings.set_index("gene_ens_id_v")["entrez_id"].to_dict()

# %% tags=[]
dict(list(gene_id_maps.items())[0:2])

# %% tags=[]
# is map from ensembl to entrez unique?
_tmp_index = [gene_id_maps[x] for x in tmp.index if x in gene_id_maps]
display(len(_tmp_index))
display(_tmp_index[:5])

# %% [markdown] tags=[]
# # Run

# %% tags=[]
n_partitions_per_file = pd.read_pickle(input_files[0]).shape[0]
display(n_partitions_per_file)

# %% tags=[]
# the number of tasks is the number of input files times number of partitions per file
n_tasks = len(input_files) * n_partitions_per_file
n_tasks = int(n_tasks)
display(f"number of tasks: {n_tasks}")

with ProcessPoolExecutor(max_workers=conf.GENERAL["N_JOBS"]) as executor, tqdm(
    total=n_tasks, ncols=100
) as pbar:
    for clustering_filepath in input_files:
        # extract from input clustering filename some sections, such as tissue name, etc
        m = re.search(filename_pattern, str(clustering_filepath.name))

        tissue = m.group("tissue")
        gene_sel_strategy = m.group("gene_sel_strategy")
        corr_method = m.group("corr_method")

        # update pbar description
        pbar.set_description(f"{tissue}/{gene_sel_strategy}")
        #         pbar.set_description(f"{corr_method}")

        # create output filepath template
        full_output_filename_template = (
            f"{clustering_filepath.stem}-{ENRICH_FUNCTION}-{ENRICH_PARAMS}.pkl"
        )

        # read clustering results
        clustering_df = pd.read_pickle(clustering_filepath)

        # get partitions' numbers
        tmp_partition = clustering_df.iloc[0].partition
        n_genes = tmp_partition.shape[0]
        n_clusters = np.unique(tmp_partition).shape[0]

        # use those sections to read the list of genes from the input data
        # file that the clustering algorithm received
        similarity_matrix_filename = SIMILARITY_MATRIX_FILENAME_TEMPLATE.format(
            tissue=tissue,
            gene_sel_strategy=gene_sel_strategy,
            corr_method=corr_method.split("_")[0]
            if not corr_method.startswith("clustermatch")
            else corr_method,
        )

        # get the universe of genes
        all_gene_ens_ids = pd.read_pickle(
            SIMILARITY_MATRICES_DIR / similarity_matrix_filename
        ).index.tolist()

        # convert gene ensembl ids to entrez and create clustering partition mask
        partition_mask = []
        all_gene_ids = []
        entrez_ids_added = set()  # this is faster

        for x in all_gene_ens_ids:
            if x not in gene_id_maps:
                partition_mask.append(False)
                continue

            new_entrez_id = gene_id_maps[x]

            # TODO: maybe this avoiding of repeated gene ids is not necessary?
            # do not add repeated ids
            if new_entrez_id in entrez_ids_added:
                partition_mask.append(False)
                continue

            all_gene_ids.append(new_entrez_id)
            entrez_ids_added.add(new_entrez_id)
            partition_mask.append(True)

        partition_mask = np.array(partition_mask, dtype=bool)
        all_gene_ids = np.array(all_gene_ids)
        assert np.unique(all_gene_ids).shape[0] == all_gene_ids.shape[0]
        assert all_gene_ids.shape[0] == np.sum(partition_mask)

        # iterate over clustering solutions (partitions) and GO ontologies
        futures = [
            executor.submit(
                run_enrich,
                all_gene_ids,
                "ENTREZID",
                cr.partition[partition_mask],
                ENRICH_FUNCTION,
                ENRICH_PARAMS,
            )
            for cr_idx, cr in clustering_df.sort_values("n_clusters").iterrows()
        ]

        # collect results
        results_full = []

        for task in as_completed(futures):
            task_results = task.result()

            # continue if no enrichment found
            if task_results is None:
                pbar.update(1)
                continue

            results_full.append(task_results)

            pbar.update(1)

        if len(results_full) == 0:
            # no significant results, continue
            continue

        # merge and serve
        pbar.set_description(f"{tissue}/{gene_sel_strategy}/saving")

        # full
        results_full_df = pd.concat(results_full, ignore_index=True).sort_values(
            ["n_clusters", "pvalue_adjust"]
        )

        results_full_df.to_pickle(
            OUTPUT_DIR
            / f"{clustering_filepath.stem}-{ENRICH_FUNCTION}-{ENRICH_PARAMS}.pkl",
        )

# %% tags=[]
