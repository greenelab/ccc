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
# It computes gene enrichment on *all* the clustering results (obtained using some correlation measure) on a dataset.
# All these settings are specified below under `Settings`.

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

# %%
DATASET_CONFIG = conf.RECOUNT2

# %% tags=[]
# we do not need to split by method for recount2
# CORRELATION_METHOD_NAME = "pearson_abs"

# %% tags=[]
# GENE_SELECTION_STRATEGY = "var_pc_log2"

# %%
# clusterProfiler settings
ENRICH_FUNCTION = "enrichGO"
SIMPLIFY_CUTOFF = 0.7
GO_ONTOLOGIES = ("BP", "CC", "MF")

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_DIR = DATASET_CONFIG["CLUSTERING_DIR"]
display(INPUT_DIR)
assert INPUT_DIR.exists()

# %%
# this directory has the input data given to the clustering methods
SIMILARITY_MATRICES_DIR = DATASET_CONFIG["SIMILARITY_MATRICES_DIR"]
display(SIMILARITY_MATRICES_DIR)

# %%
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

# %%
filename_pattern = re.compile(DATASET_CONFIG["CLUSTERING_FILENAME_PATTERN"])

# %% tags=[]
# get input data files according to Settings
input_files = sorted(
    [
        f
        for f in INPUT_DIR.iterdir()
        if (m := re.search(filename_pattern, str(f))) is not None
    ]
)
display(len(input_files))
display(input_files[:5])

assert len(input_files) > 0

# %% [markdown]
# # Preview input data

# %% [markdown]
# ## Clustering results

# %%
tmp = pd.read_pickle(input_files[0])

# %%
tmp.shape

# %%
tmp.head()

# %% [markdown]
# ## Similarity matrices (input to clustering methods)

# %%
similarity_matrix_filename = SIMILARITY_MATRIX_FILENAME_TEMPLATE.format(
    corr_method="clustermatch_k2",
)
display(similarity_matrix_filename)

# %%
tmp = pd.read_pickle(SIMILARITY_MATRICES_DIR / similarity_matrix_filename)

# %%
tmp.shape

# %%
tmp.head()

# %% [markdown]
# # Run

# %%
simplified_cutoff_str = f"{SIMPLIFY_CUTOFF:.2f}".replace(".", "")
display(simplified_cutoff_str)

# %%
n_partitions_per_file = pd.read_pickle(input_files[0]).shape[0]
display(n_partitions_per_file)

# %%
# the number of tasks is the number of input files times number of partitions per file times 3 (BP, CC, MF)
n_tasks = len(input_files) * n_partitions_per_file * 3
n_tasks = int(n_tasks)
display(f"number of tasks: {n_tasks}")

with ProcessPoolExecutor(max_workers=conf.GENERAL["N_JOBS"]) as executor, tqdm(
    total=n_tasks, ncols=100
) as pbar:
    for clustering_filepath in input_files:
        # extract from input clustering filename some sections, such as tissue name, etc
        m = re.search(filename_pattern, str(clustering_filepath.name))

        #         tissue = m.group("tissue")
        #         gene_sel_strategy = m.group("gene_sel_strategy")
        corr_method = m.group("corr_method")

        # update pbar description
        #         pbar.set_description(f"{tissue}/{gene_sel_strategy}")
        pbar.set_description(f"{corr_method}")

        # create output filepath template
        full_output_filename_template = (
            f"{clustering_filepath.stem}-{ENRICH_FUNCTION}-{{ontology}}_full.pkl"
        )
        simplified_output_filename_template = f"{clustering_filepath.stem}-{ENRICH_FUNCTION}-{{ontology}}_simplified_{simplified_cutoff_str}.pkl"

        # read clustering results
        clustering_df = pd.read_pickle(clustering_filepath)

        # get partitions' numbers
        tmp_partition = clustering_df.iloc[0].partition
        n_genes = tmp_partition.shape[0]
        n_clusters = np.unique(tmp_partition).shape[0]

        # use those sections to read the list of genes from the input data
        # file that the clustering algorithm received
        similarity_matrix_filename = SIMILARITY_MATRIX_FILENAME_TEMPLATE.format(
            #             tissue=tissue,
            #             gene_sel_strategy=gene_sel_strategy,
            corr_method=corr_method.split("_")[0]
            if not corr_method.startswith("clustermatch")
            else corr_method,
        )

        # get the universe of genes
        all_gene_ids = pd.read_pickle(
            SIMILARITY_MATRICES_DIR / similarity_matrix_filename
        ).index.tolist()
        all_gene_ids = np.array(all_gene_ids)
        assert all_gene_ids.shape[0] == n_genes

        # iterate over clustering solutions (partitions) and GO ontologies
        futures = {
            executor.submit(
                run_enrich,
                all_gene_ids,
                cr.partition,
                ENRICH_FUNCTION,
                ontology,
                key_type="SYMBOL",
                simplify_cutoff=SIMPLIFY_CUTOFF,
            ): ontology
            for cr_idx, cr in clustering_df.sort_values("n_clusters").iterrows()
            for ontology in GO_ONTOLOGIES
            if not (
                (
                    OUTPUT_DIR / full_output_filename_template.format(ontology=ontology)
                ).exists()
                and (
                    OUTPUT_DIR
                    / simplified_output_filename_template.format(ontology=ontology)
                ).exists()
            )
        }

        # FIXME: this n_expected here is horrible
        #  I leave it here for now
        futures_n_expected = int(len(GO_ONTOLOGIES) * clustering_df.shape[0])

        futures_diff = futures_n_expected - len(futures)
        if futures_diff > 0:
            pbar.update(futures_diff)

        if futures_diff == futures_n_expected:
            continue

        # collect results
        results_full = defaultdict(list)
        results_simplified = defaultdict(list)

        for task in as_completed(futures):
            ont = futures[task]
            task_results = task.result()

            # continue if no enrichment found
            if len(task_results) == 0:
                pbar.update(1)
                continue

            results_full[ont].append(task_results[0])

            if len(task_results) > 1:
                results_simplified[ont].append(task_results[1])

            pbar.update(1)

        if len(results_full) == 0:
            # no significant results, continue
            continue

        # merge and serve
        pbar.set_description(f"{corr_method}/saving")

        for ontology in GO_ONTOLOGIES:
            # full
            results_full_df = pd.concat(
                results_full[ontology], ignore_index=True
            ).sort_values(["n_clusters", "fdr_per_partition"])

            results_full_df.to_pickle(
                OUTPUT_DIR
                / f"{clustering_filepath.stem}-{ENRICH_FUNCTION}-{ontology}_full.pkl",
            )

            # simplified
            if len(results_simplified) > 0:
                results_simplified_df = pd.concat(
                    results_simplified[ontology], ignore_index=True
                ).sort_values(["n_clusters", "fdr_per_partition"])

                results_simplified_df.to_pickle(
                    OUTPUT_DIR
                    / f"{clustering_filepath.stem}-{ENRICH_FUNCTION}-{ontology}_simplified_{simplified_cutoff_str}.pkl",
                )

# %% tags=[]
