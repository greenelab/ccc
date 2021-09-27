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
# It computes gene enrichment on *all* the clustering results obtained using some correlation measure on GTEx v8 (specified under `Settings` below).

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

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
CORRELATION_METHOD_NAME = "spearman_full"

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
INPUT_DIR = conf.GTEX["CLUSTERING_DIR"]
display(INPUT_DIR)
assert INPUT_DIR.exists()

# %%
# this directory has the input data given to the clustering methods
SIMILARITY_MATRICES_DIR = conf.GTEX["SIMILARITY_MATRICES_DIR"]
display(SIMILARITY_MATRICES_DIR)

# %%
SIMILARITY_MATRIX_FILENAME_TEMPLATE = conf.GTEX["SIMILARITY_MATRIX_FILENAME_TEMPLATE"]
display(SIMILARITY_MATRIX_FILENAME_TEMPLATE)

# %% tags=[]
OUTPUT_DIR = conf.GTEX["GENE_ENRICHMENT_DIR"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Get data files

# %%
filename_pattern = re.compile(conf.GTEX["CLUSTERING_FILENAME_PATTERN"])

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
# # clusterProfiler

# %% [markdown]
# ## Define functions

# %%
simplified_cutoff_str = f"{SIMPLIFY_CUTOFF:.2f}".replace(".", "")
display(simplified_cutoff_str)


# %% tags=[]
def run_enrich(
    all_gene_ids,
    clustering_id,
    partition,
    enrich_function,
    ontology,
    simplify_cutoff=None,
):
    """
    TODO
    """
    # this modules need to be imported from inside this function (if the function will be
    # run in different processes, for instance, using ProcessPoolExecutor). Otherwise,
    # rpy2 raises some weird exceptions
    from rpy2.robjects.packages import importr
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()

    clusterProfiler = importr("clusterProfiler")

    # get partition numbers
    n_genes = partition.shape[0]
    n_clusters = np.unique(partition).shape[0]

    # create a clusterProfiler-friendly structure to indicate which
    # genes belong to each cluster
    genes_per_cluster = {}
    for c in pd.Series(partition).value_counts().index:
        genes_per_cluster[f"C{c:n}"] = [
            g.split(".")[0] for g in all_gene_ids[partition == c]
        ]

    assert len(genes_per_cluster) == n_clusters
    assert sum(map(lambda x: len(set(x)), genes_per_cluster.values())) == n_genes

    genes_per_cluster = robjects.ListVector(genes_per_cluster)

    # run clusterProfiler
    ck = clusterProfiler.compareCluster(
        geneClusters=genes_per_cluster,
        OrgDb="org.Hs.eg.db",
        keyType="ENSEMBL",
        universe=all_gene_ids,
        fun=enrich_function,
        pAdjustMethod="BH",
        pvalueCutoff=0.05,
        ont=ontology,
        readable=True,
    )

    results = []

    # save full results (all enriched terms, even if they are very similar)
    df = ck.slots["compareClusterResult"]
    df["clustering_id"] = clustering_id
    df["clustering_n_clusters"] = n_clusters
    results.append(df)

    # save simplified results
    if simplify_cutoff is not None and ENRICH_FUNCTION in ("enrichGO", "gseGO"):
        ck = clusterProfiler.simplify(ck, cutoff=simplify_cutoff)
        df = ck.slots["compareClusterResult"]
        df["clustering_id"] = clustering_id
        df["clustering_n_clusters"] = n_clusters
        results.append(df)

    return tuple(results)


# %% [markdown]
# ## Run

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

        tissue = m.group("tissue")
        gene_sel_strategy = m.group("gene_sel_strategy")
        corr_method = m.group("corr_method")

        # update pbar description
        pbar.set_description(f"{tissue}/{gene_sel_strategy}")

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
            corr_method=corr_method.split("_")[0],
        )

        # get the universe of genes
        all_gene_ids = pd.read_pickle(
            SIMILARITY_MATRICES_DIR / similarity_matrix_filename
        ).index.tolist()
        all_gene_ids = np.array([g.split(".")[0] for g in all_gene_ids])
        assert all_gene_ids.shape[0] == n_genes

        # iterate over clustering solutions (partitions) and GO ontologies
        futures = {
            executor.submit(
                run_enrich,
                all_gene_ids,
                cr_idx,
                cr.partition,
                ENRICH_FUNCTION,
                ont,
                SIMPLIFY_CUTOFF,
            ): ont
            for cr_idx, cr in clustering_df.sort_values("n_clusters").iterrows()
            for ont in GO_ONTOLOGIES
        }

        # collect results
        results_full = defaultdict(list)
        results_simplified = defaultdict(list)

        for task in as_completed(futures):
            ont = futures[task]
            task_results = task.result()

            results_full[ont].append(task_results[0])

            if len(task_results) > 1:
                results_simplified[ont].append(task_results[1])

            pbar.update(1)

        # merge and serve
        pbar.set_description(f"{tissue}/{gene_sel_strategy}/saving")

        for ontology in GO_ONTOLOGIES:
            # full
            results_full_df = pd.concat(
                results_full[ontology], ignore_index=True
            ).sort_values(["clustering_n_clusters", "p.adjust"])

            results_full_df.to_pickle(
                OUTPUT_DIR
                / f"{clustering_filepath.stem}-{ENRICH_FUNCTION}-{ontology}_full.pkl",
            )

            # simplified
            if len(results_simplified) > 0:
                results_simplified_df = pd.concat(
                    results_simplified[ontology], ignore_index=True
                ).sort_values(["clustering_n_clusters", "p.adjust"])

                results_simplified_df.to_pickle(
                    OUTPUT_DIR
                    / f"{clustering_filepath.stem}-{ENRICH_FUNCTION}-{ontology}_simplified_{simplified_cutoff_str}.pkl",
                )

# %% tags=[]
