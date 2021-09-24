import numpy as np

import rpy2.robjects as ro
from rpy2.robjects.packages import importr, data
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

clusterProfiler = importr("clusterProfiler")


def test_run_enrich_original_example():
    # example taken from:
    # https://yulab-smu.top/biomedical-knowledge-mining-book/clusterprofiler-comparecluster.html
    gc_sample_data = data(clusterProfiler).fetch("gcSample")["gcSample"]

    all_gene_ids = set()
    for r_data in gc_sample_data:
        # df = [np.array(r_data) for r_data in gc_sample_data]
        all_gene_ids.update(list(r_data))
    all_gene_ids = np.array(list(all_gene_ids))

    partition = np.zeros(all_gene_ids.shape[0]) - 1
    for cluster_id, r_data in enumerate(gc_sample_data):
        gene_cluster = np.array(r_data)
        partition[np.isin(all_gene_ids, gene_cluster)] = cluster_id

    partition_clusters = np.unique(partition)
    assert len(partition_clusters) == 8


def test_run_enrich_no_enrichment():
    pass


def test_run_enrich_enrich_function_is_not_GO():
    # it should not return a simplified set of results
    pass


def test_run_enrich_different_simplify_cutoff():
    # it should not return a simplified set of results
    pass
