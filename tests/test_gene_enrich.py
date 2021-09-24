import numpy as np
from rpy2.robjects.packages import importr, data

clusterProfiler = importr("clusterProfiler")


def test_run_enrich_original_example():
    # example based on:
    # https://yulab-smu.top/biomedical-knowledge-mining-book/clusterprofiler-comparecluster.html
    gc_sample_data = data(clusterProfiler).fetch("gcSample")["gcSample"]

    all_gene_ids = set()
    for r_data in gc_sample_data:
        # df = [np.array(r_data) for r_data in gc_sample_data]
        all_gene_ids.update(list(r_data))
    all_gene_ids = np.array(list(all_gene_ids))

    # we need a hard partition of these gene clusters
    partition = np.zeros(all_gene_ids.shape[0]) - 1
    for cluster_id, r_data in enumerate(gc_sample_data):
        gene_cluster = np.array(r_data)
        partition[np.isin(all_gene_ids, gene_cluster)] = cluster_id

    partition_clusters = np.unique(partition)
    assert len(partition_clusters) == 8
    assert partition_clusters.min() == 0
    assert partition_clusters.max() == 7

    # check that the last cluster has the expected number of genes
    # the others will be different, since the example of gene clusters have
    # overlapped genes, and here we need a hard partioning of the genes
    assert partition[partition == 7].shape[0] == 237

    # run our function
    results = run_enrich(all_gene_ids, "Some clustering id", partition, "enrichGO", "BP")


def test_run_enrich_no_enrichment():
    pass


def test_run_enrich_enrich_function_is_not_GO():
    # it should not return a simplified set of results
    pass


def test_run_enrich_different_simplify_cutoff():
    # it should not return a simplified set of results
    pass
