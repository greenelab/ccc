import numpy as np
from rpy2.robjects.packages import importr, data

from clustermatch.gene_enrich import run_enrich

clusterProfiler = importr("clusterProfiler")
dose = importr("DOSE")


def test_run_enrich_enrichGO_example_genes_in_entrez_ids():
    # example taken from here:
    # https://yulab-smu.top/biomedical-knowledge-mining-book/clusterprofiler-go.html
    gene_data = data(dose).fetch("geneList")["geneList"]

    # get gene names (Entrez IDs)
    gene_names = np.array(gene_data.names)
    assert np.unique(gene_names).shape[0] == gene_names.shape[0]  # unique

    gene_values = np.array(gene_data)
    gene = gene_names[np.abs(gene_values) > 2]
    assert gene.shape[0] == 207

    # create a "gene partition"
    gene_partition = np.zeros(gene_names.shape[0])
    gene_partition[np.isin(gene_names, gene)] = 1
    np.testing.assert_array_equal(np.unique(gene_partition), np.array([0, 1]))
    assert gene_partition[gene_partition == 1].shape[0] == gene.shape[0]

    # run
    all_results = run_enrich(
        gene_names,
        gene_partition,
        "enrichGO",
        "CC",
        key_type="ENTREZID",
        pvalue_cutoff=0.01,
        qvalue_cutoff=0.05,
    )
    assert all_results is not None
    assert len(all_results) == 1

    results = all_results[0]

    # partition information
    assert "n_clusters" in results.columns
    assert "cluster_id" in results.columns

    # renamed columns
    assert "gene_count" in results.columns
    assert "gene_ratio" in results.columns
    assert "bg_ratio" in results.columns
    assert "go_term_id" in results.columns
    assert "go_term_desc" in results.columns
    assert "fdr_per_partition" in results.columns


# TEST WHEN KEYTYPE IS SYMBOL AND READABLE HAS TO BE SET TO FALSE


# def test_run_enrich_compareCluster_example():
#     # example based on:
#     # https://yulab-smu.top/biomedical-knowledge-mining-book/clusterprofiler-comparecluster.html
#     gc_sample_data = data(clusterProfiler).fetch("gcSample")["gcSample"]
#
#     all_gene_ids = set()
#     for r_data in gc_sample_data:
#         # df = [np.array(r_data) for r_data in gc_sample_data]
#         all_gene_ids.update(list(r_data))
#     all_gene_ids = np.array(list(all_gene_ids))
#
#     # we need a hard partition of these gene clusters
#     partition = np.zeros(all_gene_ids.shape[0]) - 1
#     for cluster_id, r_data in enumerate(gc_sample_data):
#         gene_cluster = np.array(r_data)
#         partition[np.isin(all_gene_ids, gene_cluster)] = cluster_id
#
#     partition_clusters = np.unique(partition)
#     assert len(partition_clusters) == 8
#     assert partition_clusters.min() == 0
#     assert partition_clusters.max() == 7
#
#     # check that the last cluster has the expected number of genes
#     # the others will be different, since the example of gene clusters have
#     # overlapped genes, and here we need a hard partioning of the genes
#     assert partition[partition == 7].shape[0] == 237
#
#     # run our function
#     results = run_enrich(
#         all_gene_ids, "Some clustering id", partition, "enrichGO", "BP"
#     )


def test_run_enrich_no_enrichment():
    pass


def test_run_enrich_enrich_function_is_not_GO():
    # it should not return a simplified set of results
    pass


def test_run_enrich_different_simplify_cutoff():
    # it should not return a simplified set of results
    pass
