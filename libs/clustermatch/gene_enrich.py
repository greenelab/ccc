
ENRICH_GO_FUNC_NAME = "enrichGO"
ENRICH_KEGG_FUNC_NAME = "enrichKEGG"

COLUMNS_RENAME = {
    "Count": "gene_count",
    "GeneRatio": "gene_ratio",
    "BgRatio": "bg_ratio",
    "ID": "term_id",
    "Description": "term_desc",
    "Cluster": "cluster_id",
    "p.adjust": "pvalue_adjust",
    "geneID": "gene_id",
}


def _get_dataframe(ck, n_clusters):
    """
    Extracts a pandas dataframe from a clusterProfiler.compareClusters result object.

    TODO: finish
    """
    df = ck.slots["compareClusterResult"]

    df["n_clusters"] = n_clusters
    df = df.rename(columns=COLUMNS_RENAME)

    df = df.assign(
        gene_total=df["gene_ratio"].apply(lambda x: int(x.split("/")[1])),
        bg_count=df["bg_ratio"].apply(lambda x: int(x.split("/")[0])),
        bg_total=df["bg_ratio"].apply(lambda x: int(x.split("/")[1]))
    )

    return df


def run_enrich(
    all_gene_ids,
    key_type,
    partition,
    enrich_function,
    ontology,
    pvalue_cutoff=0.05,
    qvalue_cutoff=0.20,
    simplify_cutoff=None,
):
    """
    TODO
    """
    # the rpy2 modules need to be imported from inside this function (if the
    # function will be run in different processes, for instance, using
    # ProcessPoolExecutor). Otherwise, rpy2 raises some weird exceptions
    import numpy as np
    import pandas as pd

    from rpy2.robjects.packages import importr
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.rinterface_lib.embedded import RRuntimeError

    pandas2ri.activate()
    clusterProfiler = importr("clusterProfiler")

    # the universe of genes are unique
    assert all_gene_ids.shape[0] == np.unique(all_gene_ids).shape[0]

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
    compare_cluster_arguments = {
        "geneClusters": genes_per_cluster,
        "keyType": key_type,
        "universe": all_gene_ids,
        "fun": enrich_function,
        "pAdjustMethod": "BH",
        "pvalueCutoff": pvalue_cutoff,
        "qvalueCutoff": qvalue_cutoff,
    }

    if enrich_function == ENRICH_GO_FUNC_NAME:
        compare_cluster_arguments.update({
            "ont": ontology,
            "readable": True if key_type != "SYMBOL" else False,
            "OrgDb": "org.Hs.eg.db",
        })
    elif enrich_function == ENRICH_KEGG_FUNC_NAME:
        if compare_cluster_arguments["keyType"] != "ENTREZID":
            raise ValueError("Input genes must be Entrez gene IDs")

        del compare_cluster_arguments["keyType"]

    try:
        ck = clusterProfiler.compareCluster(**compare_cluster_arguments)
    except RRuntimeError as e:
        if "No enrichment found in any of gene cluster" not in str(e):
            raise

        # no enrichment found, return empty tuple
        return tuple()

    results = []

    # save full results (all enriched terms, even if they are very similar)
    df = _get_dataframe(ck, n_clusters)
    results.append(df)

    # save simplified results
    if simplify_cutoff is not None and enrich_function in (ENRICH_GO_FUNC_NAME, "gseGO"):
        ck = clusterProfiler.simplify(ck, cutoff=simplify_cutoff)
        df = _get_dataframe(ck, n_clusters)
        results.append(df)

    return tuple(results)
