def run_enrich(
    all_gene_ids,
    partition,
    enrich_function,
    ontology,
    key_type="ENSEMBL",
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
    try:
        ck = clusterProfiler.compareCluster(
            geneClusters=genes_per_cluster,
            OrgDb="org.Hs.eg.db",
            keyType=key_type,
            universe=all_gene_ids,
            fun=enrich_function,
            pAdjustMethod="BH",
            pvalueCutoff=pvalue_cutoff,
            qvalueCutoff=qvalue_cutoff,
            ont=ontology,
            readable=True if key_type != "SYMBOL" else False,
        )
    except RRuntimeError as e:
        if "No enrichment found in any of gene cluster" not in str(e):
            raise

        # no enrichment found, return empty tuple
        return tuple()

    columns_rename = {
        "Count": "gene_count",
        "GeneRatio": "gene_ratio",
        "BgRatio": "bg_ratio",
        "ID": "go_term_id",
        "Description": "go_term_desc",
        "Cluster": "cluster_id",
        "p.adjust": "fdr_per_partition",
    }

    results = []

    # save full results (all enriched terms, even if they are very similar)
    df = ck.slots["compareClusterResult"]
    df["n_clusters"] = n_clusters
    df = df.rename(columns=columns_rename)
    results.append(df)

    # save simplified results
    if simplify_cutoff is not None and enrich_function in ("enrichGO", "gseGO"):
        ck = clusterProfiler.simplify(ck, cutoff=simplify_cutoff)
        df = ck.slots["compareClusterResult"]
        df["n_clusters"] = n_clusters
        df = df.rename(columns=columns_rename)
        results.append(df)

    return tuple(results)
