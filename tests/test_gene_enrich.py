import pytest
import numpy as np
from rpy2.robjects.packages import importr, data
from rpy2.robjects import pandas2ri

from clustermatch.gene_enrich import run_enrich

clusterProfiler = importr("clusterProfiler")
dose = importr("DOSE")


def test_run_enrich_enrichgo_example_genes_in_entrez_id():
    pandas2ri.deactivate()

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
    results = run_enrich(
        gene_names,
        "ENTREZID",
        gene_partition,
        "enrichGO",
        "CC",
        pvalue_cutoff=0.01,
        qvalue_cutoff=0.05,
    )
    assert results is not None
    # assert len(all_results) == 1

    # results = all_results
    assert results.shape[0] == 23

    # partition information
    assert "n_clusters" in results.columns
    unique_n_clusters = results["n_clusters"].unique()
    assert unique_n_clusters.shape[0] == 1
    assert 2 in unique_n_clusters

    assert "cluster_id" in results.columns
    unique_cluster_id = results["cluster_id"].unique()
    # only one cluster (C1) has significant results
    assert unique_cluster_id.shape[0] == 1
    assert "C1" in unique_cluster_id

    # check one row, with go term id: GO:0005819
    assert "term_id" in results.columns
    assert "gene_count" in results.columns
    assert "gene_total" in results.columns
    assert "gene_ratio" in results.columns
    _row = results[results["term_id"] == "GO:0005819"]
    assert _row.shape[0] == 1
    _row = _row.iloc[0]
    assert _row["gene_ratio"] == 26 / 201.0
    assert _row["gene_total"] == 201
    assert _row["gene_count"] == 26

    assert "bg_count" in results.columns
    assert "bg_total" in results.columns
    assert "bg_ratio" in results.columns
    assert _row["bg_ratio"] == 299 / 11840.0
    assert _row["bg_total"] == 11840
    assert _row["bg_count"] == 299

    assert "rich_factor" in results.columns
    assert _row["rich_factor"] == 26 / 299.0

    assert "fold_enrich" in results.columns
    assert _row["fold_enrich"] == (26 / 201.0) / (299 / 11840.0)

    assert "term_desc" in results.columns
    assert _row["term_desc"] == "spindle"
    assert "pvalue" in results.columns
    assert f"{_row.pvalue:.6e}" == "6.490593e-12"
    assert "pvalue_adjust" in results.columns
    assert f"{_row.pvalue_adjust:.6e}" == "1.908234e-09"
    assert "qvalue" in results.columns
    assert f"{_row.qvalue:.6e}" == "1.735380e-09"

    assert "gene_id" in results.columns
    assert _row["gene_id"].startswith("CDCA8/CDC20/KIF23/CENPE/ASPM/DLGAP5/")


def test_run_enrich_enrichgo_example_genes_in_ensembl_id():
    pandas2ri.deactivate()

    # example taken from here:
    # https://yulab-smu.top/biomedical-knowledge-mining-book/clusterprofiler-go.html
    # but I use ensembl ids here
    # pvalues should be the same as in the previous test with entrez IDs, but
    # they are slightly different here because one gene did not match in the
    # "gene" group
    gene_data = data(dose).fetch("geneList")["geneList"]
    gene_values_orig = np.array(gene_data)
    gene_names_orig = np.array(gene_data.names)

    # get gene names (Entrez IDs)
    gene_names = clusterProfiler.bitr(
        gene_names_orig.tolist(),
        fromType="ENTREZID",
        toType="ENSEMBL",
        OrgDb="org.Hs.eg.db",
        drop=True,
    )
    gene_names_entrez = np.array(gene_names[0])
    _, unique_entrez_idx = np.unique(gene_names_entrez, return_index=True)

    gene_names_ensembl = np.array(gene_names[1])
    _, unique_ensembl_idx = np.unique(gene_names_ensembl, return_index=True)

    idx = np.intersect1d(unique_entrez_idx, unique_ensembl_idx)
    gene_names_entrez = gene_names_entrez[idx]

    gene_names = gene_names_ensembl[idx]
    assert np.unique(gene_names).shape[0] == gene_names.shape[0]

    gene_values = gene_values_orig[np.isin(gene_names_orig, gene_names_entrez)]
    gene = gene_names[np.abs(gene_values) > 2]
    assert gene.shape[0] == 206  # one less than entrez id test
    assert np.unique(gene).shape[0] == gene.shape[0]

    # create a "gene partition"
    gene_partition = np.zeros(gene_names.shape[0])
    gene_partition[np.isin(gene_names, gene)] = 1
    np.testing.assert_array_equal(np.unique(gene_partition), np.array([0, 1]))
    assert gene_partition[gene_partition == 1].shape[0] == gene.shape[0]

    # run
    results = run_enrich(
        gene_names,
        "ENSEMBL",
        gene_partition,
        "enrichGO",
        "CC",
        pvalue_cutoff=0.01,
        qvalue_cutoff=0.05,
    )
    assert results is not None
    # assert len(all_results) == 1

    # results = all_results[0]
    assert results.shape[0] == 23

    # partition information
    unique_n_clusters = results["n_clusters"].unique()
    assert unique_n_clusters.shape[0] == 1
    assert 2 in unique_n_clusters

    unique_cluster_id = results["cluster_id"].unique()
    # only one cluster (C1) has significant results
    assert unique_cluster_id.shape[0] == 1
    assert "C1" in unique_cluster_id

    # check one row, with go term id: GO:0005819
    _row = results[results["term_id"] == "GO:0005819"]
    assert _row.shape[0] == 1
    _row = _row.iloc[0]
    assert _row["gene_ratio"] == 26 / 201
    assert _row["gene_total"] == 201
    assert _row["gene_count"] == 26

    assert _row["bg_ratio"] == 299 / 11824
    assert _row["bg_total"] == 11824
    assert _row["bg_count"] == 299

    assert _row["rich_factor"] == 26 / 299.0
    assert _row["fold_enrich"] == (26 / 201.0) / (299 / 11824.0)

    assert _row["term_desc"] == "spindle"
    assert f"{_row.pvalue:.6e}" == "6.686603e-12"
    assert f"{_row.pvalue_adjust:.6e}" == "1.965861e-09"
    assert f"{_row.qvalue:.6e}" == "1.787787e-09"

    assert _row["gene_id"].startswith("CDCA8/CDC20/KIF23/CENPE/ASPM/DLGAP5/")


def test_run_enrich_enrichgo_example_genes_in_symbol_ids():
    pandas2ri.deactivate()

    # example taken from here:
    # https://yulab-smu.top/biomedical-knowledge-mining-book/clusterprofiler-go.html
    # but I use symbol ids here
    # in this example, the mapping from entrez id to symbol is perfect, so the
    # same pvalues, gene ratios, etc are expected as in the entrez test above
    gene_data = data(dose).fetch("geneList")["geneList"]
    gene_values_orig = np.array(gene_data)
    gene_names_orig = np.array(gene_data.names)

    # get gene names (Entrez IDs)
    gene_names = clusterProfiler.bitr(
        gene_names_orig.tolist(),
        fromType="ENTREZID",
        toType="SYMBOL",
        OrgDb="org.Hs.eg.db",
        drop=True,
    )
    gene_names_entrez = np.array(gene_names[0])
    _, unique_entrez_idx = np.unique(gene_names_entrez, return_index=True)

    gene_names_ensembl = np.array(gene_names[1])
    _, unique_ensembl_idx = np.unique(gene_names_ensembl, return_index=True)

    idx = np.intersect1d(unique_entrez_idx, unique_ensembl_idx)
    gene_names_entrez = gene_names_entrez[idx]

    gene_names = gene_names_ensembl[idx]
    assert np.unique(gene_names).shape[0] == gene_names.shape[0]

    gene_values = gene_values_orig[np.isin(gene_names_orig, gene_names_entrez)]
    gene = gene_names[np.abs(gene_values) > 2]
    assert gene.shape[0] == 207
    assert np.unique(gene).shape[0] == gene.shape[0]

    # create a "gene partition"
    gene_partition = np.zeros(gene_names.shape[0])
    gene_partition[np.isin(gene_names, gene)] = 1
    np.testing.assert_array_equal(np.unique(gene_partition), np.array([0, 1]))
    assert gene_partition[gene_partition == 1].shape[0] == gene.shape[0]

    # run
    results = run_enrich(
        gene_names,
        "SYMBOL",
        gene_partition,
        "enrichGO",
        "CC",
        pvalue_cutoff=0.01,
        qvalue_cutoff=0.05,
    )
    assert results is not None
    # assert len(all_results) == 1

    # results = all_results[0]
    assert results.shape[0] == 23

    # partition information
    unique_n_clusters = results["n_clusters"].unique()
    assert unique_n_clusters.shape[0] == 1
    assert 2 in unique_n_clusters

    unique_cluster_id = results["cluster_id"].unique()
    # only one cluster (C1) has significant results
    assert unique_cluster_id.shape[0] == 1
    assert "C1" in unique_cluster_id

    # check one row, with go term id: GO:0005819
    _row = results[results["term_id"] == "GO:0005819"]
    assert _row.shape[0] == 1
    _row = _row.iloc[0]
    assert _row["gene_ratio"] == 26 / 201
    assert _row["gene_total"] == 201
    assert _row["gene_count"] == 26

    assert _row["bg_ratio"] == 299 / 11840
    assert _row["bg_total"] == 11840
    assert _row["bg_count"] == 299

    assert _row["rich_factor"] == 26 / 299.0
    assert _row["fold_enrich"] == (26 / 201.0) / (299 / 11840.0)

    assert _row["term_desc"] == "spindle"
    assert f"{_row.pvalue:.6e}" == "6.490593e-12"
    assert f"{_row.pvalue_adjust:.6e}" == "1.908234e-09"
    assert f"{_row.qvalue:.6e}" == "1.735380e-09"

    assert _row["gene_id"].startswith("CDCA8/CDC20/KIF23/CENPE/ASPM/DLGAP5/")


def test_run_enrich_enrichgo_example_all_ontologies():
    pandas2ri.deactivate()

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
    results = run_enrich(
        gene_names,
        "ENTREZID",
        gene_partition,
        "enrichGO",
        "ALL",
        pvalue_cutoff=0.01,
        qvalue_cutoff=0.05,
    )
    assert results is not None
    # assert len(all_results) == 1

    # results = all_results[0]
    assert results.shape[0] == 126

    # partition information
    unique_n_clusters = results["n_clusters"].unique()
    assert unique_n_clusters.shape[0] == 1
    assert 2 in unique_n_clusters

    unique_cluster_id = results["cluster_id"].unique()
    # only one cluster (C1) has significant results
    assert unique_cluster_id.shape[0] == 1
    assert "C1" in unique_cluster_id

    # check one row, with go term id: GO:0140014
    _row = results[results["term_id"] == "GO:0140014"]
    assert _row.shape[0] == 1
    _row = _row.iloc[0]
    assert _row["gene_ratio"] == 33 / 195
    assert _row["gene_total"] == 195
    assert _row["gene_count"] == 33

    assert _row["bg_ratio"] == 241 / 11570
    assert _row["bg_total"] == 11570
    assert _row["bg_count"] == 241

    assert _row["rich_factor"] == 33 / 241.0
    assert _row["fold_enrich"] == (33 / 195.0) / (241 / 11570.0)

    assert _row["term_desc"] == "mitotic nuclear division"
    assert f"{_row.pvalue:.6e}" == "4.733990e-21"
    assert f"{_row.pvalue_adjust:.6e}" == "1.416410e-17"
    assert f"{_row.qvalue:.6e}" == "1.300103e-17"

    assert "ontology" in results.columns
    unique_ontolgoes = results["ontology"].unique()
    assert unique_ontolgoes.shape[0] == 3
    assert "BP" in unique_ontolgoes
    assert "CC" in unique_ontolgoes
    assert "MF" in unique_ontolgoes

    assert _row["gene_id"].startswith("CDCA8/CDC20/KIF23/CENPE/MYBL2/CCNB2/")


def test_run_enrich_enrichkegg_example():
    pandas2ri.deactivate()

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
    results = run_enrich(
        gene_names,
        "ENTREZID",
        gene_partition,
        "enrichKEGG",
        "hsa",
        pvalue_cutoff=0.01,
        qvalue_cutoff=0.05,
    )
    assert results is not None
    # assert len(all_results) == 1

    # results = all_results[0]
    assert results.shape[0] == 6

    # partition information
    assert "n_clusters" in results.columns
    unique_n_clusters = results["n_clusters"].unique()
    assert unique_n_clusters.shape[0] == 1
    assert 2 in unique_n_clusters

    assert "cluster_id" in results.columns
    unique_cluster_id = results["cluster_id"].unique()
    # only one cluster (C1) has significant results
    assert unique_cluster_id.shape[0] == 1
    assert "C1" in unique_cluster_id

    # check one row, with go term id: GO:0005819
    assert "term_id" in results.columns
    assert "gene_count" in results.columns
    assert "gene_total" in results.columns
    assert "gene_ratio" in results.columns
    _row = results[results["term_id"] == "hsa04110"]
    assert _row.shape[0] == 1
    _row = _row.iloc[0]
    assert _row["gene_ratio"] == 11 / 94
    assert _row["gene_total"] == 94
    assert _row["gene_count"] == 11

    assert "bg_count" in results.columns
    assert "bg_total" in results.columns
    assert "bg_ratio" in results.columns
    assert _row["bg_ratio"] == 115 / 5935
    assert _row["bg_total"] == 5935
    assert _row["bg_count"] == 115

    assert _row["rich_factor"] == 11 / 115.0
    assert _row["fold_enrich"] == (11 / 94.0) / (115 / 5935.0)

    assert "term_desc" in results.columns
    assert _row["term_desc"] == "Cell cycle"
    assert "pvalue" in results.columns
    assert f"{_row.pvalue:.6e}" == "1.596853e-06"
    assert "pvalue_adjust" in results.columns
    assert f"{_row.pvalue_adjust:.10f}" == "0.0002328239"
    assert "qvalue" in results.columns
    assert f"{_row.qvalue:.10f}" == "0.0002296865"

    assert "gene_id" in results.columns
    assert _row["gene_id"].startswith("8318/991/9133/890/983/4085/7272/1111/891/4174")


def test_run_enrich_enrichpathway_example():
    pandas2ri.deactivate()

    # example taken from here:
    # https://yulab-smu.top/biomedical-knowledge-mining-book/reactomepa.html
    gene_data = data(dose).fetch("geneList")["geneList"]

    # get gene names (Entrez IDs)
    gene_names = np.array(gene_data.names)
    assert np.unique(gene_names).shape[0] == gene_names.shape[0]  # unique

    gene_values = np.array(gene_data)
    gene = gene_names[np.abs(gene_values) > 1.5]
    assert gene.shape[0] == 513

    # create a "gene partition"
    gene_partition = np.zeros(gene_names.shape[0])
    gene_partition[np.isin(gene_names, gene)] = 1
    np.testing.assert_array_equal(np.unique(gene_partition), np.array([0, 1]))
    assert gene_partition[gene_partition == 1].shape[0] == gene.shape[0]

    # run
    results = run_enrich(
        gene_names,
        "ENTREZID",
        gene_partition,
        "enrichPathway",
        "human",
        pvalue_cutoff=0.05,
    )
    assert results is not None
    # assert len(all_results) == 1

    # results = all_results[0]
    assert results.shape[0] == 52

    # partition information
    assert "n_clusters" in results.columns
    unique_n_clusters = results["n_clusters"].unique()
    assert unique_n_clusters.shape[0] == 1
    assert 2 in unique_n_clusters

    assert "cluster_id" in results.columns
    unique_cluster_id = results["cluster_id"].unique()
    # only one cluster (C1) has significant results
    assert unique_cluster_id.shape[0] == 1
    assert "C1" in unique_cluster_id

    # check one row, with go term id: GO:0005819
    assert "term_id" in results.columns
    assert "gene_count" in results.columns
    assert "gene_total" in results.columns
    assert "gene_ratio" in results.columns
    _row = results[results["term_id"] == "R-HSA-69278"]
    assert _row.shape[0] == 1
    _row = _row.iloc[0]
    assert _row["gene_ratio"] == 55 / 328
    assert _row["gene_total"] == 328
    assert _row["gene_count"] == 55

    assert "bg_count" in results.columns
    assert "bg_total" in results.columns
    assert "bg_ratio" in results.columns
    assert _row["bg_ratio"] == 457 / 8063
    assert _row["bg_total"] == 8063
    assert _row["bg_count"] == 457

    assert _row["rich_factor"] == 55 / 457.0
    assert _row["fold_enrich"] == (55 / 328.0) / (457 / 8063.0)

    assert "term_desc" in results.columns
    assert _row["term_desc"] == "Cell Cycle, Mitotic"
    assert "pvalue" in results.columns
    assert f"{_row.pvalue:.6e}" == "1.411998e-13"
    assert "pvalue_adjust" in results.columns
    assert f"{_row.pvalue_adjust:.6e}" == "7.783842e-11"
    assert "qvalue" in results.columns
    assert f"{_row.qvalue:.6e}" == "7.012076e-11"

    assert "gene_id" in results.columns
    assert _row["gene_id"].startswith(
        "CDC45/CDCA8/MCM10/CDC20/FOXM1/KIF23/CENPE/MYBL2/CCNB2/NDC80/TOP2A/"
    )


def test_run_enrich_enrichpathway_example_no_enrichment():
    pandas2ri.deactivate()

    # example taken from here:
    # https://yulab-smu.top/biomedical-knowledge-mining-book/reactomepa.html
    gene_data = data(dose).fetch("geneList")["geneList"]

    # get gene names (Entrez IDs)
    gene_names = np.array(gene_data.names)
    assert np.unique(gene_names).shape[0] == gene_names.shape[0]  # unique

    gene_values = np.array(gene_data)
    gene_names = gene_names[np.abs(gene_values) <= 1.5]
    assert gene_names.shape[0] == 11982

    # take a random group
    gene_group_cluster = np.random.rand(gene_names.shape[0]) > 0.5
    gene = gene_names[gene_group_cluster]

    # create a "gene partition"
    gene_partition = np.zeros(gene_names.shape[0])
    gene_partition[np.isin(gene_names, gene)] = 1
    np.testing.assert_array_equal(np.unique(gene_partition), np.array([0, 1]))
    assert gene_partition[gene_partition == 1].shape[0] == gene.shape[0]

    # run
    results = run_enrich(
        gene_names,
        "ENTREZID",
        gene_partition,
        "enrichPathway",
        "human",
        pvalue_cutoff=0.01,
    )
    assert results is None


def test_run_enrich_enrichkegg_example_keytype_is_not_entrezid():
    pandas2ri.deactivate()

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
    with pytest.raises(ValueError) as e_info:
        run_enrich(
            gene_names,
            "ENSEMBL",
            gene_partition,
            "enrichKEGG",
            "hsa",
            pvalue_cutoff=0.01,
            qvalue_cutoff=0.05,
        )

    assert "Entrez" in str(e_info.value)
