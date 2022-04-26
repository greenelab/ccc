"""
Contains functions to interact with the REST API of HumanBase (GIANT networks):
https://hb.flatironinstitute.org/
"""
from pathlib import Path
import tempfile
import json

import requests
import pandas as pd

URL_GENE_INFO = "https://hb.flatironinstitute.org/api/genes/"
URL_TISSUE_PREDICTION = "https://hb.flatironinstitute.org/api/integrations/relevant/"


def gene_exists(gene_entrez_id: str) -> bool:
    """
    Given a gene Entrez ID, it checks whether it exists in GIANT models.

    Returns:
        True if gene exists, False otherwise.
    """
    url = URL_GENE_INFO + str(gene_entrez_id)
    r = requests.get(url)

    if r.status_code != 200:
        return False

    data = r.json()
    return "entrez" in data and "standard_name" in data


def predict_tissue(gene_pair: tuple[str, str]) -> tuple[str, str]:
    """
    Given a gene pair (Entrez IDs) as a tuple, it predicts a tissue or cell type
    where they are specifically expressed.

    Args:
        gene_pair: a tuple with a gene pair (two elements as string) with
          Entrez IDs.

    Returns:
        A tuple with two elements: the tissue name and the URL to predict a
        network on this tissue.
    """
    for gene in gene_pair:
        if not gene_exists(gene):
            return None

    params = {"entrez": list(gene_pair)}
    r = requests.post(URL_TISSUE_PREDICTION, json=params)
    data = r.json()

    # check if top tissue is brenda
    # looks like GIANT only considers BRENDA terms when predicting a tissue
    top_id = 0
    while data[top_id]["context"]["term"]["database"]["name"] != "BRENDA Ontology":
        top_id += 1

    return data[top_id]["slug"], data[top_id]["url"]


def rank_genes(
    all_genes: set[str], edges: pd.DataFrame, query_gene_symbols: tuple[str, str]
) -> pd.Series:
    """
    This function was coded following the HumanBase ranking of genes in
    networks. It takes all the genes and their edges in a network and ranks them
    according to how much connected they are to the query gene pair.

    Args:
        all_genes: a set with all genes in the network (gene symbols).
        edges: a dataframe with three columns: gene0 (str, gene symbol), gene1
          (str, gene symbol) and weight (float).
        query_gene_symbols: a tuple with a gene pair (gene symbols) originally
          used to obtain the network.

    Returns:
        A series with gene symbols in index and the ranks (int) as values.
        Genes with a lower rank are more important for the network because they
        are more connected to the gene pair (query_gene_symbols).
    """
    genes_query_degrees = {}
    genes_degrees = {}

    for g in all_genes:
        # connections to query genes
        g_query_genes = edges[
            ((edges["gene1"] == g) & (edges["gene2"].isin(query_gene_symbols)))
            | ((edges["gene2"] == g) & (edges["gene1"].isin(query_gene_symbols)))
        ]

        g_query_degree = g_query_genes["weight"].sum() / g_query_genes.shape[0]

        # connections to all genes
        g_all_genes = edges[(edges["gene1"] == g) | (edges["gene2"] == g)]

        g_degree = g_all_genes["weight"].sum() / g_all_genes.shape[0]

        # save
        genes_query_degrees[g] = g_query_degree
        genes_degrees[g] = g_degree

    # no degree correction (actually, genes_degrees is not used here, following
    # the default behavior of GIANT)
    gene_ranks = [
        (gene, idx)
        for idx, (gene, weight) in enumerate(
            sorted(genes_query_degrees.items(), key=lambda item: -item[1])
        )
    ]

    return (
        pd.DataFrame(gene_ranks)
        .set_index(0)
        .squeeze()
        .rename("rank")
        .rename_axis("gene")
    )


def get_network(
    gene_entrezids: tuple[str, str] = None,
    gene_symbols: tuple[str, str] = None,
    gene_ids_mappings: pd.DataFrame = None,
    max_genes: int = 15,
    tissue: tuple[str, str] = None,
) -> tuple[pd.DataFrame, str, float]:
    """
    Given a gene pair (either with Entrez IDs or symbols), it predicts a
    tissue-specific network using GIANT. If tissue is given, then it predict the
    network in that tissue, otherwise it autoselects the tissue/cell type (see
    predict_tissue).

    Args:
        gene_entrezids: a gene pair with Entrez IDs. It can be None, in that case
            gene_symbols has to be provided.
        gene_symbols: a gene pair with symbols. It can be None, in that case
            gene_entrezids has to be provided.
        gene_ids_mappings: a dataframe with gene IDs mappings with two columns:
            SYMBOL and ENTREZID.
        max_genes: maximum number of genes to be included in the network (it
            does not include query genes). All genes are ranked (see rank_genes
            function) and the top ones are taken using this parameter. Default to 15
            (same as in GIANT).
        tissue: if None, autodetect tissue; otherwise, it has to be a tuple with
            the tissue name in the first element and the GIANT URL to query in the
            second element. For example, for blood:
                ("blood", "http://hb.flatironinstitute.org/api/integrations/blood/")

    Returns:
        A tuple with three elements about the network: the network itself as a
        dataframe (with columns gene1, gene2 (both gene symbols) and weight);
        the name of the predicted tissue; the minimum cut suggested by GIANT.
    """
    if gene_entrezids is None and gene_symbols is None:
        raise ValueError("No arguments provided")

    if gene_ids_mappings is None:
        raise ValueError("Gene mappings must be provided")

    if not hasattr(gene_ids_mappings, "columns"):
        raise ValueError("gene_ids_mappings has to be a DataFrame")

    if (
        "SYMBOL" not in gene_ids_mappings.columns
        or "ENTREZID" not in gene_ids_mappings.columns
    ):
        raise ValueError("gene_ids_mappings must have columns SYMBOL and ENTREZID")

    # create dictionaries for gene ids mappings
    gene_symbol_to_entrezid = gene_ids_mappings.set_index("SYMBOL").squeeze().to_dict()
    gene_entrezid_to_symbol = (
        gene_ids_mappings.set_index("ENTREZID").squeeze().to_dict()
    )

    if gene_entrezids is not None:
        if (
            gene_entrezids[0] not in gene_entrezid_to_symbol
            or gene_entrezids[1] not in gene_entrezid_to_symbol
        ):
            return None
        gene_symbols = (
            gene_entrezid_to_symbol[gene_entrezids[0]],
            gene_entrezid_to_symbol[gene_entrezids[1]],
        )
    else:
        if (
            gene_symbols[0] not in gene_symbol_to_entrezid
            or gene_symbols[1] not in gene_symbol_to_entrezid
        ):
            return None
        gene_entrezids = (
            gene_symbol_to_entrezid[gene_symbols[0]],
            gene_symbol_to_entrezid[gene_symbols[1]],
        )

    if tissue is not None and len(tissue) == 2:
        tissue_prediction = list(tissue)
    elif tissue is None:
        tissue_prediction = predict_tissue(gene_entrezids)
        if tissue_prediction is None:
            return None
    else:
        raise ValueError("Invalid tissue value")

    # predict a tissue-specific network
    url = tissue_prediction[1] + "network/"
    params = [("entrez", gene_entrezids[0]), ("entrez", gene_entrezids[1])]
    r = requests.get(url, params)
    data = r.json()

    # mincut will be used to filter out genes and keep only those with a weight
    # larger than mincut
    mincut = data["mincut"]

    # save the network to json files
    temp_dir = Path(tempfile.mkdtemp(prefix="giant-"))
    genes_json_file = temp_dir / "genes.json"
    edges_json_file = temp_dir / "edges.json"
    with open(genes_json_file, "w") as gf, open(edges_json_file, "w") as ef:
        json.dump(data["genes"], gf)
        json.dump(data["edges"], ef)

    # load network as panda objects
    genes = pd.read_json(genes_json_file)["standard_name"]
    edges = pd.read_json(edges_json_file)[["source", "target", "weight"]]

    df = edges.join(genes.rename("gene1"), on="source", how="left").join(
        genes.rename("gene2"), on="target", how="left"
    )[["gene1", "gene2", "weight"]]

    # rank genes
    all_genes = set(df["gene1"]).union(set(df["gene2"]))
    if gene_symbols[0] not in all_genes or gene_symbols[1] not in all_genes:
        return None

    all_genes.remove(gene_symbols[0])
    all_genes.remove(gene_symbols[1])

    genes_ranks = rank_genes(all_genes, df, gene_symbols)
    top_genes = set(genes_ranks.head(max_genes).index)
    # add query genes to the ranked list of genes
    top_genes.update(gene_symbols)
    df = df[(df["gene1"].isin(top_genes)) & (df["gene2"].isin(top_genes))]

    return (
        df[df["weight"] > mincut].reset_index(drop=True),
        tissue_prediction[0],
        mincut,
    )
