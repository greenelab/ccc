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


def gene_exists(gene_entrez_id):
    url = URL_GENE_INFO + str(gene_entrez_id)
    r = requests.get(url)

    if r.status_code != 200:
        return False

    data = r.json()
    return "entrez" in data and "standard_name" in data


def predict_tissue(gene_pair_tuple):
    for gene in gene_pair_tuple:
        if not gene_exists(gene):
            return None

    params = {"entrez": list(gene_pair_tuple)}
    r = requests.post(URL_TISSUE_PREDICTION, json=params)
    data = r.json()

    # check if top tissue is brenda
    top_id = 0
    while data[top_id]["context"]["term"]["database"]["name"] != "BRENDA Ontology":
        top_id += 1

    return data[top_id]["slug"], data[top_id]["url"]


def rank_genes(all_genes, edges, query_gene_symbols):
    """
    This function was coded following the HumanBase ranking of genes in networks.
    EXPLAIN MORE
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

    # no degree correction
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
    gene_entrezids: tuple[str] = None,
    gene_symbols: tuple[str] = None,
    gene_ids_mappings: pd.DataFrame = None,
    max_genes=15,
    tissue=None,
):
    """
    TODO

    Args:
        gene_ids_mappings:
        tissue: if None, autodetect tissue; otherwise, it has to be a tuple with
          the tissue name in the first element and the GIANT URL to query in the
          second element. For example, for blood:
            ("blood", "http://hb.flatironinstitute.org/api/integrations/blood/")
    Returns:
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

    url = tissue_prediction[1] + "network/"
    params = [("entrez", gene_entrezids[0]), ("entrez", gene_entrezids[1])]
    r = requests.get(url, params)
    data = r.json()

    mincut = data["mincut"]
    # print(mincut)

    temp_dir = Path(tempfile.mkdtemp(prefix="giant-"))
    genes_json_file = temp_dir / "genes.json"
    edges_json_file = temp_dir / "edges.json"
    with open(genes_json_file, "w") as gf, open(edges_json_file, "w") as ef:
        json.dump(data["genes"], gf)
        json.dump(data["edges"], ef)

    genes = pd.read_json(genes_json_file)["standard_name"]
    edges = pd.read_json(edges_json_file)[["source", "target", "weight"]]

    df = edges.join(genes.rename("gene1"), on="source", how="left").join(
        genes.rename("gene2"), on="target", how="left"
    )[["gene1", "gene2", "weight"]]

    # prioritize genes
    all_genes = set(df["gene1"]).union(set(df["gene2"]))
    if gene_symbols[0] not in all_genes or gene_symbols[1] not in all_genes:
        return None

    all_genes.remove(gene_symbols[0])
    all_genes.remove(gene_symbols[1])

    genes_ranks = rank_genes(all_genes, df, gene_symbols)
    top_genes = set(genes_ranks.head(max_genes).index)
    top_genes.update(gene_symbols)
    df = df[(df["gene1"].isin(top_genes)) & (df["gene2"].isin(top_genes))]

    return (
        df[df["weight"] > mincut].reset_index(drop=True),
        tissue_prediction[0],
        mincut,
    )
