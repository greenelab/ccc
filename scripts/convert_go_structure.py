from pathlib import Path
import re

import pandas as pd
from tqdm import tqdm


GENE_SELECTION_STRATEGY_NAME_PATTERN = r"(?P<gene_sel_strategy>[0-9a-z_]+)"
GTEX_TISSUE_NAME_PATTERN = r"(?P<tissue>[0-9a-z_]+)"
CORRELATION_METHOD_PATTERN = r"(?P<corr_method>[0-9a-z_]+)"
CLUSTERING_METHOD_PATTERN = r"(?P<clust_method>[0-9a-zA-Z]+)"
ENRICH_FUNCTION_PATTERN = r"(?P<enrich_func>[A-Za-z_]+)"
ENRICH_FUNCTION_PARAMS_PATTERN = r"(?P<enrich_params>[0-9A-Za-z_]+)"

# GTEx v8
#ENRICHMENT_FILE_TEMPLATE = "gtex_v8_data_{tissue}-{gene_sel_strategy}-{corr_method}-{clust_method}-{enrich_func}-{enrich_params}.pkl"

# recount2
ENRICHMENT_FILE_TEMPLATE = "recount_data_prep_PLIER-{corr_method}-{clust_method}-{enrich_func}-{enrich_params}.pkl"

ENRICHMENT_FILE_PATTERN = ENRICHMENT_FILE_TEMPLATE.format(
    #tissue=GTEX_TISSUE_NAME_PATTERN,
    #gene_sel_strategy=GENE_SELECTION_STRATEGY_NAME_PATTERN,
    corr_method=CORRELATION_METHOD_PATTERN,
    clust_method=CLUSTERING_METHOD_PATTERN,
    enrich_func=ENRICH_FUNCTION_PATTERN,
    enrich_params=ENRICH_FUNCTION_PARAMS_PATTERN,
)
filename_pattern = re.compile(ENRICHMENT_FILE_PATTERN)

def convert_file(filepath, dataset_name):
    data = pd.read_pickle(filepath)

    if "gtex" in dataset_name:
        data = data.drop(columns=["clustering_id"])
    
    data = data.rename(columns={
        "Cluster": "cluster_id",
        "ID": "term_id",
        "go_term_id": "term_id",
        "Description": "term_desc",
        "go_term_desc": "term_desc",
        "GeneRatio": "gene_ratio",
        "BgRatio": "bg_ratio",
        "p.adjust": "pvalue_adjust",
        "fdr_per_partition": "pvalue_adjust",
        "geneID": "gene_id",
        "Count": "gene_count",
        "clustering_n_clusters": "n_clusters",
    })
    
    filename = filepath.name

    match = re.search(filename_pattern, filename)

    data = data.assign(ontology=match.group("enrich_params").split("_")[0])
    
    data = data.assign(
        gene_total=data["gene_ratio"].apply(lambda x: int(x.split("/")[1])),
        bg_count=data["bg_ratio"].apply(lambda x: int(x.split("/")[0])),
        bg_total=data["bg_ratio"].apply(lambda x: int(x.split("/")[1])),
    )
    
    # convert ratios to numbers
    data["gene_ratio"] = data["gene_count"].div(data["gene_total"])
    data["bg_ratio"] = data["bg_count"].div(data["bg_total"])

    data["rich_factor"] = data["gene_count"].div(data["bg_count"])
    data["fold_enrich"] = data["gene_ratio"].div(data["bg_ratio"])

    return data


	
if __name__ == "__main__":
    #dataset_name = "gtex_v8"
    dataset_name = "recount2"

    input_files = list(Path(f"base/results/{dataset_name}/gene_set_enrichment/tmp_enrichGO_old/").iterdir())
    # keep only full results
    input_files = [
        x
        for x in input_files
        if (m := re.search(filename_pattern, x.name)) is not None
        and m.group("enrich_params").endswith("_full")
    ]

    OUTPUT_DIR = Path(f"base/results/{dataset_name}/gene_set_enrichment/new_enrichGO")
    OUTPUT_DIR.mkdir(exist_ok=True)

    for f in tqdm(input_files):
        data = convert_file(f, dataset_name)
        data.to_pickle(OUTPUT_DIR / f.name)

