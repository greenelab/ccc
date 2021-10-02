"""
Gets user settings (from settings.py module) and create the final configuration values.
All the rest of the code reads configuration values from this module.
This file IS NOT intended to be modified by the user.
"""
import os
import tempfile
from pathlib import Path

from clustermatch import settings

#
# General file structure
#
ROOT_DIR = os.environ.get("CM_ROOT_DIR")
if ROOT_DIR is None and hasattr(settings, "ROOT_DIR"):
    ROOT_DIR = settings.ROOT_DIR

if ROOT_DIR is None:
    ROOT_DIR = str(Path(tempfile.gettempdir(), "cm_gene_expr").resolve())

# DATA_DIR stores input data
DATA_DIR = Path(ROOT_DIR, "data").resolve()

# RESULTS_DIR stores newly generated data
RESULTS_DIR = Path(ROOT_DIR, "results").resolve()

#
# General
#
GENERAL = {}

GENERAL["LOG_CONFIG_FILE"] = Path(
    Path(__file__).resolve().parent, "log_config.yaml"
).resolve()

# CPU usage
options = [
    os.environ.get("CM_N_JOBS"),
    getattr(settings, "N_JOBS", None),
    1,
]
GENERAL["N_JOBS"] = next(int(opt) for opt in options if opt is not None)

options = [
    os.environ.get("CM_N_JOBS_LOW"),
    getattr(settings, "N_JOBS_LOW", None),
    GENERAL["N_JOBS"],
]
GENERAL["N_JOBS_LOW"] = next(int(opt) for opt in options if opt is not None)

#
# Manuscript
#
MANUSCRIPT = {}
MANUSCRIPT["BASE_DIR"] = os.environ.get("CM_MANUSCRIPT_DIR", settings.MANUSCRIPT_DIR)
if MANUSCRIPT["BASE_DIR"] is not None:
    # these paths are specific to manubot
    MANUSCRIPT["CONTENT_DIR"] = Path(MANUSCRIPT["BASE_DIR"], "content").resolve()
    MANUSCRIPT["FIGURES_DIR"] = Path(MANUSCRIPT["CONTENT_DIR"], "images").resolve()

#
# General configurations
#
GENE_SELECTION_STRATEGY_NAME_PATTERN = r"(?P<gene_sel_strategy>[0-9a-z_]+)"
GTEX_TISSUE_NAME_PATTERN = r"(?P<tissue>[0-9a-z_]+)"
CORRELATION_METHOD_PATTERN = r"(?P<corr_method>[0-9a-z_]+)"
CLUSTERING_METHOD_PATTERN = r"(?P<clust_method>[0-9a-zA-Z]+)"
ENRICH_FUNCTION_PATTERN = r"(?P<enrich_func>[A-Za-z_]+)"
ENRICH_FUNCTION_PARAMS_PATTERN = r"(?P<enrich_params>[0-9A-Za-z_]+)"

#
# GTEx
#
GTEX = {}

# Input data
GTEX["DATA_DIR"] = Path(DATA_DIR, "gtex_v8").resolve()

GTEX["SAMPLE_ATTRS_FILE"] = Path(
    GTEX["DATA_DIR"], "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
).resolve()
GTEX["DATA_TPM_GCT_FILE"] = Path(
    GTEX["DATA_DIR"], "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
).resolve()
GTEX["N_TISSUES"] = 54

# Results
GTEX["RESULTS_DIR"] = Path(RESULTS_DIR, "gtex_v8").resolve()

## Gene selection
GTEX["GENE_SELECTION_DIR"] = Path(GTEX["RESULTS_DIR"], "gene_selection").resolve()

## Similarity matrices
GTEX["SIMILARITY_MATRICES_DIR"] = Path(
    GTEX["RESULTS_DIR"], "similarity_matrices"
).resolve()
GTEX[
    "SIMILARITY_MATRIX_FILENAME_TEMPLATE"
] = "gtex_v8_data_{tissue}-{gene_sel_strategy}-{corr_method}.pkl"

## Clustering
GTEX["CLUSTERING_DIR"] = Path(GTEX["RESULTS_DIR"], "clustering").resolve()
GTEX["CLUSTERING_FILENAME_TEMPLATE"] = (
    GTEX["SIMILARITY_MATRIX_FILENAME_TEMPLATE"][:-4] + "-{clust_method}.pkl"
)
GTEX["CLUSTERING_FILENAME_PATTERN"] = GTEX["CLUSTERING_FILENAME_TEMPLATE"].format(
    tissue=GTEX_TISSUE_NAME_PATTERN,
    gene_sel_strategy=GENE_SELECTION_STRATEGY_NAME_PATTERN,
    corr_method=CORRELATION_METHOD_PATTERN,
    clust_method=CLUSTERING_METHOD_PATTERN,
)
GTEX["CLUSTERING_COMBINED_FILE"] = (
    GTEX["CLUSTERING_DIR"] / "gtex_v8_data-clustering.pkl"
)

## Gene set enrichment analysis
GTEX["GENE_ENRICHMENT_DIR"] = Path(GTEX["RESULTS_DIR"], "gene_set_enrichment").resolve()
GTEX["GENE_ENRICHMENT_FILENAME_TEMPLATE"] = (
    GTEX["CLUSTERING_FILENAME_TEMPLATE"][:-4] + "-{enrich_func}-{enrich_params}.pkl"
)
GTEX["GENE_ENRICHMENT_FILENAME_PATTERN"] = GTEX[
    "GENE_ENRICHMENT_FILENAME_TEMPLATE"
].format(
    tissue=GTEX_TISSUE_NAME_PATTERN,
    gene_sel_strategy=GENE_SELECTION_STRATEGY_NAME_PATTERN,
    corr_method=CORRELATION_METHOD_PATTERN,
    clust_method=CLUSTERING_METHOD_PATTERN,
    enrich_func=ENRICH_FUNCTION_PATTERN,
    enrich_params=ENRICH_FUNCTION_PARAMS_PATTERN,
)
GTEX["GENE_ENRICHMENT_COMBINED_FILE"] = (
    GTEX["GENE_ENRICHMENT_DIR"] / "gtex_v8_data-gene_set_enrichment.pkl"
)

#
# recount2 (from MultiPLIER)
#
RECOUNT2 = {}

# Input data
RECOUNT2["DATA_DIR"] = Path(DATA_DIR, "recount2").resolve()

RECOUNT2["DATA_RDS_FILE"] = Path(
    RECOUNT2["DATA_DIR"], "recount_data_prep_PLIER.RDS"
).resolve()
RECOUNT2["DATA_FILE"] = Path(
    RECOUNT2["DATA_DIR"], "recount_data_prep_PLIER.pkl"
).resolve()

# Results
RECOUNT2["RESULTS_DIR"] = Path(RESULTS_DIR, "recount2").resolve()

## Similarity matrices
RECOUNT2["SIMILARITY_MATRICES_DIR"] = Path(
    RECOUNT2["RESULTS_DIR"], "similarity_matrices"
).resolve()
RECOUNT2[
    "SIMILARITY_MATRIX_FILENAME_TEMPLATE"
] = "recount_data_prep_PLIER-{corr_method}.pkl"

## Clustering
RECOUNT2["CLUSTERING_DIR"] = Path(RECOUNT2["RESULTS_DIR"], "clustering").resolve()
RECOUNT2["CLUSTERING_FILENAME_TEMPLATE"] = (
    RECOUNT2["SIMILARITY_MATRIX_FILENAME_TEMPLATE"][:-4] + "-{clust_method}.pkl"
)
RECOUNT2["CLUSTERING_FILENAME_PATTERN"] = RECOUNT2[
    "CLUSTERING_FILENAME_TEMPLATE"
].format(
    corr_method=r"(?P<corr_method>[0-9a-z_]+)",
    clust_method=r"(?P<clust_method>[0-9a-zA-Z]+)",
)
RECOUNT2["CLUSTERING_COMBINED_FILE"] = (
    RECOUNT2["CLUSTERING_DIR"] / "recount_data_prep_PLIER-clustering.pkl"
)

## Gene set enrichment analysis
RECOUNT2["GENE_ENRICHMENT_DIR"] = Path(
    RECOUNT2["RESULTS_DIR"], "gene_set_enrichment"
).resolve()
RECOUNT2["GENE_ENRICHMENT_FILENAME_TEMPLATE"] = (
    RECOUNT2["CLUSTERING_FILENAME_TEMPLATE"][:-4] + "-{enrich_func}-{enrich_params}.pkl"
)
RECOUNT2["GENE_ENRICHMENT_FILENAME_PATTERN"] = RECOUNT2[
    "GENE_ENRICHMENT_FILENAME_TEMPLATE"
].format(
    corr_method=CORRELATION_METHOD_PATTERN,
    clust_method=CLUSTERING_METHOD_PATTERN,
    enrich_func=ENRICH_FUNCTION_PATTERN,
    enrich_params=ENRICH_FUNCTION_PARAMS_PATTERN,
)
RECOUNT2["GENE_ENRICHMENT_COMBINED_FILE"] = (
    RECOUNT2["GENE_ENRICHMENT_DIR"] / "recount_data_prep_PLIER-gene_set_enrichment.pkl"
)


if __name__ == "__main__":
    # if this script is run, then it exports the configuration as environment
    # variables (for bash/R, etc)
    from pathlib import PurePath

    def print_conf(conf_dict):
        for var_name, var_value in conf_dict.items():
            if var_value is None:
                continue

            if isinstance(var_value, (str, int, PurePath)):
                new_var_name = f"CM_{var_name}"
                print(f'export {new_var_name}="{str(var_value)}"')
                yield new_var_name
            elif isinstance(var_value, dict):
                new_dict = {f"{var_name}_{k}": v for k, v in var_value.items()}
                for x in print_conf(new_dict):
                    yield x
            else:
                raise ValueError(f"Configuration type not understood: {var_name}")

    local_variables = {
        k: v for k, v in locals().items() if not k.startswith("__") and k == k.upper()
    }

    print_vars = list(print_conf(local_variables))
