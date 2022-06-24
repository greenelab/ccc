"""
Gets user settings (from settings.py module) and create the final configuration values.
All the rest of the code reads configuration values from this module.
This file IS NOT intended to be modified by the user.
"""
import os
import tempfile
from pathlib import Path

from ccc import settings

#
# General file structure
#
ROOT_DIR = os.environ.get("CM_ROOT_DIR")
if ROOT_DIR is None and hasattr(settings, "ROOT_DIR"):
    ROOT_DIR = settings.ROOT_DIR

if ROOT_DIR is None:
    ROOT_DIR = str(Path(tempfile.gettempdir(), "ccc_gene_expr").resolve())

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
    m if (m := os.environ.get("CM_N_JOBS")) is not None and m.strip() != "" else None,
    getattr(settings, "N_JOBS", None),
    1,
]
GENERAL["N_JOBS"] = next(int(opt) for opt in options if opt is not None)

options = [
    m
    if (m := os.environ.get("CM_N_JOBS_LOW")) is not None and m.strip() != ""
    else None,
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
    # convert to Path
    MANUSCRIPT["BASE_DIR"] = Path(MANUSCRIPT["BASE_DIR"]).resolve()

    # these paths are specific to manubot
    MANUSCRIPT["CONTENT_DIR"] = MANUSCRIPT["BASE_DIR"] / "content"
    MANUSCRIPT["FIGURES_DIR"] = MANUSCRIPT["CONTENT_DIR"] / "images"


#
# GTEx
#
GTEX = {}

# Input data
GTEX["DATA_DIR"] = Path(DATA_DIR, "gtex_v8").resolve()

GTEX["SAMPLE_ATTRS_FILE"] = Path(
    GTEX["DATA_DIR"], "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
).resolve()
GTEX["SUBJECTS_ATTRS_FILE"] = Path(
    GTEX["DATA_DIR"], "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"
).resolve()
GTEX["DATA_TPM_GCT_FILE"] = Path(
    GTEX["DATA_DIR"], "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
).resolve()
GTEX["N_TISSUES"] = 54

# Results
GTEX["RESULTS_DIR"] = Path(RESULTS_DIR, "gtex_v8").resolve()

GTEX["GENE_SELECTION_DIR"] = Path(GTEX["RESULTS_DIR"], "gene_selection").resolve()

GTEX["SIMILARITY_MATRICES_DIR"] = Path(
    GTEX["RESULTS_DIR"], "similarity_matrices"
).resolve()
GTEX[
    "SIMILARITY_MATRIX_FILENAME_TEMPLATE"
] = "gtex_v8_data_{tissue}-{gene_sel_strategy}-{corr_method}.pkl"
GTEX["GENE_PAIR_INTERSECTIONS"] = Path(
    GTEX["RESULTS_DIR"], "gene_pair_intersections"
).resolve()


#
# GIANT networks
#
GIANT = {}

# Results
GIANT["RESULTS_DIR"] = Path(RESULTS_DIR, "giant").resolve()


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
