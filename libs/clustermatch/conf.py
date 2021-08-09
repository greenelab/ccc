"""
Gets user settings (from settings.py module) and create the final configuration values.
All the rest of the code reads configuration values from this module.
This file IS NOT intended to be modified by the user.
"""
import os
import tempfile
from multiprocessing import cpu_count
from pathlib import Path

import settings

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
    int(cpu_count() / 2),
]
GENERAL["N_JOBS"] = next(int(opt) for opt in options if opt is not None)

options = [
    os.environ.get("CM_N_JOBS_LOW"),
    getattr(settings, "N_JOBS_LOW", None),
    GENERAL["N_JOBS"],
]
GENERAL["N_JOBS_LOW"] = next(int(opt) for opt in options if opt is not None)

#
# Results
#
RESULTS = {}
RESULTS["BASE_DIR"] = RESULTS_DIR

#
# Manuscript
#
MANUSCRIPT = {}
MANUSCRIPT["BASE_DIR"] = os.environ.get(
    "CM_MANUSCRIPT_DIR", settings.MANUSCRIPT_DIR
)
if MANUSCRIPT["BASE_DIR"] is not None:
    # these paths are specific to manubot
    MANUSCRIPT["CONTENT_DIR"] = Path(MANUSCRIPT["BASE_DIR"], "content").resolve()
    MANUSCRIPT["FIGURES_DIR"] = Path(MANUSCRIPT["CONTENT_DIR"], "images").resolve()


if __name__ == "__main__":
    # if this script is run, then it exports the configuration as environment
    # variables (for bash/R, etc)
    from pathlib import PurePath

    def print_conf(conf_dict):
        for var_name, var_value in conf_dict.items():
            if var_value is None:
                continue

            if isinstance(var_value, (str, int, PurePath)):
                print(f'export CM_{var_name}="{str(var_value)}"')
            elif isinstance(var_value, dict):
                new_dict = {f"{var_name}_{k}": v for k, v in var_value.items()}
                print_conf(new_dict)
            else:
                raise ValueError(f"Configuration type not understood: {var_name}")

    local_variables = {
        k: v for k, v in locals().items() if not k.startswith("__") and k == k.upper()
    }

    print_conf(local_variables)
