"""
Tests the conf.py module.
"""
import os
import sys
import runpy
from unittest import mock

import pytest


def test_conf_module_load():
    from clustermatch import conf

    assert conf is not None
    assert conf.__file__ is not None


def test_conf_entries():
    from clustermatch import conf

    assert conf.ROOT_DIR is not None
    assert conf.ROOT_DIR != ""

    assert conf.DATA_DIR is not None
    assert conf.DATA_DIR != ""

    assert conf.RESULTS_DIR is not None
    assert conf.RESULTS_DIR != ""

    assert conf.GENERAL is not None
    assert len(conf.GENERAL) > 0
    assert conf.GENERAL["N_JOBS"] is not None
    assert conf.GENERAL["N_JOBS"] > 0
    assert conf.GENERAL["N_JOBS_LOW"] is not None
    assert conf.GENERAL["N_JOBS_LOW"] > 0

    assert conf.MANUSCRIPT is not None
    assert "CONTENT_DIR" not in conf.MANUSCRIPT


def test_conf_main():
    t = runpy.run_module("clustermatch.conf", run_name="__main__")
    assert t is not None
    assert "print_vars" in t
    assert "CM_ROOT_DIR" in t["print_vars"]
    assert "CM_RESULTS_DIR" in t["print_vars"]
    assert "CM_GENERAL_N_JOBS" in t["print_vars"]


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="exporting variables is only supported in non-Windows platforms",
)
def test_conf_export_variables():
    from pathlib import Path
    import subprocess
    from clustermatch import conf

    conf_filepath = Path(conf.__file__).resolve()
    assert conf_filepath is not None
    assert conf_filepath.exists()

    # check output
    r = subprocess.run(["python", conf_filepath], stdout=subprocess.PIPE)
    assert r is not None
    assert r.returncode == 0
    r_output = r.stdout.decode("utf-8")
    assert r_output is not None
    assert len(r_output) > 10
    assert r_output.count("export ") > 5

    # check variable
    r = subprocess.run(
        f"eval `python {conf_filepath}` && echo $CM_ROOT_DIR",
        shell=True,
        stdout=subprocess.PIPE,
    )
    assert r is not None
    assert r.returncode == 0
    r_output = r.stdout.decode("utf-8").strip()
    assert r_output is not None
    assert len(r_output) > 10
    assert r_output.startswith("/")

    # check dict variable
    r = subprocess.run(
        f"eval `python {conf_filepath}` && echo $CM_GENERAL_N_JOBS",
        shell=True,
        stdout=subprocess.PIPE,
    )
    assert r is not None
    assert r.returncode == 0
    r_output = r.stdout.decode("utf-8").strip()
    assert r_output is not None
    assert r_output.isdigit()
    assert int(r_output) > 0


@mock.patch.dict(os.environ, {"CM_MANUSCRIPT_DIR": "/tmp/some/dir"})
def test_conf_with_manuscript_dir():
    from clustermatch import conf
    import importlib

    importlib.reload(conf)

    assert conf.MANUSCRIPT is not None
    assert "CONTENT_DIR" in conf.MANUSCRIPT
    assert conf.MANUSCRIPT["CONTENT_DIR"] is not None
    assert conf.MANUSCRIPT["CONTENT_DIR"] != ""


def test_conf_gtex_clustering_filename_regex():
    import re
    from clustermatch import conf

    assert "CLUSTERING_FILENAME_PATTERN" in conf.GTEX
    pat = re.compile(conf.GTEX["CLUSTERING_FILENAME_PATTERN"])

    filename = "gtex_v8_data_adipose_subcutaneous-var_pc_log2-clustermatch_k2-SpectralClustering.pkl"
    m = re.search(pat, filename)
    assert m.group("tissue") == "adipose_subcutaneous"
    assert m.group("gene_sel_strategy") == "var_pc_log2"
    assert m.group("corr_method") == "clustermatch_k2"
    assert m.group("clust_method") == "SpectralClustering"

    filename = "gtex_v8_data_muscle_skeletal-var_raw-spearman_full-AgglomerativeClustering.pkl"
    m = re.search(pat, filename)
    assert m.group("tissue") == "muscle_skeletal"
    assert m.group("gene_sel_strategy") == "var_raw"
    assert m.group("corr_method") == "spearman_full"
    assert m.group("clust_method") == "AgglomerativeClustering"
