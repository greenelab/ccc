"""
Tests the conf.py module.
"""
import os
import sys
import runpy
from unittest import mock

import pytest


def test_conf_module_load():
    from ccc import conf

    assert conf is not None
    assert conf.__file__ is not None


@mock.patch.dict(os.environ, {}, clear=True)
def test_conf_entries():
    from ccc import conf
    import importlib

    importlib.reload(conf)

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
    t = runpy.run_module("ccc.conf", run_name="__main__")
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
    from ccc import conf

    conf_filepath = Path(conf.__file__).resolve()
    assert conf_filepath is not None
    assert conf_filepath.exists()

    # check output
    r = subprocess.run(["python", conf_filepath], stdout=subprocess.PIPE)
    assert r is not None
    assert r.returncode == 0
    r_output = r.stdout.decode("utf-8")
    assert r_output is not None
    assert len(r_output) > 8, r_output
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
    assert len(r_output) > 8, r_output
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
    from ccc import conf
    import importlib

    importlib.reload(conf)

    assert conf.MANUSCRIPT is not None
    assert "CONTENT_DIR" in conf.MANUSCRIPT
    assert conf.MANUSCRIPT["CONTENT_DIR"] is not None
    assert conf.MANUSCRIPT["CONTENT_DIR"] != ""


@mock.patch.dict(os.environ, {"CM_N_JOBS": ""})
def test_conf_cm_n_jobs_is_empty_string():
    from ccc import conf
    import importlib

    importlib.reload(conf)

    assert conf.GENERAL is not None
    assert len(conf.GENERAL) > 0
    assert conf.GENERAL["N_JOBS"] is not None
    assert conf.GENERAL["N_JOBS"] > 0
    assert conf.GENERAL["N_JOBS_LOW"] is not None
    assert conf.GENERAL["N_JOBS_LOW"] > 0
