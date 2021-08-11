"""
Tests the conf.py module.
"""
import sys
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

    assert conf.RESULTS["BASE_DIR"] is not None
    assert conf.RESULTS["BASE_DIR"] != ""

    assert conf.MANUSCRIPT is not None


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

    r = subprocess.run(
        f"eval `python {conf_filepath}` && echo $CM_ROOT_DIR",
        shell=True,
        stdout=subprocess.PIPE,
    )
    assert r is not None
    assert r.returncode == 0
    r_output = r.stdout.decode("utf-8")
    assert r_output is not None
    assert len(r_output) > 10
    assert r_output.startswith("/")
