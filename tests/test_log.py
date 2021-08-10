"""
Tests the log.py module.
"""


def test_log_module_load():
    from clustermatch import log

    assert log is not None
    assert log.__file__ is not None


def test_log_get_logger():
    from clustermatch import log

    l = log.get_logger("testing")
    assert l is not None
    assert hasattr(l, "info")
    assert hasattr(l, "debug")
    assert hasattr(l, "error")

    l.info("test")
