"""
Tests the log.py module.
"""


def test_log_module_load():
    from ccc import log

    assert log is not None
    assert log.__file__ is not None


def test_log_get_logger():
    from ccc import log

    logger = log.get_logger("testing")
    assert logger is not None
    assert hasattr(logger, "info")
    assert hasattr(logger, "debug")
    assert hasattr(logger, "error")

    logger.info("test")
    logger.warning("test warn")
