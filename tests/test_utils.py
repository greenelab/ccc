"""
Tests the utility_functions.py module.
"""
import sys
from unittest.mock import MagicMock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def reload_package(root_module):
    """
    Reloads all modules given a root module. It deletes loaded modules from sys.modules and reloads them again.
    This is helpful in scenarios where you want to test what happens if a certain module does not have access
    to other modules.

    Taken and adapted from here: https://stackoverflow.com/a/2918951/3120414
    """
    from importlib import import_module
    import types

    package_name = root_module.__name__

    # get a reference to each loaded module
    loaded_package_modules = dict(
        [
            (key, value)
            for key, value in sys.modules.items()
            if key.startswith(package_name) and isinstance(value, types.ModuleType)
        ]
    )

    # delete references to these loaded modules from sys.modules
    for key in loaded_package_modules:
        del sys.modules[key]

    # load each of the modules again
    for key in loaded_package_modules:
        import_module(key)


def test_utils_module_load():
    from ccc import utils

    assert utils is not None
    assert utils.__file__ is not None

    del sys.modules["ccc.utils"]


def test_utils_module_loads_without_log():
    # make sure ccc.utils could be loaded even if ccc.log is not available.
    # this happens when CCC is distributed as a PyPI package (ccc.log is not available).
    sys.modules["ccc.log"] = None

    try:
        from ccc import utils

        reload_package(utils)

        assert utils is not None
        assert utils.__file__ is not None
    finally:
        del sys.modules["ccc.log"]
        from ccc import utils

        reload_package(utils)


def test_curl_file_exists():
    from ccc.utils import curl

    download_func = MagicMock()

    curl(
        "http://some-url.com/file.txt",
        Path(__file__).parent / "data" / "file.txt",
        download_file_func=download_func,
    )

    # since file exists, download_file_func should not have been called
    download_func.assert_not_called()


def test_curl_file_exists_hash_matches():
    from ccc.utils import curl

    download_func = MagicMock()

    curl(
        "http://some-url.com/file.txt",
        Path(__file__).parent / "data" / "file.txt",
        "4acd80b502319ce7c44eaf490338894c",
        download_file_func=download_func,
    )

    # since file exists and md5 hash matches, download_file_func should not
    # have been called
    download_func.assert_not_called()


def test_curl_file_exists_hash_do_not_match():
    from ccc.utils import curl

    download_func = MagicMock()

    # here the file to be downloaded exists, but md5 hash given do not match
    # it is expected that download_func is called

    url = "http://some-url.com/file.txt"
    output_file = Path(__file__).resolve().parent / "data" / "file.txt"
    invalid_md5hash = ("5acd80b50231ace7c44eaf490338894a",)  # invalid hash

    curl(
        url,
        output_file,
        invalid_md5hash,
        download_file_func=download_func,
        raise_on_md5hash_mismatch=False,
    )

    download_func.assert_called_with(url, output_file)


def test_curl_file_is_downloaded_successfully():
    from ccc.utils import curl

    def copy_orig_file(url, output_file):
        """Mimics a download function that actually just copies the expected
        file"""
        import shutil

        orig_file = Path(__file__).resolve().parent / "data" / "file.txt"
        shutil.copy(orig_file, output_file)

    download_func = MagicMock(side_effect=copy_orig_file)

    url = "http://some-url.com/file.txt"
    output_file = Path(__file__).resolve().parent / "data" / "another_file.txt"
    md5hash = "4acd80b502319ce7c44eaf490338894c"

    if output_file.exists():
        output_file.unlink()

    curl(
        url,
        output_file,
        md5hash,
        download_file_func=download_func,
    )

    download_func.assert_called_with(url, output_file)
    output_file.unlink()


def test_curl_file_is_downloaded_with_error():
    from ccc.utils import curl

    # this tests that after an unsuccessful file download, the curl function
    # raises an AssertionException, because the given md5 hash does not match

    def copy_wrong_file(url, output_file):
        """Mimics a download error by copying the wrong file (file2.txt instead
        of file1.txt)"""
        import shutil

        orig_file = Path(__file__).resolve().parent / "data" / "file2.txt"
        shutil.copy(orig_file, output_file)

    download_func = MagicMock(side_effect=copy_wrong_file)

    url = "http://some-url.com/file.txt"
    output_file = Path(__file__).resolve().parent / "data" / "another_file.txt"
    md5hash = "4acd80b502319ce7c44eaf490338894c"

    if output_file.exists():
        output_file.unlink()

    with pytest.raises(AssertionError):
        curl(
            url,
            output_file,
            md5hash,
            download_file_func=download_func,
        )

    download_func.assert_called_with(url, output_file)
    output_file.unlink()


def test_simplify_string_simple():
    from ccc.utils import simplify_string

    # lower
    orig_value = "Whole Blood"
    exp_value = "whole_blood"

    obs_value = simplify_string(orig_value.lower())
    assert obs_value is not None
    assert obs_value == exp_value

    # keep original
    orig_value = "Whole Blood"
    exp_value = "Whole_Blood"

    obs_value = simplify_string(orig_value)
    assert obs_value is not None
    assert obs_value == exp_value


def test_simplify_string_with_dash():
    from ccc.utils import simplify_string

    orig_value = "Muscle - Skeletal"
    exp_value = "muscle_skeletal"

    obs_value = simplify_string(orig_value.lower())
    assert obs_value is not None
    assert obs_value == exp_value


def test_simplify_string_with_number():
    from ccc.utils import simplify_string

    orig_value = "Brain - Frontal Cortex (BA9)"
    exp_value = "brain_frontal_cortex_ba9"

    obs_value = simplify_string(orig_value.lower())
    assert obs_value is not None
    assert obs_value == exp_value


def test_simplify_string_other_special_chars():
    from ccc.utils import simplify_string

    orig_value = "Skin - Sun Exposed (Lower leg)"
    exp_value = "skin_sun_exposed_lower_leg"

    obs_value = simplify_string(orig_value.lower())
    assert obs_value is not None
    assert obs_value == exp_value


def test_chunker_simple():
    from ccc.utils import chunker

    assert list(chunker([0, 1, 2, 3], 1)) == [[0], [1], [2], [3]]
    assert list(chunker([0, 1, 2, 3], 2)) == [[0, 1], [2, 3]]
    assert list(chunker([0, 1, 2, 3], 3)) == [[0, 1, 2], [3]]


def test_chunker_larger():
    from ccc.utils import chunker

    assert list(chunker(list(range(100)), 33)) == [
        list(range(0, 33)),
        list(range(33, 66)),
        list(range(66, 99)),
        [99],
    ]

    assert list(chunker(list(range(100)), 34)) == [
        list(range(0, 34)),
        list(range(34, 68)),
        list(range(68, 100)),
    ]


def test_human_format():
    from ccc.utils import human_format

    assert human_format(1) == "1"
    assert human_format(10) == "10"
    assert human_format(100) == "100"
    assert human_format(500) == "500"
    assert human_format(1000) == "1K"
    assert human_format(1100) == "1.1K"
    assert human_format(10000) == "10K"
    assert human_format(100000) == "100K"
    assert human_format(1000000) == "1M"
    assert human_format(1390000) == "1.39M"


def test_upper_triag_square_dataframe():
    from ccc.utils import get_upper_triag

    sim_matrix_df = pd.DataFrame(
        [
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0],
        ]
    )

    up_matrix_df = get_upper_triag(sim_matrix_df)

    assert up_matrix_df is not None
    assert sim_matrix_df.shape == up_matrix_df.shape
    pd.testing.assert_frame_equal(
        up_matrix_df,
        pd.DataFrame(
            [
                [np.nan, 1, 2],
                [np.nan, np.nan, 3],
                [np.nan, np.nan, np.nan],
            ]
        ),
    )


def test_upper_triag_square_dataframe_k0():
    from ccc.utils import get_upper_triag

    sim_matrix_df = pd.DataFrame(
        [
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0],
        ]
    )

    up_matrix_df = get_upper_triag(sim_matrix_df, k=0)

    assert up_matrix_df is not None
    assert sim_matrix_df.shape == up_matrix_df.shape
    pd.testing.assert_frame_equal(
        up_matrix_df,
        pd.DataFrame(
            [
                [0, 1, 2],
                [np.nan, 0, 3],
                [np.nan, np.nan, 0],
            ]
        ),
    )
