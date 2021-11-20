"""
Tests the utils.py module.
"""
from unittest.mock import MagicMock
from pathlib import Path

import pytest

from clustermatch.utils import simplify_string


def test_utils_module_load():
    from clustermatch import utils

    assert utils is not None
    assert utils.__file__ is not None


def test_curl_file_exists():
    from clustermatch.utils import curl

    download_func = MagicMock()

    curl(
        "http://some-url.com/file.txt",
        Path(__file__).parent / "data" / "file.txt",
        download_file_func=download_func,
    )

    # since file exists, download_file_func should not have been called
    download_func.assert_not_called()


def test_curl_file_exists_hash_matches():
    from clustermatch.utils import curl

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
    from clustermatch.utils import curl

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
    from clustermatch.utils import curl

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
    from clustermatch.utils import curl

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
    orig_value = "Muscle - Skeletal"
    exp_value = "muscle_skeletal"

    obs_value = simplify_string(orig_value.lower())
    assert obs_value is not None
    assert obs_value == exp_value


def test_simplify_string_with_number():
    orig_value = "Brain - Frontal Cortex (BA9)"
    exp_value = "brain_frontal_cortex_ba9"

    obs_value = simplify_string(orig_value.lower())
    assert obs_value is not None
    assert obs_value == exp_value


def test_simplify_string_other_special_chars():
    orig_value = "Skin - Sun Exposed (Lower leg)"
    exp_value = "skin_sun_exposed_lower_leg"

    obs_value = simplify_string(orig_value.lower())
    assert obs_value is not None
    assert obs_value == exp_value


#
#
# def test_copy_func():
#     def myfunc(x, y):
#         return x + y
#
#     assert myfunc(7, 13) == 20
#     assert "myfunc" in str(myfunc)
#
#     new_func = copy_func(myfunc)
#     assert new_func(7, 13) == 20
#     assert myfunc != new_func
#     assert "myfunc" in str(new_func)
#
#
# def test_copy_func_new_name():
#     def myfunc(x, y):
#         return x + y
#
#     assert myfunc(7, 13) == 20
#     assert "myfunc" in str(myfunc)
#
#     new_func = copy_func(myfunc, "new_func")
#     assert new_func(7, 13) == 20
#     assert myfunc != new_func
#     assert "myfunc" not in str(new_func)
#     assert "new_func" in str(new_func)
