"""
Tests the utils.py module.
"""
from clustermatch.utils import simplify_string


def test_utils_module_load():
    from clustermatch import utils

    assert utils is not None
    assert utils.__file__ is not None


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
