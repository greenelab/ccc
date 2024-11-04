import pytest

import numpy as np
from numpy.testing import assert_array_equal

from ccc.coef.impl_gpu import (
    get_perc_from_k,
)

from ccc.coef import get_perc_from_k as get_perc_from_k_cpu


def test_get_perc_from_k_with_k_less_than_two():
    empty_array = np.empty(0)
    assert_array_equal(get_perc_from_k(1), empty_array)
    assert_array_equal(get_perc_from_k(0), empty_array)
    assert_array_equal(get_perc_from_k(-1), empty_array)


@pytest.mark.parametrize("k", [
    2, 3, 4, 5, 6, 7, 8, 9, 10
])
def test_get_perc_from_k(k):
    np.set_printoptions(precision=17)
    gpu_result = get_perc_from_k(k)
    cpu_result = get_perc_from_k_cpu(k)
    assert np.allclose(gpu_result, cpu_result)
