import pytest
from typing import List

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numpy.typing import NDArray

from ccc.coef.impl_gpu import (
    get_perc_from_k,
    get_range_n_percentages,
    convert_n_clusters,
    get_range_n_clusters,
    get_parts,
)

from ccc.coef import get_perc_from_k as get_perc_from_k_cpu

def test_get_perc_from_k_with_k_less_than_two():
    empty_array = np.empty(0)
    assert_array_equal(get_perc_from_k(1), empty_array)
    assert_array_equal(get_perc_from_k(0), empty_array)
    assert_array_equal(get_perc_from_k(-1), empty_array)


@pytest.mark.parametrize("k", [
    # 2, 3, 4, 5, 6, 7, 8, 9, 10
    3
])
def test_get_perc_from_k(k):
    np.set_printoptions(precision=17)
    gpu_result = get_perc_from_k(k)
    cpu_result = get_perc_from_k_cpu(k)
    assert np.equal(gpu_result, cpu_result).all()
