import pytest
from typing import List

import numpy as np
from numba import cuda
from numpy.testing import assert_array_equal, assert_allclose
from numpy.typing import NDArray

from ccc.coef.impl_gpu import (
    get_perc_from_k,
    get_range_n_percentages,
    convert_n_clusters,
    get_range_n_clusters,
    get_parts,
)

from ccc.coef import get_parts as get_parts_cpu


@pytest.mark.parametrize("feature_size", [100, 1000, 10000])
@pytest.mark.parametrize("cluster_settings", [
    # ([2], (2,)),
    # ([2, 3], (2, 3)),
    # ([2, 3, 4], (2, 3, 4)),
    # ([5], (5,)),
    ([6], (6,)),
    # ([2, 3, 4, 5, 6, 7, 8, 9, 10], (2, 3, 4, 5, 6, 7, 8, 9, 10)),
])
def test_get_parts(feature_size, cluster_settings):
    np.random.seed(0)
    
    gpu_clusters, cpu_clusters = cluster_settings
    feature = np.random.rand(feature_size)
    
    # GPU implementation
    parts_gpu = get_parts(feature, np.array(gpu_clusters, dtype=np.uint8)).get()
    
    # CPU implementation
    parts_cpu = get_parts_cpu(feature, cpu_clusters)
    
    assert parts_gpu is not None
    assert len(parts_gpu) == 1, "should have only one feature"
    assert len(parts_gpu[0]) == len(gpu_clusters), f"should have {len(gpu_clusters)} partition(s)"
    
    for i, n_clusters in enumerate(gpu_clusters):
        assert len(np.unique(parts_gpu[0][i])) == n_clusters, f"should have {n_clusters} cluster indexes"
        assert np.array_equal(parts_gpu[0][i], parts_cpu[i]), f"GPU and CPU results don't match for {n_clusters} clusters"
    
    # Additional checks for multi-cluster settings
    if len(gpu_clusters) > 1:
        for i in range(len(gpu_clusters)):
            for j in range(i + 1, len(gpu_clusters)):
                assert not np.array_equal(parts_gpu[0][i], parts_cpu[j]), f"Partitions {i} and {j} should not be equal"


def test_get_parts_with_singletons():
    np.random.seed(0)

    feature0 = np.array([1.3] * 100)

    # run
    parts = get_parts(feature0, np.array([2], dtype=np.uint8)).get()
    parts_cpu = get_parts_cpu(feature0, (2,))
    assert parts is not None
    assert len(parts) == 1 # 1 feature
    assert len(parts[0]) == 1 # 1 partition
    # all the elements (2D) should be -2
    np.testing.assert_array_equal(np.unique(parts[0]), np.array([-2]))
    assert np.array_equal(parts[0], parts_cpu)

    parts = get_parts(feature0, np.array([2, 3], dtype=np.uint8)).get()
    parts_cpu = get_parts_cpu(feature0, (2, 3))
    assert parts is not None
    assert len(parts) == 1
    assert len(parts[0]) == 2, "feature should have 2 clusters"
    np.testing.assert_array_equal(np.unique(parts[0][0]), np.array([-2]))
    np.testing.assert_array_equal(np.unique(parts[0][1]), np.array([-2]))
    assert np.array_equal(parts[0][0], parts_cpu[0])
    assert np.array_equal(parts[0][1], parts_cpu[1])



def test_get_parts_with_categorical_feature():
    np.random.seed(0)

    feature0 = np.array([4] * 10)

    # run
    # only one partition is requested
    parts = get_parts(feature0, np.array([2], dtype=np.uint8), data_is_numerical=False).get()
    parts_cpu = get_parts_cpu(feature0, (2,), data_is_numerical=False)
    assert parts is not None
    assert len(parts) == 1
    assert len(parts[0]) == 1
    np.testing.assert_array_equal(np.unique(parts[0]), np.array([4]))
    assert np.array_equal(parts[0], parts_cpu)

    # more partitions are requested; only the first one has valid information
    parts = get_parts(feature0, np.array([2, 3], dtype=np.uint8), data_is_numerical=False).get()
    parts_cpu = get_parts_cpu(feature0, (2, 3), data_is_numerical=False)
    assert parts is not None
    assert len(parts) == 1
    assert len(parts[0]) == 2
    print("parts:")
    print(parts)
    print("parts_cpu:")
    print(parts_cpu)
    np.testing.assert_array_equal(np.unique(parts[0][0]), np.array([4]))
    np.testing.assert_array_equal(np.unique(parts[0][1]), np.array([-1]))
    assert (parts == parts_cpu).all()
    assert np.array_equal(parts[0][0], parts_cpu[0])
    assert np.array_equal(parts[0][1], parts_cpu[1])

def test_get_parts_2d_simple():
    np.random.seed(0)
    array = np.random.rand(5, 1000)
    print(f"array : \n{array}")
    parts = get_parts(array, np.array([3], dtype=np.uint8)).get()
    parts_cpu_row0 = get_parts_cpu(array[0], (3, ))
    parts_cpu_row1 = get_parts_cpu(array[1], (3, ))
    assert parts is not None
    assert (parts[0] == parts_cpu_row0).all()
    assert (parts[1] == parts_cpu_row1).all()
    print("parts:")
    print(parts)
    print("parts_cpu_row0:")
    print(parts_cpu_row0)
    print("parts_cpu_row1:")
    print(parts_cpu_row1)
