import pytest
import numpy as np
import cuda_ccc

# Test cases taken from sklearn.metrics.adjusted_rand_score
@pytest.mark.parametrize("parts, expected_ari", [
    (
        np.array([
            [[0, 0, 1, 2]],
            [[0, 0, 1, 1]]
        ], dtype=np.int32),
        0.57
    ),
    (
         np.array([
            [[0, 0, 1, 1]],
            [[0, 1, 0, 1]]
        ], dtype=np.int32),
        -0.5
    ),
    (
         np.array([
            [[0, 0, 1, 1]],
            [[0, 0, 1, 1]]
        ], dtype=np.int32),
        1.0
    ),
    (
        np.array([
            [[0, 0, 1, 1]],
            [[1, 1, 0, 0]]
        ], dtype=np.int32),
        1.0
    ),
    (
         np.array([
            [[0, 0, 0, 0]],
            [[0, 1, 2, 3]]
        ], dtype=np.int32),
        0.0
    )
])
def test_cuda_ari_cases(parts, expected_ari):
    n_features, n_parts, n_objs = parts.shape
    ari = cuda_ccc.ari(parts, n_features, n_parts, n_objs)
    assert np.isclose(ari[0], expected_ari, atol=1e-2)
