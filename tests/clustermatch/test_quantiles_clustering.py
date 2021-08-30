from random import shuffle

import numpy as np
from sklearn.metrics import adjusted_rand_score as ari

from clustermatch.coef import run_quantile_clustering


def test_two_clusters01():
    # Prepare
    np.random.seed(0)

    data = np.concatenate((np.random.normal(0, 1, 10), np.random.normal(5, 1, 10)))
    data_ref = np.concatenate(([0] * 10, [1] * 10))

    idx_shuffled = list(range(len(data)))
    shuffle(idx_shuffled)

    data = data[idx_shuffled]
    data_ref = data_ref[idx_shuffled]

    # Run
    part = run_quantile_clustering(data, 2)

    # Validate
    assert part is not None
    assert len(part) == 20
    assert len(np.unique(part)) == 2
    assert ari(data_ref, part) == 1.0, ari(data_ref, part)


def test_two_clusters02():
    # Prepare
    np.random.seed(0)

    data = np.concatenate(
        (
            np.random.normal(-3, 0.5, 5),
            np.random.normal(0, 1, 5),
            np.random.normal(5, 1, 5),
            np.random.normal(10, 1, 5),
        )
    )
    data_ref = np.concatenate(([0] * 10, [1] * 10))

    idx_shuffled = list(range(len(data)))
    shuffle(idx_shuffled)

    data = data[idx_shuffled]
    data_ref = data_ref[idx_shuffled]

    # Run
    part = run_quantile_clustering(data, 2)

    # Validate
    assert part is not None
    assert len(part) == 20
    assert len(np.unique(part)) == 2
    assert ari(data_ref, part) == 1.0, ari(data_ref, part)


def test_four_clusters01():
    # Prepare
    np.random.seed(0)

    data = np.concatenate(
        (
            np.random.normal(-3, 0.5, 5),
            np.random.normal(0, 1, 5),
            np.random.normal(5, 1, 5),
            np.random.normal(10, 1, 5),
        )
    )
    data_ref = np.concatenate(([0] * 5, [1] * 5, [2] * 5, [3] * 5))

    idx_shuffled = list(range(len(data)))
    shuffle(idx_shuffled)

    data = data[idx_shuffled]
    data_ref = data_ref[idx_shuffled]

    # Run
    part = run_quantile_clustering(data, 4)

    # Validate
    assert part is not None
    assert len(part) == 20
    assert len(np.unique(part)) == 4
    assert ari(data_ref, part) == 1.0
