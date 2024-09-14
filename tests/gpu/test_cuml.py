from cuml.metrics import adjusted_rand_score as cu_rnd_sc
from sklearn.metrics import adjusted_rand_score as sk_rnd_sc

import cupy as cp


def test_rand_score():
    x, y = cp.array([0, 0]), cp.array([0, 0])
    c1 = cu_rnd_sc(x, y)
    c2 = sk_rnd_sc(cp.asnumpy(x), cp.asnumpy(y))
    print(c1, c2)