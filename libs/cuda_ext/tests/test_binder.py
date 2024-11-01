import cuda_ccc
import inspect
import numpy as np


parts = np.array([[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]], dtype=np.int32, order="C")
print(parts.ndim)
n_features = 3
n_parts = 1
n_samples = 3
r = cuda_ccc.ari(parts, n_samples, n_features, n_parts)
print(r)
