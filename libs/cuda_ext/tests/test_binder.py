import cuda_ccc
import inspect
import numpy as np


parts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
n_features = 3
n_parts = 1
n_samples = 3
r = cuda_ccc.ari(parts, n_samples, n_features, n_parts)
print(r)
