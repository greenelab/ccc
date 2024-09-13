import pytest

import numpy as np
from ccc.coef.impl_gpu import ccc as ccc_gpu

def test_temp():
    np.random.seed(0)
    feature1 = np.random.rand(100)
    feature2 = np.random.rand(100)
    c = ccc_gpu(feature1, feature2)
    print(c)
