from ccc.coef.impl_gpu import ccc
import numpy as np


def test_compute_parts():
    # random_feature1 = np.random.rand(100)
    # random_feature2 = np.random.rand(100)
    #
    # res = ccc(random_feature1, random_feature2, n_jobs=2)
    # print(res)

    data = np.random.rand(10, 100)
    c = ccc(data)
    print(c)






