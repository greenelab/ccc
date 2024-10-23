from ccc.coef import ccc
import numpy as np


def test_ccc():
    part0 = np.array([2, 3, 6, 1, 0, 5, 4, 3, 6, 2])
    part1 = np.array([0, 6, 2, 5, 1, 3, 4, 6, 0, 2])
    c = ccc(part0, part1)
    print(c)