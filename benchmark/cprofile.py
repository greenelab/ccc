import numpy as np
from ccc.coef import ccc

# python -m cProfile myscript.py

random_feature1 = np.random.rand(100000)
random_feature2 = np.random.rand(100000)

res = ccc(random_feature1, random_feature2)

