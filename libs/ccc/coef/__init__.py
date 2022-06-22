from ccc.coef.impl import *  # noqa: F403, F401

try:
    # Run CCC to initialize/compile its functions with numba
    from ccc.coef.impl import ccc
    import numpy as np

    ccc(np.random.rand(10), np.random.rand(10))
except Exception as e:
    raise e
