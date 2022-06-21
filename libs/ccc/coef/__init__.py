from ccc.coef.impl import *  # noqa: F403, F401

try:
    # Run CCC to initialize/compile its functions with numba
    ccc(np.random.rand(10), np.random.rand(10))
except Exception as e:
    raise e
