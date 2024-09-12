import functools
import cupy as cp

def clean_gpu_memory(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
    return wrapper
