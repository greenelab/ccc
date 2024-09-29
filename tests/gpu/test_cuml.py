import cupy as cp
import numpy as np
import pytest
from cuml.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_rand_score
from cuml.common import CumlArray
from cuml.internals.memory_utils import using_output_type
from cuml.internals.safe_imports import gpu_only_import
import time
from pylibraft.common import Stream, DeviceResources


def generate_random_labels(size, n_classes):
    return cp.random.randint(0, n_classes, size=size, dtype=cp.int32)


def compute_ari_with_stream(handle, labels1, labels2):
    with using_output_type("cupy"):
        return adjusted_rand_score(labels1, labels2, handle=handle)


def test_stream():
    n_samples = 10000
    n_classes = 5
    n_iterations = 100

    cupy_stream = cp.cuda.Stream()
    # Create a RAFT handle
    handle = DeviceResources(stream=cupy_stream.ptr)

    # Generate random labels
    labels1 = [generate_random_labels(n_samples, n_classes) for _ in range(n_iterations)]
    labels2 = [generate_random_labels(n_samples, n_classes) for _ in range(n_iterations)]

    # Create CUDA streams
    n_streams = 4  # You can adjust this number based on your GPU
    streams = [cp.cuda.Stream() for _ in range(n_streams)]

    results = []
    start_time = time.time()

    # for i in range(n_iterations):
    #     stream = streams[i % n_streams]
    #     with stream:
    #         handle.set_stream(stream.ptr)
    #         ari = compute_ari_with_stream(handle, labels1[i], labels2[i])
    #         results.append(ari)
    #
    # # Synchronize all streams
    # for stream in streams:
    #     stream.synchronize()

    ari = compute_ari_with_stream(handle, labels1[0], labels2[0])
    results.append(ari)

    end_time = time.time()

    # Print results
    print(f"Computed {n_iterations} ARI scores")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(results)


def generate_data(size):
    np.random.seed(42)
    labels_true = np.random.randint(0, 10, size=size)
    labels_pred = np.random.randint(0, 10, size=size)
    return labels_true, labels_pred


def time_function(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    return result, end_time - start_time


@pytest.mark.parametrize("size", [1000, 10000, 100000, 1000000])
def test_adjusted_rand_score_speedup(size):
    from cuml.metrics import adjusted_rand_score as cuml_ari
    from sklearn.metrics import adjusted_rand_score as sklearn_ari
    labels_true, labels_pred = generate_data(size)

    # Sklearn (CPU) implementation
    _, sklearn_time = time_function(sklearn_ari, labels_true, labels_pred)

    # cuML (GPU) implementation
    labels_true_gpu = cp.asarray(labels_true)
    labels_pred_gpu = cp.asarray(labels_pred)
    _, cuml_time = time_function(cuml_ari, labels_true_gpu, labels_pred_gpu)

    speedup = sklearn_time / cuml_time

    print(f"\nData size: {size}")
    print(f"Sklearn time: {sklearn_time:.6f} seconds")
    print(f"cuML time: {cuml_time:.6f} seconds")
    print(f"Speedup: {speedup:.2f}x")

    # assert speedup > 1, f"cuML should be faster than sklearn, but speedup was only {speedup:.2f}x"

