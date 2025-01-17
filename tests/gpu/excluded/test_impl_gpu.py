import pytest
import time

import numpy as np

from ccc.coef.impl_gpu import ccc as ccc_gpu
from ccc.coef.impl import ccc
from utils import clean_gpu_memory
# This test needs to be improved


def test_ccc_gpu_1d_simple():
    np.random.seed(0)
    feature1 = np.random.rand(10)
    feature2 = np.random.rand(10)
    c1 = ccc_gpu(feature1, feature2)
    c2 = ccc(feature1, feature2)
    print(f"GPU: {c1}, CPU: {c2}")
    assert np.isclose(c1, c2, atol=1e-3), f"GPU: {c1}, CPU: {c2}"


@clean_gpu_memory
def run_ccc_test(size, seed, distribution, params):
    np.random.seed(seed)
    absolute_tolerance = 1e-3  # allow 0.001 as max coefficient difference

    # Generate random features based on the specified distribution
    if distribution == "rand":
        random_feature1 = np.random.rand(size)
        random_feature2 = np.random.rand(size)
    elif distribution == "randn":
        random_feature1 = np.random.randn(size)
        random_feature2 = np.random.randn(size)
    elif distribution == "randint":
        random_feature1 = np.random.randint(params["low"], params["high"], size)
        random_feature2 = np.random.randint(params["low"], params["high"], size)
    elif distribution == "exponential":
        random_feature1 = np.random.exponential(params["scale"], size)
        random_feature2 = np.random.exponential(params["scale"], size)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    c1 = ccc_gpu(random_feature1, random_feature2)
    c2 = ccc(random_feature1, random_feature2)

    is_close = np.isclose(c1, c2, atol=absolute_tolerance)
    return is_close, c1, c2


@pytest.mark.parametrize(
    "distribution, params",
    [
        ("rand", {}),  # Uniform distribution
        ("randn", {}),  # Normal distribution
        (
            "randint",
            {"low": 0, "high": 100},
        ),  # Integer distribution, expect to have the largest difference due to partition errors
        ("exponential", {"scale": 2.0}),  # Exponential distribution
    ],
)
def test_ccc_gpu_1d(distribution, params):
    """
    This test allows for a small percentage (10%) of individual tests to fail for each distribution.
    """
    sizes = np.linspace(100, 100000, num=5, dtype=int)
    seeds = np.linspace(0, 1000, num=5, dtype=int)
    allowed_failure_rate = 0.10  # 10% allowed failure rate

    total_tests = len(sizes) * len(seeds)
    max_allowed_failures = int(total_tests * allowed_failure_rate)
    failures = 0

    for size in sizes:
        for seed in seeds:
            is_close, c1, c2 = run_ccc_test(size, seed, distribution, params)

            if not np.all(is_close):
                failures += 1
                print(
                    f"\nTest failed for size={size}, seed={seed}, distribution={distribution}"
                )
                print(f"GPU result: {c1}")
                print(f"CPU result: {c2}")
                print(f"Differences: {np.abs(c1 - c2)}")

    print(f"\nDistribution: {distribution}")
    print(f"Total tests: {total_tests}")
    print(f"Failed tests: {failures}")
    print(f"Maximum allowed failures: {max_allowed_failures}")

    assert (
        failures <= max_allowed_failures
    ), f"Too many failures for {distribution} distribution: {failures} > {max_allowed_failures}"

    if failures > 0:
        print(
            f"Warning: {failures} tests failed, but within the allowed failure rate of {allowed_failure_rate * 100}%"
        )
    else:
        print("All tests passed successfully")


# Additional test for edge cases
@clean_gpu_memory
@pytest.mark.parametrize(
    "case",
    [
        "identical",
        "opposite",
        "constant",
        "single_value",
    ],
)
def test_ccc_gpu_1d_edge_cases(case):
    if case == "identical":
        feature = np.random.rand(1000)
        ccc_gpu(feature, feature)
        ccc(feature, feature)
    elif case == "opposite":
        feature = np.random.rand(1000)
        ccc_gpu(feature, -feature)
        ccc(feature, -feature)
    elif case == "constant":
        feature1 = np.full(1000, 5)
        feature2 = np.full(1000, 3)
        ccc_gpu(feature1, feature2)
        ccc(feature1, feature2)
    elif case == "single_value":
        # Too few objects
        feature = np.array([1])
        with pytest.raises(ValueError) as e:
            ccc_gpu(feature, feature)
            assert "Too few objects" in e.value
        with pytest.raises(ValueError) as e:
            ccc(feature, feature)
            assert "Too few objects" in e.value
    return


@clean_gpu_memory
def test_ccc_gpu_2d_simple():
    np.random.seed(0)
    shape = (20, 200)  # 200 features, 1,000 samples
    print(f"Testing with {shape[0]} features and {shape[1]} samples")
    df = np.random.rand(*shape)

    # Time GPU version
    start_gpu = time.time()
    c1 = ccc_gpu(df)
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    # Time CPU version
    start_cpu = time.time()
    c2 = ccc(df)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu

    # Calculate speedup
    speedup = cpu_time / gpu_time

    print(f"\nGPU time: {gpu_time:.4f} seconds")
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")

    print(f"GPU coef:\n {c1}")
    print(f"CPU coef:\n {c2}")

    assert np.allclose(c1, c2, rtol=1e-5, atol=1e-5)

    return gpu_time, cpu_time


# Test for very large arrays (may be slow and memory-intensive)
@clean_gpu_memory
@pytest.mark.slow
def test_ccc_gpu_2d_very_large():
    np.random.seed(0)
    shape = (200, 1000)  # 200 features, 1,000 samples
    print(f"Testing with {shape[0]} features and {shape[1]} samples")
    df = np.random.rand(*shape)

    # Time GPU version
    start_gpu = time.time()
    c1 = ccc_gpu(df)
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    # Time CPU version
    start_cpu = time.time()
    c2 = ccc(df)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu

    # Calculate speedup
    speedup = cpu_time / gpu_time

    print(f"Length of the array: {len(c1)}")
    print(f"\nGPU time: {gpu_time:.4f} seconds")
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")

    # Set tolerance parameters
    rtol = 1e-5
    atol = 1e-2
    max_diff_count = int(len(c1) * 0.01)  # Allow up to 1% of elements to be different

    # Compare results
    is_close = np.isclose(c1, c2, rtol=rtol, atol=atol)
    diff_count = np.sum(~is_close)

    print(f"Number of differing elements: {diff_count}")
    print(f"Maximum allowed differences: {max_diff_count}")

    if diff_count > 0:
        # Find indices of differing elements
        diff_indices = np.where(~is_close)

        # Print details of the first 10 differences
        print("\nFirst 10 differences:")
        for i in range(min(10, diff_count)):
            idx = tuple(index[i] for index in diff_indices)
            print(
                f"Index {idx}: GPU = {c1[idx]:.8f}, CPU = {c2[idx]:.8f}, Diff = {abs(c1[idx] - c2[idx]):.8f}"
            )

        # Calculate and print max absolute difference
        max_abs_diff = np.max(np.abs(c1 - c2))
        print(f"\nMaximum absolute difference: {max_abs_diff:.8f}")

    assert (
        diff_count <= max_diff_count
    ), f"Too many differing elements: {diff_count} > {max_diff_count}"

    return gpu_time, cpu_time, speedup
