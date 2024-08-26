import pytest
import time
from ccc.coef.impl_gpu import ccc as ccc_gpu
from ccc.coef.impl import ccc
import numpy as np



@pytest.mark.parametrize("seed, size, distribution, params", [
    (0, 1000, "rand", {}),  # Uniform distribution
    (42, 5000, "randn", {}),  # Normal distribution
    (123, 100, "randint", {"low": 0, "high": 100}),  # Integer distribution
    (456, 10000, "exponential", {"scale": 2.0}),  # Exponential distribution
    # (789, 100, "binomial", {"n": 10, "p": 0.5}),  # Binomial distribution
])
def test_ccc_gpu_1d(seed, size, distribution, params):
    np.random.seed(seed)

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
    elif distribution == "binomial":
        random_feature1 = np.random.binomial(params["n"], params["p"], size)
        random_feature2 = np.random.binomial(params["n"], params["p"], size)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    c1 = ccc_gpu(random_feature1, random_feature2)
    c2 = ccc(random_feature1, random_feature2)

    assert np.isclose(c1, c2, rtol=1e-5, atol=1e-8)


# Additional test for edge cases
@pytest.mark.parametrize("case", [
    "identical",
    "opposite",
    "constant",
    "single_value",
])
def test_ccc_gpu_1d_edge_cases(case):
    if case == "identical":
        feature = np.random.rand(1000)
        c1 = ccc_gpu(feature, feature)
        c2 = ccc(feature, feature)
    elif case == "opposite":
        feature = np.random.rand(1000)
        c1 = ccc_gpu(feature, -feature)
        c2 = ccc(feature, -feature)
    elif case == "constant":
        feature1 = np.full(1000, 5)
        feature2 = np.full(1000, 3)
        c1 = ccc_gpu(feature1, feature2)
        c2 = ccc(feature1, feature2)
    elif case == "single_value":
        # Too few objects
        feature = np.array([1])
        with pytest.raises(ValueError) as e:
            c1 = ccc_gpu(feature, feature)
            assert "Too few objects" in e.value
        with pytest.raises(ValueError) as e:
            c2 = ccc(feature, feature)
            assert "Too few objects" in e.value
    return


def test_ccc_gpu_2d_simple():
    np.random.seed(0)
    shape = (20       , 200)  # 200 features, 1,000 samples
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

    print(f"\nGPU time: {gpu_time:.4f} seconds")
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")

    assert np.allclose(c1, c2, rtol=1e-5, atol=1e-5)

    return gpu_time, cpu_time, speedup