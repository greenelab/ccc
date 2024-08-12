import cupy as cp
import numpy as np
import matplotlib.pyplot as plt


def test_digitize():
    # random_feature1 = np.random.rand(100)
    # random_feature2 = np.random.rand(100)
    #
    # res = ccc(random_feature1, random_feature2, n_jobs=2)
    # print(res)

    # Create a sample CuPy array
    x = cp.array([1.2, 3.0, 4.5, 6.7, 8.9, 10.1, 12.3, 14.5, 16.7, 18.9])

    # Create bins
    bins = cp.array([0, 5, 10, 15, 20])

    # Use digitize to find which bin each value in x belongs to
    indices = cp.digitize(x, bins)

    print("Input array x:", x)
    print("Bins:", bins)
    print("Bin indices:", indices)

    # Demonstrate the effect of the 'right' parameter
    indices_right = cp.digitize(x, bins, right=True)
    print("Bin indices (right=True):", indices_right)

    # Use digitize with decreasing bins
    decreasing_bins = cp.array([20, 15, 10, 5, 0])
    indices_decreasing = cp.digitize(x, decreasing_bins)
    print("Bin indices (decreasing bins):", indices_decreasing)

    # Create a larger random dataset
    large_x = cp.random.uniform(0, 100, 1000000)
    large_bins = cp.linspace(0, 100, 11)  # 10 bins

    # Digitize the large dataset
    large_indices = cp.digitize(large_x, large_bins)

    # Compute histogram
    hist, _ = cp.histogram(large_x, bins=large_bins)

    print("Histogram of large dataset:", hist)

    # Plot the histogram (using CPU arrays for matplotlib)
    plt.figure(figsize=(10, 6))
    plt.hist(cp.asnumpy(large_x), bins=cp.asnumpy(large_bins))
    plt.title("Histogram of Large Dataset")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig('histogram.png')  # Saves as PNG

    # Compare with NumPy results
    np_x = cp.asnumpy(x)
    np_bins = cp.asnumpy(bins)
    np_indices = np.digitize(np_x, np_bins)

    print("CuPy indices:", indices)
    print("NumPy indices:", np_indices)
    print("Results match:", cp.allclose(indices, cp.asarray(np_indices)))




