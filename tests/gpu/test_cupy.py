# import cupy as cp
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def test_digitize():
#     # random_feature1 = np.random.rand(100)
#     # random_feature2 = np.random.rand(100)
#     #
#     # res = ccc(random_feature1, random_feature2, n_jobs=2)
#     # print(res)
#
#     # Create a sample CuPy array
#     x = cp.array([1.2, 3.0, 4.5, 6.7, 8.9, 10.1, 12.3, 14.5, 16.7, 18.9])
#
#     # Create bins
#     bins = cp.array([0, 5, 10, 15, 20])
#
#     # Use digitize to find which bin each value in x belongs to
#     indices = cp.digitize(x, bins)
#
#     print("Input array x:", x)
#     print("Bins:", bins)
#     print("Bin indices:", indices)
#
#     # Demonstrate the effect of the 'right' parameter
#     indices_right = cp.digitize(x, bins, right=True)
#     print("Bin indices (right=True):", indices_right)
#
#     # Use digitize with decreasing bins
#     decreasing_bins = cp.array([20, 15, 10, 5, 0])
#     indices_decreasing = cp.digitize(x, decreasing_bins)
#     print("Bin indices (decreasing bins):", indices_decreasing)
#
#     # Create a larger random dataset
#     large_x = cp.random.uniform(0, 100, 1000000)
#     large_bins = cp.linspace(0, 100, 11)  # 10 bins
#
#     # Digitize the large dataset
#     large_indices = cp.digitize(large_x, large_bins)
#
#     # Compute histogram
#     hist, _ = cp.histogram(large_x, bins=large_bins)
#
#     print("Histogram of large dataset:", hist)
#
#     # Plot the histogram (using CPU arrays for matplotlib)
#     plt.figure(figsize=(10, 6))
#     plt.hist(cp.asnumpy(large_x), bins=cp.asnumpy(large_bins))
#     plt.title("Histogram of Large Dataset")
#     plt.xlabel("Value")
#     plt.ylabel("Frequency")
#     plt.savefig('histogram.png')  # Saves as PNG
#
#     # Compare with NumPy results
#     np_x = cp.asnumpy(x)
#     np_bins = cp.asnumpy(bins)
#     np_indices = np.digitize(np_x, np_bins)
#
#     print("CuPy indices:", indices)
#     print("NumPy indices:", np_indices)
#     print("Results match:", cp.allclose(indices, cp.asarray(np_indices)))
#
#
# def test_quantile():
#     # Create a sample CuPy array
#     a = cp.array([[10, 7, 4], [3, 2, 1]])
#
#     # Simple usage: compute the median (50th percentile) of the entire array
#     median = cp.quantile(a, 0.5)
#     print("Median of the entire array:", median)
#
#     # Compute multiple quantiles
#     quantiles = cp.quantile(a, [0.25, 0.5, 0.75])
#     print("25th, 50th, and 75th percentiles:", quantiles)
#
#     # Compute quantiles along a specific axis
#     axis_quantiles = cp.quantile(a, 0.5, axis=0)
#     print("Median along axis 0:", axis_quantiles)
#
#     # Compute quantiles for a larger array
#     large_array = cp.random.randn(1000000)
#     large_quantiles = cp.quantile(large_array, [0.1, 0.5, 0.9])
#     print("Quantiles of large array:", large_quantiles)
#
#     # Use an output array
#     out_array = cp.zeros(3)
#     cp.quantile(large_array, [0.1, 0.5, 0.9], out=out_array)
#     print("Output array:", out_array)
#
#     # Compare with NumPy (CPU) results
#     np_array = cp.asnumpy(large_array)
#     np_quantiles = np.quantile(np_array, [0.1, 0.5, 0.9])
#     print("NumPy quantiles:", np_quantiles)
#     print("CuPy and NumPy results are close:", cp.allclose(large_quantiles, np_quantiles))
#
#     # NANs in array
#     nan_array = cp.array([1, 2, cp.nan, 4, 5])
#     nan_quantiles = cp.quantile(nan_array, 0.5)
#     print("Quantile with NaNs:", nan_quantiles)
#
#     # NANs in q
#     array_with_q = cp.array([1, 2, 3, 4, 5])
#     q_with_nan = cp.array([0.5, cp.nan])
#     quantiles_with_nan = cp.quantile(array_with_q, q_with_nan)
#     print("Quantiles with NaN in q:", quantiles_with_nan)