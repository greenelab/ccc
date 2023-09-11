# Clustermatch profiling

This folder contains profiling results (with cProfile) of different
optimizations of the clustermatch code. A brief description of each subfolder is
below.

* `05_cm_optimized`:
  * ari implementation with numba
  * precomputing of internal partitions

* `06_cm_optimized`:
  * cm function fully implemented in numba

* `07_cm_optimized`:
  * cm function now supports parallelization (from numba)

* `10_cm_optimized`:
  * optimization for computing ari in parallel (function cdist_parts)
  * many optimizations in other functions associated to _get_parts, such as rank, run_quantile_clustering, etc.
  * the idea here is to optimize the single variable pair processing

* `11_cm_optimized`:
  * after all optimization in 10_cm_optimized, this is a copy of 07_cm_optimized to check
    if the matrix data input keeps working correctly.

* `12_cm_optimized`:
  * a copy of `11_cm_optimized` with some other optimizations.
