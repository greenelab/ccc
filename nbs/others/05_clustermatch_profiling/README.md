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

* `08_cm_optimized`:
  * this test was not performed since it didn't seem to improve results; I leave
    it here in case it is tried again in the future
  * e idea was that the code is exactly the same as in `07_cm_optimized`, but
    the conda environment contained the package `icc_rt` suggested in the numba
    documentation
    [here](https://numba.pydata.org/numba-doc/latest/user/performance-tips.html#intel-svml)
    to use the Intel SVML library to speed up math operations.
