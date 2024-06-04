# CCC-GPU profiling

This folder contains profiling results (with cProfile) of different
optimizations of the clustermatch code. A brief description of each subfolder is
below.

- `00_cpu_version_ref`:
  - Contains benchmarks of the CPU version of CCC (nbs/others/05_clustermatch_profiling/10_cm_optimized):
  1. Numba-enabled, multi-threaded
  2. Numba-disabled, multi-threaded
  - Newly added:
  3. Numba-enabled, single-threaded
  4. Numba-disabled, single-threaded
  

* `01_ari_cuda_v0`:
  - Contains benchmarks of the CUDA version of CCC, functions rewritten in CUDA:
  - `ari`

The tests were run on a System76 Thelio machine with the following specifications:
- 5.3 GHz Threadripper 7960X (24 Cores - 48 Threads)
- 256 GB ECC DDR5 4800 MHz (4x64)
- 24 GB NVIDIA GeForce RTX 4090