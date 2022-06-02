"""
Contains implementations of different statistical methods in SciPy but optimized
for numba.

Some code (indicated in each function) is based on SciPy code base
(https://github.com/scipy/scipy), for which the copyright notice, the original
code disclaimer, and license are shown below.

Copyright 2002 Gary Strangman.  All rights reserved
Copyright 2002-2016 The SciPy Developers

The original code from Gary Strangman was heavily adapted for
use in SciPy by Travis Oliphant.  The original code came with the
following disclaimer:

This software is provided "as-is".  There are no expressed or implied
warranties of any kind, including, but not limited to, the warranties
of merchantability and fitness for a given application.  In no event
shall Gary Strangman be liable for any direct, indirect, incidental,
special, exemplary or consequential damages (including, but not limited
to, loss of use, data or profits, or business interruption) however
caused and on any theory of liability, whether in contract, strict
liability or tort (including negligence or otherwise) arising in any way
out of the use of this software, even if advised of the possibility of
such damage.

License from SciPy:

Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(cache=True, nogil=True)
def rank(data: NDArray, sorted_data_idx: NDArray = None) -> NDArray:
    """
    It returns the ranks of a numpy array. It's an implementation based on
    scipy.stats.rankdata (method="average") that can be compiled by numba.

    Args:
        data: a 1d array with numeric data.
        sorted_data_idx: indexes of data that sort the array in ascending order.
          This is usually obtained with np.argsort.

    Returns:
        A 1d array with the ranks of the input data. Ranks start with 1.
    """

    arr = np.ravel(np.asarray(data))
    if sorted_data_idx is None:
        sorter = np.argsort(arr, kind="quicksort")
    else:
        sorter = sorted_data_idx

    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    arr = arr[sorter]
    obs = np.ones(arr.shape[0], dtype=np.bool_)
    obs[1:] = arr[1:] != arr[:-1]
    dense = obs.cumsum()[inv]

    obs_nz = np.nonzero(obs)[0]
    count = np.zeros(obs_nz.shape[0] + 1, dtype=np.int64)
    count[:-1] = obs_nz
    count[-1] = len(obs)

    return 0.5 * (count[dense] + count[dense - 1] + 1)
