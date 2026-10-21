"""Banded one-to-many DTW copied from the benchmarked scratch/new_dtw.py.

Squared-error costs, rolling cost rows and byte traceback pointers preserve
the benchmark implementation, including left-first tie breaking. The checked
wrapper supplies the missing array conversion and limits traceback memory.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _banded_onetomany_core_rolling(x, y, starts, ends):
    """Return cost and (base, mean-bin) path inside inclusive row bands."""
    n, m = len(x), len(y)

    max_bw = 0
    for i in range(n):
        bw = ends[i] - starts[i] + 1
        if bw > max_bw:
            max_bw = bw

    # Only two rows of DP costs.
    prev = np.empty(max_bw, dtype=np.float64)
    curr = np.empty(max_bw, dtype=np.float64)

    # Store only whether the optimal predecessor was diagonal.
    # 1 = diagonal, 0 = left.
    diag_ptr = np.empty((n, max_bw), dtype=np.uint8)

    # First row: only left transitions are possible.
    xi = x[0]
    d = xi - y[0]
    prev[0] = d * d

    for j in range(1, ends[0] + 1):
        d = xi - y[j]
        prev[j] = prev[j - 1] + d * d

    # Subsequent rows.
    for i in range(1, n):
        lo, hi = starts[i], ends[i]
        prev_lo = starts[i - 1]
        prev_hi = ends[i - 1]
        xi = x[i]

        for j in range(lo, hi + 1):
            k = j - lo

            d = xi - y[j]
            c = d * d

            # Left: extend the current short-sequence index.
            left = curr[k - 1] + c if k > 0 else np.inf

            # Diagonal: start the next short-sequence index.
            diag = (
                prev[j - 1 - prev_lo] + c
                if prev_lo <= j - 1 <= prev_hi
                else np.inf
            )

            # Preserve the original left-first tie-breaking.
            use_diag = diag < left

            curr[k] = diag if use_diag else left
            diag_ptr[i, k] = use_diag

        prev, curr = curr, prev

    cost = prev[m - 1 - starts[n - 1]]

    if not np.isfinite(cost):
        return np.inf, np.empty((0, 2), dtype=np.int32)

    # Every long-sequence index occurs exactly once.
    # Therefore the path has exactly m rows.
    path = np.empty((m, 2), dtype=np.int32)

    i = n - 1
    for j in range(m - 1, -1, -1):
        path[j, 0] = i
        path[j, 1] = j

        if j > 0 and i > 0 and diag_ptr[i, j - starts[i]]:
            i -= 1

    return cost, path


@njit(cache=True)
def _banded_onetomany_fused(x, y, window):
    """Generate the endpoint-to-endpoint sloped band and execute DTW."""
    n, m = len(x), len(y)

    # With equal lengths, every move must be diagonal.
    if n == m:
        path = np.empty((m, 2), dtype=np.int32)
        cost = 0.0

        for j in range(m):
            d = x[j] - y[j]
            cost += d * d
            path[j, 0] = j
            path[j, 1] = j

        if not np.isfinite(cost):
            return np.inf, np.empty((0, 2), dtype=np.int32)
        return cost, path

    # With one short-sequence element, every move is left.
    if n == 1:
        path = np.empty((m, 2), dtype=np.int32)
        cost = 0.0

        for j in range(m):
            d = x[0] - y[j]
            cost += d * d
            path[j, 0] = 0
            path[j, 1] = j

        if not np.isfinite(cost):
            return np.inf, np.empty((0, 2), dtype=np.int32)
        return cost, path

    # Generate sloped band directly in compiled code.
    starts = np.empty(n, dtype=np.int32)
    ends = np.empty(n, dtype=np.int32)

    scale = (m - 1) / (n - 1)

    for i in range(n):
        center = int(np.rint(i * scale))
        starts[i] = max(0, center - window)
        ends[i] = min(m - 1, center + window)

    starts[0] = 0
    ends[n - 1] = m - 1

    return _banded_onetomany_core_rolling(
        x, y, starts, ends
    )


def warping_path_sloped_fast(s1, s2, window=1000, *, max_traceback_bytes=256*1024**2):
    """Return the monotone (base, mean-bin) path and summed squared error.

    Inputs are finite 1D arrays with 1 <= len(s1) <= len(s2). The window is
    the nonnegative half-band in observed bins, not raw samples. The default
    256 MiB traceback limit bounds the per-read allocation in loader workers;
    an oversized read raises before allocation. Select hierarchical chunking
    for such reads. No window expansion or backend fallback is performed.
    """
    x = np.ascontiguousarray(s1, dtype=np.float64)
    y = np.ascontiguousarray(s2, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or not len(x) or not len(y):
        raise ValueError("DTW sequences must be nonempty 1D arrays")
    if len(x) > len(y):
        raise ValueError("DTW requires at least one mean bin per expected base")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("DTW sequences must be finite")
    if not isinstance(window, (int, np.integer)) or isinstance(window, bool) or window < 0:
        raise ValueError("DTW window must be a nonnegative integer")
    if max(len(x), len(y)) > np.iinfo(np.int32).max:
        raise ValueError("DTW input exceeds the int32 path coordinate range")
    # Equal-length and single-base paths do not allocate a traceback matrix.
    if 1 < len(x) < len(y):
        required = len(x) * min(len(y), 2*int(window)+1)
        if required > max_traceback_bytes:
            raise ValueError(
                f"DTW traceback requires up to {required / 1024**2:.1f} MiB; "
                "use hierarchical chunking for this read")
    cost, path = _banded_onetomany_fused(x, y, int(window))
    return path, cost
