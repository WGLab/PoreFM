"""Compiled peak selection on stitched canonical-null read profiles.

This module is imported only for peak filtering. Detection uses the original
float32 p-values, without logarithms, smoothing, or copies of the profile.
"""

import numpy as np
from numba import njit


def select_pvalue_peaks(pvalues, valid, eligible, prune_radius=0):
    """Return qualifying local p-value minima in ascending read-position order.

    Parameters
    ----------
    pvalues : numpy.ndarray, shape (L,)
        Contiguous float32 p-values indexed by zero-based read position.
    valid : numpy.ndarray of bool, shape (L,)
        Structural validity of each base (label and complete k-mer). False
        entries and nonfinite p-values break the profile. A peak needs valid,
        finite, strictly higher p-values on both sides of its entire plateau;
        segment endpoints are never peaks.
    eligible : numpy.ndarray of bool, shape (L,)
        Output eligibility, including the chosen score or p-value threshold.
        This mask is tested only at peak midpoints, never used to remove
        neighboring values when identifying peaks.
    prune_radius : int, default=0
        Nonnegative suppression radius in read bases; zero disables pruning.
        Lower p-values take priority, with lower read position breaking ties.
        A retained peak suppresses other candidates at distance <= radius.

    Returns
    -------
    numpy.ndarray of int64
        Selected positions. Equal-valued plateaus use their midpoint rounded
        toward the lower read position. Inputs are not modified.

    Notes
    -----
    The caller supplies equally sized arrays and validates the radius. Work
    is O(L) without pruning and O(L + K log K) with K qualifying peaks. Scratch
    storage is O(L + K), local to one read. Stable sorting makes suppression
    independent of traversal direction, except for the documented tie rule.
    """
    peaks = _find_pvalue_peaks(pvalues, valid, eligible)
    if prune_radius == 0 or peaks.size < 2:
        return peaks
    # Keep sorting in NumPy: its compiled stable sort is faster than Numba's
    # mergesort on dense peak profiles. Coordinate order resolves equal p's.
    priority = np.argsort(pvalues[peaks], kind='stable')
    return _prune_peaks(peaks, priority, prune_radius)


@njit(cache=True, nogil=True, fastmath=False)
def _find_pvalue_peaks(pvalues, valid, eligible):
    """Scan equal-length read arrays once; return eligible plateau midpoints."""
    # There can be at most L // 2 distinct minima. Allocate one index buffer,
    # and record only candidates that pass the output filter.
    peaks = np.empty(pvalues.size // 2, dtype=np.int64)
    count = 0
    i = 1
    last = pvalues.size - 1
    while i < last:
        if (valid[i - 1] and valid[i]
                and np.isfinite(pvalues[i - 1]) and np.isfinite(pvalues[i])
                and pvalues[i - 1] > pvalues[i]):
            end = i + 1
            while end < last and valid[end] and pvalues[end] == pvalues[i]:
                end += 1
            midpoint = (i + end - 1) // 2
            if (valid[end] and np.isfinite(pvalues[end])
                    and pvalues[end] > pvalues[i] and eligible[midpoint]):
                peaks[count] = midpoint
                count += 1
            # Revisit the first unequal sample: a falling p-value there can
            # itself begin the next minimum. Never jump past that sample.
            i = end
        else:
            i += 1

    return peaks[:count]


@njit(cache=True, nogil=True, fastmath=False)
def _prune_peaks(peaks, priority, prune_radius):
    """Suppress coordinate-sorted peaks in the supplied strongest-first order."""
    count = peaks.size
    keep = np.ones(count, dtype=np.bool_)
    for j in priority:
        if not keep[j]:
            continue
        k = j - 1
        while k >= 0 and peaks[j] - peaks[k] <= prune_radius:
            keep[k] = False
            k -= 1
        k = j + 1
        while k < count and peaks[k] - peaks[j] <= prune_radius:
            keep[k] = False
            k += 1
    return peaks[keep]


def warm_peak_filter():
    """Compile/load detection and suppression once before read processing."""
    empty = np.empty(0, dtype=np.float32)
    mask = np.empty(0, dtype=np.bool_)
    peaks = _find_pvalue_peaks(empty, mask, mask)
    _prune_peaks(peaks, np.empty(0, dtype=np.intp), 0)
