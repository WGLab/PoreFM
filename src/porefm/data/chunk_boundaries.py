"""Sequence/signal maps for approximate chunk endpoints.

Coordinates are raw ADC samples and zero-based, half-open original read bases.
Six-sample means are used only for inference; model signal is untouched.
Hierarchical anchors and full banded DTW support DNA/RNA 9-mer current models.
Neither estimator consumes mv. RNA coordinates index the reversed raw signal.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from numba import njit


def validate_chunking_mode(mode):
    """Validate the explicit data-loader policy, without importing torch."""
    if mode not in ("auto", "mv", "hierarchical", "dtw", "inferred"):
        raise ValueError(f"chunking_mode must be auto, mv, hierarchical, dtw, or inferred; got {mode!r}")
    return mode


def uses_inferred_chunks(record, mode):
    """Select inference without treating malformed present moves as absent."""
    validate_chunking_mode(mode)
    return mode in ("hierarchical", "dtw", "inferred") or (mode == "auto" and not record.has_tag("mv"))


def _chunk_windows(start, end, stride, size, step):
    """Raw-sample windows matching the shared padded signal chunk builder."""
    if size < 1 or step < 1:
        raise ValueError("Chunk size and step must be positive")
    rows = (end - start) // stride
    count = max(1, math.ceil((rows - size) / step + 1))
    starts = start + np.arange(count) * step * stride
    return starts, np.minimum(starts + size * stride, end)


@dataclass
class InferredBoundaries:
    """A read's sparse map and actual model-signal extent in raw samples.

    ``signal_knots``/``base_knots`` define a monotone piecewise-linear map.
    ``signal_start:signal_end`` is the cropped signal, with length divisible
    by ``stride`` (the model input width, independent of anchor averaging).
    ``metadata`` records the trusted endpoint tags and anchor diagnostics.
    No per-base dwell times or move events are represented.
    DNA coordinates index the original raw array; RNA coordinates
    index its full reversal, so time increases with sequence position.
    """

    signal_knots: np.ndarray
    base_knots: np.ndarray
    signal_start: int
    signal_end: int
    stride: int
    length: int
    metadata: dict

    def bounds(self, size, step):
        """Return one half-open sequence slice per padded signal window.

        ``size`` and ``step`` count model signal rows, each ``stride`` raw
        samples wide. The final partial window is clipped before interpolation.
        """
        starts, stops = _chunk_windows(self.signal_start, self.signal_end,
                                      self.stride, size, step)
        left = np.floor(interpolate_extrapolate(starts, self.signal_knots, self.base_knots))
        right = np.ceil(interpolate_extrapolate(stops, self.signal_knots, self.base_knots))
        return np.stack((np.clip(left, 0, self.length),
                         np.clip(right, 0, self.length)), axis=1).astype(np.int64)


def _prepare_signal(raw, expected, ts, ns, stride, seq_type):
    """Validate metadata and orient raw coordinates in original sequence order."""
    raw, expected = np.asarray(raw), np.asarray(expected, dtype=np.float64)
    if raw.ndim != 1 or expected.ndim != 1 or len(expected) < 64:
        raise ValueError("Inferred chunking requires 1D signal and at least 64 complete 9-mers")
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("Inferred signal stride must be a positive integer")
    for name, value in (("ts", ts), ("ns", ns)):
        if value is not None and (not isinstance(value, (int, np.integer))
                                  or isinstance(value, bool) or value < 0 or value > len(raw)):
            raise ValueError(f"Invalid {name} raw-sample coordinate: {value!r}")
    if seq_type not in ("dna", "rna"):
        raise ValueError("seq_type must be dna or rna")
    trusted_start = ts is not None and ts > 0
    trusted_end = ns is not None
    origin = int(ts) if trusted_start else 0
    limit = len(raw) if ns is None else int(ns)
    if limit <= origin:
        raise ValueError("ns must be greater than the trusted start")
    if seq_type == "rna":
        # Tags are acquisition-time coordinates. Reverse the full coordinate
        # axis, then swap which sequence endpoint each trusted tag constrains.
        origin, limit = len(raw) - limit, len(raw) - origin
        trusted_start, trusted_end = trusted_end, trusted_start
        raw = raw[::-1]
    return raw, expected, origin, limit, trusted_start, trusted_end


def infer_boundaries(raw, expected, *, ts=None, ns=None, stride=6, seq_type="dna"):
    """Estimate hierarchical anchors from raw signal and complete 9-mer levels.

    ``expected`` excludes the five leading and three trailing undefined
    positions of this repository's 9-mer extraction convention. Positive ts
    fixes the acquisition-time start; zero/absent ts leaves it unknown. ns is
    the exclusive acquisition-time end. Neither tag is required. Invalid
    bounds raise. RNA reverses raw coordinates and swaps endpoint constraints
    while retaining original 5'-to-3' sequence coordinates.

    Queries of 192 bases are recursively refined to roughly 256-base spacing;
    endpoint queries use up to 384 bases. Unknown endpoints are extrapolated.
    Metadata-assisted searches cover neighboring-anchor intervals; tag-free
    searches are bounded. RNA scales search/duration/cost bounds by the mean
    bins/base divided by two (minimum one), allowing longer RNA dwells.
    """
    raw, expected, origin, limit, trusted_start, trusted_end = _prepare_signal(
        raw, expected, ts, ns, stride, seq_type)
    x, y = normalize(raw[origin:limit], expected)
    if len(y) < len(x):
        raise ValueError("Too few six-sample bins for inferred sequence anchors")
    nbase = len(expected) + 8
    wide = trusted_start or trusted_end
    rate_scale = max(1., len(y) / nbase / 2.) if seq_type == "rna" else 1.
    edge_query = min(384, len(x))
    edge_inner = min(32, edge_query // 4) if wide else None
    region = min(len(y), max(int(4096 * rate_scale), len(y) // 3))
    left = ((-5., 0., 0., 1.) if trusted_start else
            local_anchor(x, y, edge_query / 2, edge_query, region / 2,
                         region / 2, 0., edge_inner))
    right = ((float(nbase - 5), (limit - origin) / 6, 0., 1.) if trusted_end else
             local_anchor(x, y, len(x) - edge_query / 2, edge_query,
                          len(y) - region / 2, region / 2, 0.,
                          None if edge_inner is None else edge_query - edge_inner))
    if right[0] <= left[0] or right[1] <= left[1]:
        raise ValueError("Unable to order inferred terminal anchors")
    knots, pending = [left, right], [(left, right)]
    attempted, rejected = int(not trusted_start) + int(not trusted_end), 0
    query = min(192, len(x))
    while pending:
        a, b = pending.pop()
        if b[0] - a[0] <= 256:
            continue
        prediction = (a[1] + b[1]) / 2
        radius = (max(query * 2.5 * rate_scale, .5 * (b[1] - a[1]) + query * 2 * rate_scale) if wide
                  else max(query * 2.5 * rate_scale, min(1500 * rate_scale, .4 * (b[1] - a[1]))))
        knot = local_anchor(x, y, (a[0] + b[0]) / 2, query, prediction, radius, 0.)
        attempted += 1
        accepted = (a[0] < knot[0] < b[0] and a[1] < knot[1] < b[1]
                    and .9 * rate_scale <= knot[3] <= 8 * rate_scale and knot[2] < .65 * rate_scale
                    and .4 * rate_scale <= (knot[1] - a[1]) / (knot[0] - a[0]) <= 12 * rate_scale
                    and .4 * rate_scale <= (b[1] - knot[1]) / (b[0] - knot[0]) <= 12 * rate_scale)
        if accepted:
            knots.append(knot)
            pending.extend(((a, knot), (knot, b)))
        else:
            rejected += 1
    knots.sort()
    signal = np.array([k[1] * 6 + origin for k in knots])
    bases = np.array([k[0] + 5 for k in knots])
    start = origin if trusted_start else int(np.clip(
        np.floor(interpolate_extrapolate([0], bases, signal)[0]), 0, limit))
    end = limit if trusted_end else int(np.clip(
        np.ceil(interpolate_extrapolate([nbase], bases, signal)[0]), start, len(raw)))
    end = start + (end - start) // stride * stride
    if end <= start:
        raise ValueError("Inferred signal extent is empty")
    return InferredBoundaries(signal, bases, start, end, int(stride), nbase,
                              dict(method="hierarchical", ts=ts, ns=ns, trusted_start=trusted_start, seq_type=seq_type,
                                   trusted_end=trusted_end, rate_scale=rate_scale,
                                   attempted=attempted, rejected=rejected))


@dataclass
class DtwBoundaries:
    """Chunk endpoints from a DTW assignment per six-sample mean bin.

    Coordinates follow :class:`InferredBoundaries`. ``base_by_bin`` stores
    original read base indices, not move events or raw-sample dwell times.
    Chunk starts include the base covering their first sample; exclusive ends
    include the base covering their last sample. The model row width ``stride``
    is independent of the fixed inference averaging factor of six.
    """

    base_by_bin: np.ndarray
    signal_start: int
    signal_end: int
    stride: int
    length: int
    metadata: dict

    def bounds(self, size, step):
        """Return half-open base intervals for padded model signal windows."""
        starts, stops = _chunk_windows(self.signal_start, self.signal_end,
                                      self.stride, size, step)
        first = np.clip((starts - self.signal_start) // 6, 0, len(self.base_by_bin)-1)
        last = np.clip((stops - 1 - self.signal_start) // 6, 0, len(self.base_by_bin)-1)
        return np.column_stack((self.base_by_bin[first], self.base_by_bin[last]+1)).astype(np.int64)


def infer_dtw_boundaries(raw, expected, *, ts=None, ns=None, stride=6, seq_type="dna"):
    """Align whole-read six-sample means using a 1000-bin sloped half-band.

    Trim/orientation rules match :func:`infer_boundaries`. With both endpoints
    known, no hierarchical search runs. Missing endpoints use that estimator's
    extent before full DTW. Five/three undefined expected edge positions are
    filled from the nearest valid level, as in the paired benchmark. Alignment
    uses squared standardized-current error and diagonal/left steps only.
    Unreachable paths and excessive traceback allocations raise ValueError;
    the method never silently switches to another chunking backend.
    """
    from porefm.data.chunk_dtw import warping_path_sloped_fast

    oriented, expected, start, end, trusted_start, trusted_end = _prepare_signal(
        raw, expected, ts, ns, stride, seq_type)
    estimated = not (trusted_start and trusted_end)
    if estimated:
        anchors = infer_boundaries(raw, expected, ts=ts, ns=ns, stride=stride, seq_type=seq_type)
        start, end = anchors.signal_start, anchors.signal_end
    x, y = normalize(oriented[start:end], expected)
    x = np.pad(x, (5, 3), mode="edge")
    path, cost = warping_path_sloped_fast(x, y, 1000)
    if not np.isfinite(cost) or len(path) != len(y):
        raise ValueError("No finite DTW path within the 1000-bin half-band")
    end = start + (end - start) // stride * stride
    if end <= start:
        raise ValueError("Inferred signal extent is empty")
    return DtwBoundaries(path[:, 0].copy(), start, end, int(stride), len(x),
                         dict(method="dtw", ts=ts, ns=ns, seq_type=seq_type,
                              trusted_start=trusted_start, trusted_end=trusted_end,
                              estimated_endpoints=estimated, mean_stride=6,
                              band_half_width=1000, cost=float(cost)))


def decode_inferred_signal(record, tags, raw, base_seq, exp_signal, kmer_size,
                           seq_type, expected_ts_dim, chunking_mode="auto"):
    """Validate eligibility, infer a map, and retain original ADC samples.

    Both methods support DNA/RNA 9-mer current models. Auto selects hierarchical
    for DNA and DTW for RNA; the caller has already handled any real move table.
    The legacy ``inferred`` spelling continues to select hierarchical anchoring.
    """
    if record.flag & 0x900:
        raise ValueError("Inferred chunking accepts primary/unmapped records only")
    if seq_type not in ("dna", "rna") or kmer_size != 9:
        raise ValueError("Inferred chunking supports DNA/RNA 9-mer models only; RNA002/custom 5-mer inputs require BAM mv tags and chunking_mode=mv (or auto with moves)")
    if np.any((base_seq < 0) | (base_seq > 3)):
        raise ValueError("Inferred chunking requires an A/C/G/T sequence")
    validate_chunking_mode(chunking_mode)
    if chunking_mode == "mv":
        raise ValueError("mv mode requires the move-based decoder")
    method = ("dtw" if seq_type == "rna" else "hierarchical") if chunking_mode == "auto" else chunking_mode
    estimate = infer_dtw_boundaries if method == "dtw" else infer_boundaries
    boundary = estimate(raw, exp_signal[5:-3], ts=tags.get("ts"),
                                ns=tags.get("ns"),
                                stride=6 if expected_ts_dim is None else expected_ts_dim,
                                seq_type=seq_type)
    oriented = raw[::-1] if seq_type == "rna" else raw
    mat = np.ascontiguousarray(oriented[boundary.signal_start:boundary.signal_end]).reshape(-1, boundary.stride)
    return mat, boundary

def normalize(raw, expected):
    """Affine quantile calibration without tags, followed by six-sample means.

    Fit 5th--95th quantiles on the central 90% of untrimmed raw signal, reducing
    adapter/tail influence without knowing sequence boundaries. Standardize
    both arrays in expected-level units. Mean bins start at raw sample zero.
    """
    n = len(raw)
    quantiles = np.linspace(.05, .95, 91)
    if n < 6 or len(expected) < 2 or not np.all(np.isfinite(expected)):
        raise ValueError("Insufficient or non-finite signal for inferred chunking")
    a = np.quantile(raw[n//20:n-n//20], quantiles)
    b = np.quantile(expected, quantiles)
    variance = np.dot(a-a.mean(), a-a.mean())
    if not np.all(np.isfinite(a)) or variance <= 0 or expected.std() <= 0:
        raise ValueError("Degenerate signal calibration for inferred chunking")
    scale = np.dot(a-a.mean(), b-b.mean()) / variance
    shift = b.mean() - scale*a.mean()
    mu, sd = expected.mean(), expected.std()
    x = np.ascontiguousarray((expected-mu)/sd)
    y = raw[:n//6*6].reshape(-1, 6).mean(axis=1)
    y = np.ascontiguousarray((y*scale+shift-mu)/sd)
    return x, y


@njit(cache=True)
def subsequence(x, y, reward):
    """Open-ended one-to-many DTW, returning each query base's mean-bin start.

    The local cost is squared standardized-level error minus a per-bin reward.
    Reward zero is ordinary SSE; a positive reward counters the preference for
    short, compressed matches. There are no vertical transitions/base skips.
    Both observed prefix and suffix are free. Ties prefer extending a base.
    This function is used only for short anchors, never whole-read alignment.
    """
    n, m = len(x), len(y)
    prev = np.empty(m)
    curr = np.empty(m)
    pointer = np.zeros((n, m), np.uint8)
    for j in range(m):
        cost = (x[0]-y[j])**2-reward
        if j > 0 and prev[j-1] < 0:
            prev[j] = prev[j-1]+cost
        else:
            prev[j] = cost
            pointer[0, j] = 1
    for i in range(1, n):
        curr[0] = np.inf
        for j in range(1, m):
            cost = (x[i]-y[j])**2-reward
            if prev[j-1] < curr[j-1]:
                curr[j] = prev[j-1]+cost
                pointer[i, j] = 1
            else:
                curr[j] = curr[j-1]+cost
        prev, curr = curr, prev
    end = np.argmin(prev)
    best_cost = prev[end]
    starts = np.empty(n+1, np.int64)
    starts[n] = end+1
    i, j = n-1, end
    while i >= 0 and j >= 0:
        if pointer[i, j]:
            starts[i] = j
            i -= 1
        j -= 1
    return starts, best_cost


def local_anchor(x, y, base, query, predicted, radius, reward, center_index=None):
    """Locate a short query and return its central signal/base knot and QC.

    ``base`` is an expected-array coordinate (original base coordinate minus
    five). The central knot reduces sensitivity to open query boundaries.
    Search coordinates and returned signal position are six-sample bins.
    """
    q0 = int(np.clip(round(base-query/2), 0, len(x)-query))
    lo = max(0, int(predicted-radius))
    hi = min(len(y), int(predicted+radius))
    if hi-lo < query:
        lo, hi = 0, len(y)
    starts, cost = subsequence(x[q0:q0+query], y[lo:hi], reward)
    center = query//2 if center_index is None else center_index
    t = float(starts[center]+lo)
    duration = int(starts[-1]-starts[0])
    return (float(q0+center), t, float(cost/query), duration/query)




def interpolate_extrapolate(query, knots, values):
    """Linear monotone interpolation with linear terminal extrapolation."""
    query = np.atleast_1d(query).astype(float)
    result = np.interp(query, knots, values)
    for mask, a, b in ((query<knots[0], 0, 1), (query>knots[-1], -2, -1)):
        result[mask] = values[a]+(query[mask]-knots[a])*(values[b]-values[a])/(knots[b]-knots[a])
    return result
