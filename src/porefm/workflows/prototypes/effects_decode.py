"""CPU score packing and optional compiled effect-decoder dispatch."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _accelerator():
    """Load the optional dependency once; never suppress compilation failures."""
    try:
        from numba import config
        if config.DISABLE_JIT:
            return None
        from .effects_decode_numba import decode_arrays
    except ImportError:
        return None
    return decode_arrays


def decoder_backend(requested='auto'):
    """Resolve auto/numba/python without compiling or inspecting model inputs."""
    if requested not in ('auto', 'numba', 'python'):
        raise ValueError('effect decoder must be auto, numba, or python')
    if requested == 'python':
        return 'python'
    if _accelerator() is not None:
        return 'numba'
    if requested == 'numba':
        raise ImportError('Numba is unavailable or JIT is disabled; install porefm[effect-decoder], '
                          'enable JIT, or use --effect_decoder python')
    return 'python'


def warm_decoder(backend):
    """Compile/load the float32 kernel before the GPU/input pipeline starts."""
    if decoder_backend(backend) == 'numba':
        decode_arrays(np.zeros((1, 1), dtype=np.float32), np.ones(1, dtype=np.int64),
                      np.zeros((1, 2, 1, 1), dtype=np.float32),
                      np.full((1, 2), -1, dtype=np.int32), 1, 1)


def pack_potentials(unaries, interactions, radius):
    """Pack current sparse CPU potentials, preserving pair insertion order.

    Returns unaries[L,S], counts[L], pairs[L,2R,S,S], order[L,2R]. S is the
    largest local state count. The float dtype is unchanged; no padding is
    decoded. This adapter is for the CPU scorer; GPU score packets are already
    dense and bypass it entirely.
    """
    if radius < 0 or len(interactions) != len(unaries):
        raise ValueError('invalid effect radius or interaction length')
    length, history = len(unaries), 2 * radius
    counts = np.asarray([len(row) for row in unaries], dtype=np.int64)
    if (counts < 1).any():
        raise ValueError('every position needs at least one state')
    width = max(counts, default=1)
    dtype = unaries[0].detach().numpy().dtype if length else np.dtype('float32')
    values = np.zeros((length, width), dtype=dtype)
    pairs = np.zeros((length, history, width, width), dtype=dtype)
    order = np.full((length, history), -1, dtype=np.int32)
    for pos, (row, incoming) in enumerate(zip(unaries, interactions)):
        array = row.detach().numpy()
        if array.dtype != dtype:
            raise TypeError('compiled decoding requires one shared score dtype')
        values[pos, :counts[pos]] = array
        for edge, (previous, matrix) in enumerate(incoming.items()):
            if not max(0, pos - history) <= previous < pos:
                raise ValueError('effect interaction lies outside the earlier 2R frontier')
            array = matrix.detach().numpy()
            if array.dtype != dtype:
                raise TypeError('compiled decoding requires one shared score dtype')
            if array.shape != (counts[pos], counts[previous]):
                raise ValueError('effect pair state dimensions do not match unaries')
            distance_index = pos - previous - 1
            pairs[pos, distance_index, :counts[pos], :counts[previous]] = array
            order[pos, edge] = distance_index
    return values, counts, pairs, order


def packet_order(packet):
    """Describe active dense packet edges in the CPU builder's earlier-first order."""
    length, history = len(packet.counts), 2 * packet.radius
    called = packet.classes[:, 0].numpy() >= 0
    order = np.full((length, history), -1, dtype=np.int32)
    for column, distance in enumerate(range(history, 0, -1)):
        if distance < length:
            order[distance:, column] = np.where(called[distance:] & called[:-distance], distance - 1, -1)
    return order


def decode_arrays(unaries, counts, pairs, order, beam_width=32, max_sweeps=8):
    """Validate and decode finite float32/64 CPU arrays; never mutate inputs.

    Kernel arithmetic matches the reference's beam and conditional precision.
    Objective summation is finalized by Python to preserve the running Python
    version's built-in float-sum behavior. Compiled code releases the GIL.
    """
    if beam_width < 1 or max_sweeps < 1:
        raise ValueError('beam width and coordinate sweeps must be positive')
    if unaries.ndim != 2 or pairs.ndim != 4 or order.ndim != 2:
        raise ValueError('invalid dense effect score dimensions')
    length, width = unaries.shape
    history = order.shape[1]
    if (width < 1 or counts.shape != (length,) or order.shape != (length, history)
            or pairs.shape != (length, history, width, width)
            or (counts < 1).any() or (counts > width).any()):
        raise ValueError('inconsistent dense effect score dimensions/counts')
    if unaries.dtype not in (np.dtype('float32'), np.dtype('float64')) or pairs.dtype != unaries.dtype:
        raise TypeError('compiled decoding requires float32 or float64 scores with one shared dtype')
    if counts.dtype.kind not in 'iu' or order.dtype.kind not in 'iu':
        raise TypeError('effect state counts and edge order must be integers')
    if not np.isfinite(unaries).all() or not np.isfinite(pairs).all():
        raise ValueError('compiled effect scores must be finite')
    if ((order < -1) | (order >= history) | (order >= np.arange(length)[:, None])).any():
        raise ValueError('invalid effect edge distance')
    # A radius-zero or single-state read has only one live suffix regardless
    # of the requested beam width. Bound traceback by reachable suffixes, not
    # by an arbitrarily large user limit that the reference never fills.
    capacity = 1
    max_states = int(counts.max(initial=1))
    for _ in range(min(history, length)):
        capacity = min(beam_width, capacity * max_states)
        if capacity == beam_width:
            break
    if capacity > np.iinfo(np.int32).max:
        raise ValueError('effect beam exceeds the supported traceback index range')
    kernel = _accelerator()
    if kernel is None:
        raise ImportError('Numba is unavailable; use the Python decoder')
    states, conditional, converged, unary_values, pair_values = kernel(
        unaries, counts, pairs, order, capacity, max_sweeps)
    objective = sum(unary_values.tolist()) - sum(pair_values.tolist())
    return states, conditional, bool(converged), objective
