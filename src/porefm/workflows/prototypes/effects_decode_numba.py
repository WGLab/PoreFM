"""Compiled finite-range beam decoding; no CUDA, model fitting, or shared state.

Kernels deliberately disable fast math. Beam arithmetic uses float64, while
conditional subtraction rounds after each edge in the input score dtype. The
module is imported lazily so Numba remains an optional inference accelerator.
"""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True, fastmath=False, inline='always')
def _conditional(pos, states, unaries, counts, pairs, order, active, output):
    """Subtract earlier edges in insertion order, then later edges by position."""
    length, history = order.shape
    for state in range(counts[pos]):
        output[state] = unaries[pos, state]
    for edge in range(history):
        distance_index = order[pos, edge]
        if distance_index >= 0:
            previous = pos - distance_index - 1
            for state in range(counts[pos]):
                output[state] -= pairs[pos, distance_index, state, states[previous]]
    for later in range(pos + 1, min(length, pos + history + 1)):
        distance_index = later - pos - 1
        if active[later, distance_index]:
            for state in range(counts[pos]):
                output[state] -= pairs[later, distance_index, states[later], state]


@njit(cache=True, nogil=True, fastmath=False)
def decode_arrays(unaries, counts, pairs, order, beam_width, max_sweeps):
    """Decode owned CPU arrays with stable suffix merging and exact traceback.

    Unaries are (L,S), counts (L,), pairs (L,2R,S,S), and order (L,2R).
    Each nonnegative order entry is distance-1; -1 denotes an absent edge.
    Padding is not a state. All inputs must be finite, dimensionally valid,
    float32/float64 arrays; the Python adapter validates these preconditions.

    Parent beams sharing their last 2R-1 states share extension keys. Grouping
    those parents preserves dictionary first-insertion order without Python
    tuples, integer-packed suffix overflow, or quadratic-length path copies.
    Traceback storage is O(L*beam_width); other beam storage is O(beam_width*S
    + beam_width*R). Conditional polishing retains the original sweep order,
    first-state tie breaking, and strict improvement threshold of 1e-6.
    """
    length, width = unaries.shape
    history = order.shape[1]
    beam_scores = np.zeros(beam_width, dtype=np.float64)
    suffix = np.full((beam_width, history), -1, dtype=np.int32)
    next_suffix = np.empty_like(suffix)
    next_scores = np.empty_like(beam_scores)
    trace_parent = np.empty((length, beam_width), dtype=np.int32)
    trace_state = np.empty((length, beam_width), dtype=np.int32)
    groups = np.empty(beam_width, dtype=np.int32)
    representatives = np.empty(beam_width, dtype=np.int32)
    candidate_scores = np.empty(beam_width * width, dtype=np.float64)
    candidate_parent = np.empty(beam_width * width, dtype=np.int32)
    candidate_state = np.empty(beam_width * width, dtype=np.int32)
    nbeams = 1
    for pos in range(length):
        ngroups = 0
        for parent in range(nbeams):
            group = -1
            for existing in range(ngroups):
                representative = representatives[existing]
                equal = True
                for column in range(1, history):
                    if suffix[parent, column] != suffix[representative, column]:
                        equal = False
                        break
                if equal:
                    group = existing
                    break
            if group < 0:
                group = ngroups
                representatives[group] = parent
                ngroups += 1
            groups[parent] = group
        ncandidates = ngroups * counts[pos] if history else 1
        candidate_parent[:ncandidates] = -1
        for parent in range(nbeams):
            for state in range(counts[pos]):
                score = beam_scores[parent] + np.float64(unaries[pos, state])
                for edge in range(history):
                    distance_index = order[pos, edge]
                    if distance_index >= 0:
                        old_state = suffix[parent, history - distance_index - 1]
                        score -= np.float64(pairs[pos, distance_index, state, old_state])
                slot = groups[parent] * counts[pos] + state if history else 0
                if candidate_parent[slot] < 0 or score > candidate_scores[slot]:
                    candidate_scores[slot] = score
                    candidate_parent[slot] = parent
                    candidate_state[slot] = state
        # Stable sorting is essential: equal-scoring keys retain first insertion
        # order even if a later parent replaced the score for one of those keys.
        ranking = np.argsort(-candidate_scores[:ncandidates], kind='mergesort')
        nbeams = min(beam_width, ncandidates)
        for index in range(nbeams):
            slot = ranking[index]
            parent, state = candidate_parent[slot], candidate_state[slot]
            next_scores[index] = candidate_scores[slot]
            for column in range(history - 1):
                next_suffix[index, column] = suffix[parent, column + 1]
            if history:
                next_suffix[index, history - 1] = state
            trace_parent[pos, index] = parent
            trace_state[pos, index] = state
        beam_scores, next_scores = next_scores, beam_scores
        suffix, next_suffix = next_suffix, suffix
    states = np.empty(length, dtype=np.int64)
    node = 0
    for pos in range(length - 1, -1, -1):
        states[pos] = trace_state[pos, node]
        node = trace_parent[pos, node]
    active = np.zeros((length, history), dtype=np.bool_)
    nedges = 0
    for pos in range(length):
        for edge in range(history):
            distance_index = order[pos, edge]
            if distance_index >= 0:
                active[pos, distance_index] = True
                nedges += 1
    temporary = np.empty(width, dtype=unaries.dtype)
    improvement = np.empty(1, dtype=unaries.dtype)
    converged = False
    for sweep in range(max_sweeps):
        changed = False
        for step in range(length):
            pos = step if sweep % 2 == 0 else length - step - 1
            _conditional(pos, states, unaries, counts, pairs, order, active, temporary)
            best = 0
            for state in range(1, counts[pos]):
                if temporary[state] > temporary[best]:
                    best = state
            improvement[0] = temporary[best] - temporary[states[pos]]
            if np.float64(improvement[0]) > 1e-6:
                states[pos] = best
                changed = True
        if not changed:
            converged = True
            break
    conditional = np.zeros_like(unaries)
    unary_values = np.empty(length, dtype=np.float64)
    pair_values = np.empty(nedges, dtype=np.float64)
    edge_index = 0
    for pos in range(length):
        _conditional(pos, states, unaries, counts, pairs, order, active, conditional[pos])
        unary_values[pos] = unaries[pos, states[pos]]
        for edge in range(history):
            distance_index = order[pos, edge]
            if distance_index >= 0:
                previous = pos - distance_index - 1
                pair_values[edge_index] = pairs[pos, distance_index, states[pos], states[previous]]
                edge_index += 1
    # The adapter uses Python's built-in sum, preserving its version-specific
    # float summation (including Python 3.12's compensated summation).
    return states, conditional, converged, unary_values, pair_values
