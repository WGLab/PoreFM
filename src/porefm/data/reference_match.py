"""Exact read/reference k-mer eligibility shared by aligned workflows."""

from __future__ import annotations

import numpy as np


def validate_match_kmer_size(kmer_size: int, *, allow_zero: bool = False) -> int:
    """Validate an odd centered-window size, optionally accepting zero as off."""
    if isinstance(kmer_size, bool):
        raise TypeError(f"kmer_size must be an integer, got {kmer_size!r}")
    kmer_size = int(kmer_size)
    if allow_zero and kmer_size == 0:
        return 0
    if kmer_size <= 0 or kmer_size % 2 != 1:
        qualifier = "zero or a positive odd integer" if allow_zero else "a positive odd integer"
        raise ValueError(f"kmer_size must be {qualifier}, got {kmer_size}")
    return kmer_size


def build_exact_match_kmer_arrays_from_read_arrays(
    base_seq,
    ref_pos_by_read,
    exact_by_read,
    *,
    k: int,
    quality_ok=None,
    eligible_center_mask=None,
):
    """Encode centered k-mers whose complete read window matches reference.

    Parameters
    ----------
    base_seq : array-like, shape (read_length,)
        Read-oriented zero-based base IDs; only IDs 0--3 are canonical.
    ref_pos_by_read : array-like, shape (read_length,)
        Zero-based reference coordinate for every read position, or a negative
        value when unaligned.
    exact_by_read : array-like of bool, shape (read_length,)
        Whether each read base is aligned and exactly equals the reference.
    k : int
        Positive odd centered-window size.
    quality_ok : array-like of bool, optional
        Additional per-base eligibility mask, such as a Phred threshold.
    eligible_center_mask : array-like of bool, optional
        Additional restriction applied only to candidate center positions.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Base-4 k-mer IDs (``-1`` when invalid) and a boolean center-validity
        mask. A valid window is entirely canonical, exact, quality-eligible,
        aligned, and contiguous on either reference strand. Consequently,
        windows crossing insertions, deletions, or splice skips are rejected.
    """
    k = validate_match_kmer_size(k)
    base_seq = np.asarray(base_seq)
    ref_pos_by_read = np.asarray(ref_pos_by_read)
    exact_by_read = np.asarray(exact_by_read, dtype=np.bool_)
    if base_seq.ndim != 1:
        raise ValueError(f"base_seq must be one-dimensional, got shape {base_seq.shape}")
    n = len(base_seq)
    if ref_pos_by_read.shape != (n,) or exact_by_read.shape != (n,):
        raise ValueError("Reference positions and exact-match flags must align with base_seq")

    if quality_ok is None:
        quality_ok = np.ones(n, dtype=np.bool_)
    else:
        quality_ok = np.asarray(quality_ok, dtype=np.bool_)
        if quality_ok.shape != (n,):
            raise ValueError("quality_ok must align with base_seq")
    if eligible_center_mask is not None:
        eligible_center_mask = np.asarray(eligible_center_mask, dtype=np.bool_)
        if eligible_center_mask.shape != (n,):
            raise ValueError("eligible_center_mask must align with base_seq")

    kmer_ids = np.full(n, -1, dtype=np.int64)
    kmer_valid = np.zeros(n, dtype=np.bool_)
    if n < k:
        return kmer_ids, kmer_valid

    canonical = (base_seq >= 0) & (base_seq < 4)
    base_eligible = canonical & exact_by_read & quality_ok & (ref_pos_by_read >= 0)
    window_count = n - k + 1
    eligible_windows = (
        np.convolve(base_eligible.astype(np.int16), np.ones(k, dtype=np.int16), mode="valid")
        == k
    )
    if k > 1:
        contiguous_steps = np.abs(np.diff(ref_pos_by_read.astype(np.int64, copy=False))) == 1
        contiguous_windows = (
            np.convolve(
                contiguous_steps.astype(np.int16),
                np.ones(k - 1, dtype=np.int16),
                mode="valid",
            )
            == k - 1
        )
        eligible_windows &= contiguous_windows

    half = k // 2
    centers = np.arange(window_count, dtype=np.int64) + half
    if eligible_center_mask is not None:
        eligible_windows &= eligible_center_mask[centers]

    base_windows = np.lib.stride_tricks.sliding_window_view(base_seq, k)
    weights = np.power(4, np.arange(k - 1, -1, -1), dtype=np.int64)
    encoded = base_windows.astype(np.int64, copy=False) @ weights
    valid_centers = centers[eligible_windows]
    kmer_ids[valid_centers] = encoded[eligible_windows]
    kmer_valid[valid_centers] = True
    return kmer_ids, kmer_valid


def build_md_exact_match_kmer_arrays(
    bam_read,
    oriented_base_seq,
    *,
    k: int,
    quality_ok=None,
    eligible_center_mask=None,
):
    """Encode exact-match k-mers from one BAM record's MD+CIGAR.

    ``oriented_base_seq`` must follow the repository's read-oriented sequence
    convention: reverse-aligning BAM sequences are reverse-complemented. BAM
    query coordinates are reflected to that orientation before applying the
    shared window rule. ``quality_ok`` applies a per-base filter, and
    ``eligible_center_mask`` restricts only candidate center positions.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Per-read base-4 k-mer IDs and their boolean validity mask, both in
        read-oriented coordinates.
    """
    k = validate_match_kmer_size(k)
    if bam_read.is_unmapped:
        raise ValueError(
            f"Read {bam_read.query_name!r} must be mapped for reference-match filtering"
        )
    if not bam_read.has_tag("MD"):
        raise ValueError(
            f"Read {bam_read.query_name!r} lacks the MD tag required for reference-match filtering"
        )
    if not bam_read.cigartuples:
        raise ValueError(
            f"Read {bam_read.query_name!r} lacks a CIGAR required for reference-match filtering"
        )

    query_sequence = bam_read.query_sequence
    if query_sequence is None:
        raise ValueError(f"Read {bam_read.query_name!r} lacks a query sequence")
    n = len(query_sequence)
    ref_pos_by_query = np.full(n, -1, dtype=np.int64)
    exact_by_query = np.zeros(n, dtype=np.bool_)
    try:
        aligned_pairs = bam_read.get_aligned_pairs(
            matches_only=False,
            with_seq=True,
        )
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(
            f"Could not decode MD+CIGAR for read {bam_read.query_name!r}"
        ) from exc

    for query_pos, ref_pos, ref_base in aligned_pairs:
        if query_pos is None or ref_pos is None or ref_base is None:
            continue
        query_base = query_sequence[int(query_pos)].upper()
        reference_base = str(ref_base).upper()
        ref_pos_by_query[int(query_pos)] = int(ref_pos)
        exact_by_query[int(query_pos)] = (
            query_base in "ACGT"
            and reference_base in "ACGT"
            and query_base == reference_base
        )

    if bam_read.is_reverse:
        ref_pos_by_query = ref_pos_by_query[::-1].copy()
        exact_by_query = exact_by_query[::-1].copy()

    oriented_base_seq = np.asarray(oriented_base_seq)
    if oriented_base_seq.shape != (n,):
        raise ValueError(
            f"Read-oriented base sequence has shape {oriented_base_seq.shape}, expected ({n},)"
        )
    return build_exact_match_kmer_arrays_from_read_arrays(
        oriented_base_seq,
        ref_pos_by_query,
        exact_by_query,
        k=k,
        quality_ok=quality_ok,
        eligible_center_mask=eligible_center_mask,
    )


def build_md_exact_match_kmer_mask(bam_read, oriented_base_seq, *, k: int):
    """Build the exact-window mask from one BAM record's MD+CIGAR."""
    _, valid = build_md_exact_match_kmer_arrays(
        bam_read,
        oriented_base_seq,
        k=k,
    )
    return valid
