"""Stitch raw classifier predictions and encode multiclass MM/ML BAM tags."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class CheckpointPredictions:
    """Read-wise logits, probabilities, coordinates, and source BAM records.

    Array fields contain one entry per read. Variable-length list elements are
    aligned by base position and ``lengths`` records their row counts.
    ``logits`` and ``probs_raw`` have class dimension last.
    """
    readnames: np.ndarray
    refnames: np.ndarray
    strands: np.ndarray
    lengths: np.ndarray
    logits: List[np.ndarray]
    probs_raw: List[np.ndarray]
    read_pos: List[np.ndarray]
    refpos: List[np.ndarray]
    pysam_data: Dict[str, dict] = field(default_factory=dict)


def _unique_sorted_mask(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros(0, dtype=bool)

    mask = np.empty(arr.shape, dtype=bool)
    mask[0] = True
    mask[1:] = arr[1:] != arr[:-1]
    return mask


def stitch_readwise_arrays(metric_array, read_positions, ref_positions, pad_base_seq, readnames):
    """Stitch chunked per-base vectors into one deduplicated array per read.

    Overlapping chunks from the same read are split at the midpoint of the
    overlap. Chunks for each read must be contiguous and ordered by increasing
    read position. CPU tensors ``metric_array`` and the coordinate/mask arrays
    have shapes ``(chunks, bases, features)`` and ``(chunks, bases)`` respectively.
    Read and reference positions are zero-based; ``readnames`` identifies each
    chunk's source read.

    Returns
    -------
    tuple
        Boolean mask selecting the first chunk of each read, lists of stitched
        metric arrays, retained zero-based read positions, and corresponding
        reference positions. Rows containing any NaN metric component are
        omitted from all three per-read outputs.
    """
    read_wise_metric = []
    read_pos = []
    read_wise_refpos = []

    readnames = np.asarray(readnames)
    read_mask = _unique_sorted_mask(readnames)
    boundaries = np.where(read_mask[1:])[0] + 1
    chunks_per_read = np.split(np.arange(len(readnames)), boundaries)

    for read_indices in chunks_per_read:
        read_length = int(read_positions[read_indices].max().item()) + 1
        accum_metric = torch.full(
            (read_length, metric_array.size(2)),
            float("nan"),
            dtype=metric_array.dtype,
        )
        accum_ref = torch.empty((read_length,), dtype=ref_positions.dtype)

        last_written_pos = -1

        for idx in read_indices:
            mask = pad_base_seq[idx]

            curr_metric = metric_array[idx][mask]
            curr_pos = read_positions[idx][mask]
            curr_ref = ref_positions[idx][mask]

            if curr_pos.numel() == 0:
                continue

            start_pos = curr_pos[0].item()

            if start_pos <= last_written_pos:
                overlap_len = last_written_pos - start_pos + 1
                split_offset = (overlap_len + 1) // 2
            else:
                split_offset = 0

            write_metric = curr_metric[split_offset:]
            write_pos = curr_pos[split_offset:]
            write_ref = curr_ref[split_offset:]

            if write_pos.numel() > 0:
                write_start = write_pos[0].item()
                write_end = write_pos[-1].item() + 1

                accum_metric[write_start:write_end] = write_metric
                accum_ref[write_start:write_end] = write_ref

                last_written_pos = write_end - 1

        valid_label_rows = torch.where(~accum_metric.isnan().any(dim=1))[0]

        read_wise_metric.append(accum_metric[valid_label_rows])
        read_pos.append(valid_label_rows)
        read_wise_refpos.append(accum_ref[valid_label_rows])

    return read_mask, read_wise_metric, read_pos, read_wise_refpos


def collect_checkpoint_predictions(
    eval_data,
    temperature: float,
) -> CheckpointPredictions:
    """Stitch model batches into read-wise predictions for one checkpoint.

    ``eval_data`` entries contain chunk logits, coordinates, masks, read
    metadata, and optional pysam dictionaries in the eight-item inference
    tuple. Logits are divided by ``temperature`` before softmax. Arrays are
    returned on CPU with one variable-length array per read.
    """
    all_readnames, all_refnames, all_strands = [], [], []
    all_logits, all_probs_raw = [], []
    all_read_pos, all_refpos = [], []
    pysam_data = {}

    for data in eval_data:
        (
            total_logits,
            read_positions,
            ref_positions,
            pad_base_seq,
            readnames,
            refnames,
            strands,
            batch_pysam_data,
        ) = data

        read_mask, read_wise_logits, read_pos, read_wise_refpos = stitch_readwise_arrays(
            total_logits,
            read_positions,
            ref_positions,
            pad_base_seq,
            readnames,
        )

        final_readnames = np.asarray(readnames)[read_mask]
        all_readnames.append(final_readnames)
        all_refnames.append(np.asarray(refnames)[read_mask])
        all_strands.append(np.asarray(strands)[read_mask])

        for logits in read_wise_logits:
            probs = torch.softmax(logits / temperature, dim=-1).numpy()
            all_logits.append(logits.numpy())
            all_probs_raw.append(probs)

        all_read_pos.extend([x.numpy() for x in read_pos])
        all_refpos.extend([x.numpy() for x in read_wise_refpos])
        pysam_data.update(batch_pysam_data or {})

    if len(all_probs_raw) == 0:
        return CheckpointPredictions(
            readnames=np.asarray([]),
            refnames=np.asarray([]),
            strands=np.asarray([]),
            lengths=np.asarray([], dtype=np.int32),
            logits=[],
            probs_raw=[],
            read_pos=[],
            refpos=[],
            pysam_data=pysam_data,
        )

    readnames = np.concatenate(all_readnames)
    refnames = np.concatenate(all_refnames)
    strands = np.concatenate(all_strands)
    lengths = np.fromiter((len(x) for x in all_read_pos), dtype=np.int32, count=len(all_read_pos))

    return CheckpointPredictions(
        readnames=readnames,
        refnames=refnames,
        strands=strands,
        lengths=lengths,
        logits=all_logits,
        probs_raw=all_probs_raw,
        read_pos=all_read_pos,
        refpos=all_refpos,
        pysam_data=pysam_data,
    )


def _candidate_base_positions(seq: str, canonical_base: str) -> np.ndarray:
    base = canonical_base.upper()
    seq = seq.upper()
    if base == "U" and "U" not in seq:
        base = "T"
    seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    if base == "N":
        return np.arange(len(seq), dtype=np.int64)
    return np.flatnonzero(seq_bytes == ord(base)).astype(np.int64)


def make_mm_ml_tags(
    seq: str,
    base: str,
    coords,
    probs,
    seq_type: str,
    mod_codes: List[str],
    strand: str = "+",
    skipped_flag: Optional[str] = "?",
) -> Tuple[str, List[int]]:
    """Encode multiclass probabilities as one MM group per modification code.

    ``coords`` are zero-based read positions and ``probs`` has shape
    ``(N, len(mod_codes)+1)`` with canonical probability in column zero. Invalid
    coordinates or positions not matching ``base`` are skipped, remaining
    positions are stably sorted, and each mod column is quantized to ML bytes.
    The returned MM string concatenates groups in ``mod_codes`` order and the
    ML list uses the same group-major order. ``seq_type`` is validated for
    caller consistency; T/U candidate matching is handled by the sequence.
    """
    seq = seq.upper()
    base = base.upper()
    seq_type = seq_type.lower()

    if base not in {"A", "C", "G", "T", "U", "N"}:
        raise ValueError("base must be one of A,C,G,T,U,N")
    if seq_type not in {"dna", "rna"}:
        raise ValueError("seq_type must be 'dna' or 'rna'")
    if strand not in {"+", "-"}:
        raise ValueError("strand must be '+' or '-'")
    if skipped_flag not in {"?", ".", None, ""}:
        raise ValueError("skipped_flag must be '?', '.', or None")

    coords = np.asarray(coords, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != len(mod_codes) + 1:
        raise ValueError(f"probs must have shape (N, {len(mod_codes) + 1})")
    if len(coords) != len(probs):
        raise ValueError("coords length must match probs length")

    flag = "" if skipped_flag in (None, "") else skipped_flag
    empty_mm = "".join([f"{base}{strand}{mc}{flag};" for mc in mod_codes])
    if len(coords) == 0:
        return empty_mm, []

    in_range = (coords >= 0) & (coords < len(seq))
    coords = coords[in_range]
    probs = probs[in_range]
    if len(coords) == 0:
        return empty_mm, []

    order = np.argsort(coords, kind="stable")
    coords = coords[order]
    probs = probs[order]

    base_positions = _candidate_base_positions(seq, base)
    if len(base_positions) == 0:
        return empty_mm, []

    ranks_all = np.searchsorted(base_positions, coords)
    in_bounds = ranks_all < len(base_positions)
    valid = np.zeros(len(coords), dtype=bool)
    if np.any(in_bounds):
        valid[in_bounds] = base_positions[ranks_all[in_bounds]] == coords[in_bounds]

    coords = coords[valid]
    probs = probs[valid]
    ranks = ranks_all[valid]

    if len(coords) == 0:
        return empty_mm, []

    deltas = np.diff(np.concatenate(([-1], ranks))) - 1
    deltas_str = ",".join(map(str, deltas.tolist()))
    mm_parts = []
    ml_all: List[int] = []

    for i, mod_code in enumerate(mod_codes):
        mm_parts.append(f"{base}{strand}{mod_code}{flag},{deltas_str};")
        mod_probs = probs[:, i + 1]
        ml = np.minimum(
            (np.clip(mod_probs, 0.0, 1.0) * 256.0).astype(np.int64),
            255,
        ).astype(np.uint8)
        ml_all.extend(ml.tolist())

    return "".join(mm_parts), ml_all
