"""Detect deviations at labelled read positions using a calibrated canonical reference."""

from porefm.preprocessing import validate_artifact_preprocessing
from porefm.cli.preprocessing_args import add_preprocessing_arguments, resolve_cli_preprocessing
from porefm.cli.chunking_args import add_chunking_arguments, resolve_chunking_mode
import datetime
import os
import sys
import time
from functools import partial

from porefm.cli._parser import PoreFMArgumentParser
from porefm.cli.model_args import add_model_arguments, resolve_model_arguments
from porefm.common.chunking import resolve_downstream_step
from porefm.common.model_bundle import apply_bundle_arguments, use_model_bundle
import porefm.workflows.prototypes.utils as shared
np = None
pa = None
pq = None
torch = None
DataLoader = None
load_canonical_null_checkpoint = None
load_canonical_calibration = None
prepare_canonical_calibration = None
lookup_canonical_pvalues = None
normalize_score_type = None
required_scale_signal_overlap = None
requires_rna_structure_features = None
RAW_COSINE_SCORE_TYPE = None
RAW_WHITENING_SCORE_TYPE = None


def lazy_import_runtime_dependencies():
    """Import tensor, parquet, and canonical-null dependencies on first use.

    The CLI defers these relatively expensive imports so argument parsing and
    ``--help`` do not initialize the PyTorch runtime. Imported objects are
    cached in module globals and subsequent calls are no-ops.
    """
    global np, pa, pq, torch, DataLoader
    global load_canonical_null_checkpoint
    global load_canonical_calibration
    global prepare_canonical_calibration, lookup_canonical_pvalues
    global normalize_score_type
    global required_scale_signal_overlap
    global requires_rna_structure_features
    global RAW_COSINE_SCORE_TYPE, RAW_WHITENING_SCORE_TYPE

    if torch is not None:
        return

    shared.lazy_import_runtime_dependencies()
    from porefm.workflows.canonical_null.predictor import (
        RAW_COSINE_SCORE_TYPE as _RAW_COSINE_SCORE_TYPE,
        RAW_WHITENING_SCORE_TYPE as _RAW_WHITENING_SCORE_TYPE,
        load_canonical_calibration as _load_canonical_calibration,
        load_canonical_null_checkpoint as _load_canonical_null_checkpoint,
        lookup_canonical_pvalues as _lookup_canonical_pvalues,
        required_scale_signal_overlap as _required_scale_signal_overlap,
        requires_rna_structure_features as _requires_rna_structure_features,
        normalize_score_type as _normalize_score_type,
        prepare_canonical_calibration as _prepare_canonical_calibration,
    )

    np = shared.np
    pa = shared.pa
    pq = shared.pq
    torch = shared.torch
    DataLoader = shared.DataLoader
    load_canonical_null_checkpoint = _load_canonical_null_checkpoint
    load_canonical_calibration = _load_canonical_calibration
    prepare_canonical_calibration = _prepare_canonical_calibration
    lookup_canonical_pvalues = _lookup_canonical_pvalues
    required_scale_signal_overlap = _required_scale_signal_overlap
    requires_rna_structure_features = _requires_rna_structure_features
    normalize_score_type = _normalize_score_type
    RAW_COSINE_SCORE_TYPE = _RAW_COSINE_SCORE_TYPE
    RAW_WHITENING_SCORE_TYPE = _RAW_WHITENING_SCORE_TYPE


def load_canonical_null_model(
    model_path,
    device,
    whitening_artifact_path=None,
    seq_type=None,
):
    """Restore a unified canonical-null model and its checkpoint metadata.

    Returns
    -------
    tuple
        ``(model, backbone_config, metadata)`` with the model in evaluation
        mode. Optional whitening and sequence-type arguments are validated by
        :func:`load_canonical_null_checkpoint`.
    """
    model, checkpoint = load_canonical_null_checkpoint(
        model_path,
        device,
        whitening_artifact_path=whitening_artifact_path,
        expected_whitening_seq_type=seq_type,
    )
    scale = getattr(model, 'scale_model', None)
    null = getattr(model, 'null_predictor', None)
    if (any(m is not None and getattr(m.config, 'use_rna_structure_features', False) for m in (null, scale))
            and seq_type is not None and seq_type != 'rna'):
        raise ValueError('A null/scale checkpoint with RNA structure features requires seq_type rna')
    model.eval()
    return model, model.backbone.config, checkpoint.get('meta', {})


def compute_distance_batch_canonical_null(
    batch,
    canonical_model,
    calibration,
    kmer_size,
    device,
    num_trim,
    batch_size,
    score_type=None,
    include_sequence=False,
):
    """Compute calibrated canonical-null scores for one complete-read batch.

    The canonical collator appends precomputed (B,L,109) RNA structure inputs
    (or None) to the complete-read batch. Each GPU sub-batch takes the same
    slice from these CPU features; no folding occurs during model execution.

    ``score_type`` chooses raw cosine or whitening distance. The corresponding
    normalized model score is converted to empirical p-values by k-mer. Only
    labelled, unpadded, edge-trimmed positions with complete k-mers and finite
    calibration results are retained; all other floating outputs are ``NaN``.
    When ``include_sequence`` is true, also return the original CPU model-input
    tokens as ``base_seq_input`` for output context, including unscored bases.
    This does not change model execution or scoring eligibility.

    Returns
    -------
    dict
        CPU tensors and metadata matching the batch layout. Core entries are
        ``distance``, normalized ``score``, ``pvalue``, calibration count,
        k-mer IDs, positions, masks, and identifiers. Available explicit
        cosine/whitening scores are included as additional keys.
    """
    (
        base_seq_input,
        ts_input,
        percentiles_input,
        pad_base_seq,
        pad_ts,
        labels_idx,
        readnames,
        refnames,
        ref_positions,
        read_positions,
        _min_start,
        strands,
        _pysam_data,
        base_quality,
        global_features,
        read_lengths,
        rna_structure_features,
    ) = batch

    device = torch.device(device)
    score_type = normalize_score_type(score_type or RAW_COSINE_SCORE_TYPE)
    # The scale CNN's receptive field is accommodated by chunk overlap. Keep
    # the caller's trim as the eligibility policy so midpoint stitching can
    # retain boundary-aware predictions at true read ends.
    eligibility_num_trim = int(num_trim)
    distance = torch.full(base_seq_input.shape, float('nan'), dtype=torch.float32)
    score = torch.full(base_seq_input.shape, float('nan'), dtype=torch.float32)
    pvalue = torch.full(base_seq_input.shape, float('nan'), dtype=torch.float32)
    calibration_count = torch.zeros(base_seq_input.shape, dtype=torch.long)
    explicit_scores = {
        's_cos': torch.full(base_seq_input.shape, float('nan'), dtype=torch.float32),
        'z_cos': torch.full(base_seq_input.shape, float('nan'), dtype=torch.float32),
        's_whiten': torch.full(base_seq_input.shape, float('nan'), dtype=torch.float32),
        'z_whiten': torch.full(base_seq_input.shape, float('nan'), dtype=torch.float32),
    }
    kmer_ids_cpu = torch.full(base_seq_input.shape, -1, dtype=torch.long)

    for sub_batch in range(0, base_seq_input.size(0), batch_size):
        sl = slice(sub_batch, sub_batch + batch_size)
        batch_base_seq_input = base_seq_input[sl].to(device, non_blocking=True)
        batch_ts_input = ts_input[sl].to(device, non_blocking=True)
        batch_percentiles_input = percentiles_input[sl].to(device, non_blocking=True)
        batch_pad_base_seq = pad_base_seq[sl].to(device, non_blocking=True)
        batch_pad_ts = pad_ts[sl].to(device, non_blocking=True)
        batch_labels_idx = labels_idx[sl].to(device, non_blocking=True)
        batch_base_quality = (
            base_quality[sl].to(device, non_blocking=True)
            if base_quality is not None
            else None
        )
        batch_global_features = (
            global_features[sl].to(device, non_blocking=True)
            if global_features is not None
            else None
        )
        batch_rna_structure_features = (
            rna_structure_features[sl].to(device, non_blocking=True)
            if rna_structure_features is not None else None
        )

        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == 'cuda',
        ):
            out = canonical_model(
                batch_base_seq_input,
                batch_ts_input,
                percentiles_input=batch_percentiles_input,
                pad_base_seq=batch_pad_base_seq,
                pad_ts=batch_pad_ts,
                embeddings_only=True,
                base_quality=batch_base_quality,
                global_features=batch_global_features,
                read_positions=read_positions[sl],
                read_lengths=read_lengths[sl],
                rna_structure_features=batch_rna_structure_features,
            )
            if score_type == RAW_WHITENING_SCORE_TYPE:
                if 's_whiten' not in out:
                    raise ValueError("--score_type whitening requires --whitening_artifact")
                batch_distance = out['s_whiten'].float()
                batch_score = out.get('z_whiten', batch_distance).float()
            else:
                batch_score = out['canonical_null_score'].float()
                batch_distance = out.get('canonical_null_score_raw', batch_score).float()

        kmer_ids, kmer_valid = shared.compute_chunk_kmer_ids(
            batch_base_seq_input,
            kmer_size,
        )
        valid_pos = (
            shared.ignore_edges(batch_pad_base_seq.bool(), eligibility_num_trim)
            & batch_labels_idx.bool()
            & kmer_valid
        )
        batch_pvalue, batch_calibration_count = lookup_canonical_pvalues(
            batch_score,
            kmer_ids,
            calibration,
        )
        valid_pos = valid_pos & torch.isfinite(batch_pvalue)

        batch_distance_out = torch.full(
            batch_base_seq_input.shape,
            float('nan'),
            dtype=torch.float32,
            device=device,
        )
        batch_score_out = torch.full_like(batch_distance_out, float('nan'))
        batch_pvalue_out = torch.full_like(batch_distance_out, float('nan'))
        batch_count_out = torch.zeros(
            batch_base_seq_input.shape,
            dtype=torch.long,
            device=device,
        )
        batch_explicit_out = {
            key: torch.full_like(batch_distance_out, float('nan'))
            for key in explicit_scores
        }
        if valid_pos.any():
            batch_distance_out[valid_pos] = batch_distance[valid_pos].float()
            batch_score_out[valid_pos] = batch_score[valid_pos].float()
            batch_pvalue_out[valid_pos] = batch_pvalue[valid_pos].float()
            batch_count_out[valid_pos] = batch_calibration_count[valid_pos].long()
            for key in batch_explicit_out:
                if key in out:
                    batch_explicit_out[key][valid_pos] = out[key][valid_pos].float()

        distance[sl] = batch_distance_out.cpu()
        score[sl] = batch_score_out.cpu()
        pvalue[sl] = batch_pvalue_out.cpu()
        calibration_count[sl] = batch_count_out.cpu()
        for key, values in batch_explicit_out.items():
            explicit_scores[key][sl] = values.cpu()
        kmer_ids_cpu[sl] = kmer_ids.cpu()

    result = {
        'distance': distance,
        'score': score,
        'pvalue': pvalue,
        'calibration_count': calibration_count,
        'kmer_ids': kmer_ids_cpu,
        'read_positions': read_positions.cpu(),
        'ref_positions': ref_positions.cpu(),
        'pad_base_seq': pad_base_seq.cpu(),
        'labels_idx': labels_idx.cpu(),
        'readnames': readnames,
        'refnames': refnames,
        'strands': strands,
    }
    if include_sequence:
        result['base_seq_input'] = base_seq_input.cpu()
    for key, values in explicit_scores.items():
        if torch.isfinite(values).any():
            result[key] = values
    return result


def validate_output_filters(min_metric, max_pvalue, peak_only, prune_radius, *, calibrated):
    """Reject inconsistent output filters without importing runtime packages."""
    if max_pvalue is not None:
        if not 0.0 <= max_pvalue <= 1.0:
            raise ValueError('--max_pvalue must be finite and between 0 and 1 inclusive')
        if min_metric is not None:
            raise ValueError('--max_pvalue cannot be combined with --min_metric')
    if (max_pvalue is not None or peak_only) and not calibrated:
        raise ValueError('P-value filtering and --peak_only require calibrated canonical-null results')
    if prune_radius < 0:
        raise ValueError('--prune_radius must be nonnegative')
    if prune_radius and not peak_only:
        raise ValueError('--prune_radius requires --peak_only')


def stitch_batch_rows(
    batch_result, kmer_size, min_metric=None, *, max_pvalue=None,
    peak_only=False, prune_radius=0, output_context_radius=0,
    output_context_format='packed',
):
    """Stitch overlapping chunks into non-duplicated per-read score rows.

    Overlaps are split at their midpoint, with the later chunk replacing the
    second half. Rows are then restricted to labelled positions whose k-mer is
    valid and whose selected output threshold passes. Calibrated results
    always require finite p-values. Optional peak selection uses the original
    stitched profile before thresholding, independently for each read.

    Parameters
    ----------
    batch_result : dict
        Output of compute_distance_batch_canonical_null.
    kmer_size : int
        Length used to decode integer k-mer IDs.
    min_metric : float or None, default=None
        Strict lower bound on emitted raw distances; defaults to 0.0 when
        neither threshold is supplied. Mutually exclusive with max_pvalue.
    max_pvalue : float or None, default=None
        Inclusive p-value upper bound in [0, 1], replacing the distance cutoff.
    peak_only : bool, default=False
        Keep local p-value minima, using plateau midpoints rounded down.
        Requires calibrated all-base results; the CLI enforces all-base mode.
        Missing/invalid positions break the profile. Endpoints are not peaks.
    prune_radius : int, default=0
        Suppress weaker qualifying peaks within this inclusive read-base
        radius. Zero disables pruning; positive values require peak_only.
    output_context_radius : int, default=0
        Positive values add model-input sequence_context (2R+1 bases) and
        context_complete columns. Requires base_seq_input in batch_result.
        Unavailable flanks/gaps are N-padded; observed N counts as available.
        Context includes unscored/unlabelled bases. Zero disables this output.
    output_context_format : {'packed', 'text'}, default='packed'
        Lossless two-bit binary with an optional N mask, or readable strings.

    Returns
    -------
    dict[str, list]
        Column-oriented rows ready for parquet serialization.
    """
    validate_output_filters(
        min_metric, max_pvalue, peak_only, prune_radius,
        calibrated=batch_result.get('pvalue') is not None,
    )
    if output_context_radius < 0:
        raise ValueError('--output_context_radius must be nonnegative')
    if output_context_format not in ('packed', 'text'):
        raise ValueError("--output_context_format must be 'packed' or 'text'")
    include_context = output_context_radius > 0
    base_seq_input = batch_result.get('base_seq_input') if include_context else None
    if include_context and base_seq_input is None:
        raise ValueError('Output sequence context requires base_seq_input in batch_result')
    if include_context:
        from porefm.workflows.canonical_null.sequence_context import encode_sequence_contexts
    if peak_only:
        from porefm.workflows.canonical_null.peak_filter import select_pvalue_peaks
    if min_metric is None and max_pvalue is None:
        min_metric = 0.0
    rows = empty_rows(
        include_calibration='pvalue' in batch_result, include_context=include_context,
    )
    explicit_score_keys = [
        key
        for key in ('s_cos', 'z_cos', 's_whiten', 'z_whiten')
        if key in batch_result
    ]
    for key in explicit_score_keys:
        rows[key] = []

    readnames = np.asarray(batch_result['readnames'])
    refnames = np.asarray(batch_result['refnames'])
    strands = np.asarray(batch_result['strands'])
    _, chunks_per_read = shared.unique_read_boundaries(readnames)

    distance = batch_result['distance']
    score = batch_result.get('score')
    pvalue = batch_result.get('pvalue')
    calibration_count = batch_result.get('calibration_count')
    explicit_scores = {key: batch_result[key] for key in explicit_score_keys}
    kmer_ids = batch_result['kmer_ids']
    read_positions = batch_result['read_positions']
    ref_positions = batch_result['ref_positions']
    pad_base_seq = batch_result['pad_base_seq']
    labels_idx = batch_result['labels_idx']
    has_pvalue = pvalue is not None

    for read_indices in chunks_per_read:
        read_length = int(read_positions[read_indices].max().item()) + 1
        accum_distance = torch.full((read_length,), float('nan'), dtype=torch.float32)
        accum_score = (
            torch.full((read_length,), float('nan'), dtype=torch.float32)
            if has_pvalue
            else None
        )
        accum_pvalue = (
            torch.full((read_length,), float('nan'), dtype=torch.float32)
            if has_pvalue
            else None
        )
        accum_calibration_count = (
            torch.zeros((read_length,), dtype=torch.long) if has_pvalue else None
        )
        accum_explicit = {
            key: torch.full((read_length,), float('nan'), dtype=torch.float32)
            for key in explicit_score_keys
        }
        accum_ref = torch.full((read_length,), -1, dtype=ref_positions.dtype)
        accum_kmer = torch.full((read_length,), -1, dtype=torch.long)
        accum_label = torch.zeros((read_length,), dtype=torch.bool)
        accum_sequence = (
            torch.zeros((read_length,), dtype=torch.uint8) if include_context else None
        )

        last_written_pos = -1
        for idx in read_indices:
            mask = pad_base_seq[idx]
            curr_pos = read_positions[idx][mask]
            if curr_pos.numel() == 0:
                continue
            start_pos = curr_pos[0].item()
            if start_pos <= last_written_pos:
                overlap_len = last_written_pos - start_pos + 1
                split_offset = (overlap_len + 1) // 2
            else:
                split_offset = 0
            write_pos = curr_pos[split_offset:]
            if write_pos.numel() == 0:
                continue
            write_start = write_pos[0].item()
            write_end = write_pos[-1].item() + 1
            chunk_masked = mask.nonzero(as_tuple=True)[0][split_offset:]

            accum_distance[write_start:write_end] = distance[idx][chunk_masked]
            if has_pvalue:
                accum_score[write_start:write_end] = score[idx][chunk_masked]
                accum_pvalue[write_start:write_end] = pvalue[idx][chunk_masked]
                accum_calibration_count[write_start:write_end] = calibration_count[idx][chunk_masked]
            for key in explicit_score_keys:
                accum_explicit[key][write_start:write_end] = explicit_scores[key][idx][chunk_masked]
            accum_ref[write_start:write_end] = ref_positions[idx][chunk_masked]
            accum_kmer[write_start:write_end] = kmer_ids[idx][chunk_masked]
            accum_label[write_start:write_end] = labels_idx[idx][chunk_masked]
            if include_context:
                # Use the unpadded sequence, not the score/label/output mask:
                # an unscored neighboring base is still valid sequence context.
                accum_sequence[write_start:write_end] = base_seq_input[idx][chunk_masked]
            last_written_pos = write_end - 1

        valid = accum_label & (accum_kmer >= 0)
        keep = valid
        if has_pvalue:
            keep = keep & torch.isfinite(accum_pvalue)
        if max_pvalue is not None:
            keep = keep & (accum_pvalue <= max_pvalue)
        else:
            keep = keep & (accum_distance > min_metric)

        if peak_only:
            # NumPy views avoid copying the stitched CPU tensors. Thresholds
            # affect candidate eligibility, not the shape of the peak profile.
            kept_read_pos = torch.from_numpy(select_pvalue_peaks(
                accum_pvalue.numpy(), valid.numpy(), keep.numpy(), prune_radius,
            ))
            if kept_read_pos.numel() == 0:
                continue
            # Gather only selected positions in each column, rather than
            # repeatedly scanning a full-read boolean mask for sparse peaks.
            keep = kept_read_pos
        else:
            if not keep.any():
                continue
            kept_read_pos = keep.nonzero(as_tuple=True)[0]

        read_idx = read_indices[0]
        kept_kmers = accum_kmer[keep].numpy()
        rows['readname'].extend([str(readnames[read_idx])] * len(kept_read_pos))
        rows['chrom'].extend([str(refnames[read_idx])] * len(kept_read_pos))
        rows['strand'].extend([int(strands[read_idx])] * len(kept_read_pos))
        rows['read_pos'].extend(kept_read_pos.numpy().astype(np.int64).tolist())
        rows['position'].extend(accum_ref[keep].numpy().astype(np.int64).tolist())
        rows['kmer'].extend(
            [shared.kmer_index_to_string(kmer_id, kmer_size) for kmer_id in kept_kmers]
        )
        if include_context:
            contexts, complete = encode_sequence_contexts(
                accum_sequence.numpy(), kept_read_pos.numpy(), output_context_radius,
                output_context_format,
            )
            rows['sequence_context'].extend(contexts)
            rows['context_complete'].extend(complete)
        rows['distance'].extend(accum_distance[keep].numpy().astype(np.float32).tolist())
        if has_pvalue:
            rows['score'].extend(accum_score[keep].numpy().astype(np.float32).tolist())
            rows['pvalue'].extend(accum_pvalue[keep].numpy().astype(np.float32).tolist())
            rows['calibration_count'].extend(
                accum_calibration_count[keep].numpy().astype(np.int64).tolist()
            )
        for key in explicit_score_keys:
            rows[key].extend(accum_explicit[key][keep].numpy().astype(np.float32).tolist())

    return rows


def append_rows(dst, src):
    """Append column-oriented ``src`` rows to ``dst`` in place."""
    for key, values in src.items():
        if key not in dst:
            dst[key] = []
        dst[key].extend(values)


def empty_rows(include_calibration=False, include_context=False):
    """Create score columns, optionally including calibration and sequence context."""
    rows = {
        'readname': [],
        'chrom': [],
        'strand': [],
        'read_pos': [],
        'position': [],
        'kmer': [],
        'distance': [],
    }
    if include_calibration:
        rows['score'] = []
        rows['pvalue'] = []
        rows['calibration_count'] = []
    if include_context:
        rows['sequence_context'] = []
        rows['context_complete'] = []
    return rows


def write_rows(rows, output_dir, part_idx, *, output_context_radius=0, output_context_format='packed'):
    """Write paired metadata/data Parquets, returning the data path or None.

    Scores use lossless Zstandard level 1 with float byte-stream splitting.
    Optional context uses level 3 and dictionaries only for repeated contexts.
    Context radius/format must match the rows. Empty input returns None.
    """
    if len(rows['distance']) == 0:
        return None
    os.makedirs(output_dir, exist_ok=True)
    columns = {
        'readname': pa.array(rows['readname'], pa.string()),
        'chrom': pa.array(rows['chrom'], pa.dictionary(pa.int32(), pa.string())),
        'strand': pa.array(rows['strand'], pa.int8()),
        'read_pos': pa.array(rows['read_pos'], pa.int64()),
        'position': pa.array(rows['position'], pa.int64()),
        'kmer': pa.array(rows['kmer'], pa.dictionary(pa.int32(), pa.string())),
        'distance': pa.array(rows['distance'], pa.float32()),
    }
    if 'pvalue' in rows:
        columns['score'] = pa.array(rows['score'], pa.float32())
        columns['pvalue'] = pa.array(rows['pvalue'], pa.float32())
        columns['calibration_count'] = pa.array(rows['calibration_count'], pa.int64())
    for key in ('s_cos', 'z_cos', 's_whiten', 'z_whiten'):
        if key in rows:
            columns[key] = pa.array(rows[key], pa.float32())
    if 'sequence_context' in rows:
        if output_context_radius <= 0 or output_context_format not in ('packed', 'text'):
            raise ValueError('Writing sequence context requires its positive radius and valid format')
        context_type = pa.binary() if output_context_format == 'packed' else pa.string()
        columns['sequence_context'] = pa.array(rows['sequence_context'], context_type)
        columns['context_complete'] = pa.array(rows['context_complete'], pa.bool_())
    table = pa.table(columns)
    if 'sequence_context' in rows:
        from porefm.workflows.canonical_null.sequence_context import PACKED_ENCODING
        index = table.schema.get_field_index('sequence_context')
        field = table.schema.field(index).with_metadata({
            'encoding': PACKED_ENCODING if output_context_format == 'packed' else 'utf8',
            'length': str(2 * output_context_radius + 1),
            'radius': str(output_context_radius),
            'orientation': 'model_input',
            'alphabet': 'ACGTN',
        })
        table = table.set_column(index, field, table.column(index))
    from porefm.outputs.writers import write_split_part
    dictionary = ['readname', 'chrom', 'strand', 'kmer', 'calibration_count']
    if 'sequence_context' in rows:
        sample = rows['sequence_context'][:4096]
        if len(set(sample)) <= 0.9 * len(sample):
            dictionary.append('sequence_context')
    return str(write_split_part(
        table, output_dir, f'part_{part_idx:06d}', format='canonical_null',
        compression='zstd',
        compression_level={name: 3 if name == 'sequence_context' else 1 for name in table.column_names},
        use_dictionary=[name for name in dictionary if name in table.column_names],
        use_byte_stream_split=[field.name for field in table.schema if pa.types.is_floating(field.type)],
    ))


def _configure_deterministic_detection():
    """Enable process-wide deterministic execution for an opted-in CLI run.

    Call before the first CUDA operation. PyTorch's deterministic mode also
    fixes Inductor's numerically sensitive kernel choices. The workspace
    setting supports deterministic CUDA matrix products; an explicit setting
    is preserved. Settings remain active for this CLI process. Repeatability
    requires identical inputs, batching, model artifacts, hardware, and
    software versions. Existing autocast and matmul precision are retained.

    Returns a JSON-compatible numerical-policy record for the run manifest.
    """
    workspace = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    lazy_import_runtime_dependencies()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return {
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cublas_workspace_config': workspace,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'torch_version': str(torch.__version__),
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
    }


@use_model_bundle('zero-shot')
def run(args):
    """Run canonical-null inference and stream stitched rows to parquet parts.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments selecting the calibrated model, BAM inputs, chunking
        policy, and output cadence.

    Notes
    -----
    This workflow is intentionally single-process at the orchestration level;
    data-loader workers may still read BAM records concurrently. Output is
    flushed every ``checkpoint_batch_step`` batches and once at shutdown.
    Omitting ``--chrom`` selects all reference contigs in reference-based mode.
    With ``--read_based``, it instead streams every BAM record, including
    unmapped reads, and selects motifs in the called sequence. Existing quality,
    length, flag, and chunk-trimming filters still apply. Missing reference
    coordinates are written as ``position=-1``; ``read_pos`` remains available.
    With ``--pod5``, one BAM joins raw signal in POD5 order. The parent prepares
    a saved BAM index before model loading; workers own disjoint POD5 files.
    ``--deterministic`` opts into process-wide deterministic numerical kernels
    and records the policy in the manifest. It can affect speed and rounding;
    it does not regenerate calibration or alter the default execution policy.
    """
    resolve_cli_preprocessing(args)
    if args.deterministic:
        args.numerical_policy = _configure_deterministic_detection()
    from porefm.common.run_outputs import prepare_run_outputs
    prepare_run_outputs(args, command="zero-shot detect",
                        directories={"output_dir": args.output_dir})
    pod5_options = shared.prepare_cli_pod5_input(args)
    lazy_import_runtime_dependencies()
    if args.peak_only:
        from porefm.workflows.canonical_null.peak_filter import warm_peak_filter
        warm_peak_filter()
    if args.output_context_radius > 0 and args.output_context_format == 'packed':
        from porefm.workflows.canonical_null.sequence_context import warm_context_encoder
        warm_context_encoder()
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_bases = shared.is_all_base_motif(args.motif)
    label_all_aligned_bases = all_bases and not args.read_based
    # The ordinary read-motif parser requires concrete target bases. Expand
    # the N-0 shorthand within each read, never into genome-wide label arrays.
    sample_motif = args.motif
    if all_bases:
        sample_motif = 'A 0;C 0;G 0;T 0' if args.read_based else None
    rank = 0
    world_size = 1
    device = torch.device(args.device)
    calibration = None
    score_type = RAW_COSINE_SCORE_TYPE

    if args.score_type == 'auto':
        calibration_probe = load_canonical_calibration(args.calibration_path)
        metadata = calibration_probe.get('metadata', {})
        score_type = normalize_score_type(
            metadata.get('raw_score_type')
            or metadata.get('score_type')
            or RAW_COSINE_SCORE_TYPE
        )
        calibration = calibration_probe
    else:
        score_type = normalize_score_type(args.score_type)
    canonical_model, _pretrain_config, _canonical_metadata = load_canonical_null_model(
        args.canonical_null_model_path,
        device,
        whitening_artifact_path=args.whitening_artifact or None,
        seq_type=args.seq_type,
    )
    active_pretrain_config = _pretrain_config
    if calibration is None:
        calibration = load_canonical_calibration(
            args.calibration_path,
            expected_score_type=score_type,
        )
    else:
        stored_metadata = calibration.get('metadata', {})
        stored_score_type = normalize_score_type(
            stored_metadata.get('raw_score_type')
            or stored_metadata.get('score_type')
            or RAW_COSINE_SCORE_TYPE
        )
        if stored_score_type != score_type:
            raise ValueError(
                "Calibration score type mismatch: "
                f"stored={stored_score_type}, requested={score_type}"
            )
    if score_type == RAW_WHITENING_SCORE_TYPE and not args.whitening_artifact:
        raise ValueError("--score_type whitening requires --whitening_artifact")
    validate_artifact_preprocessing(calibration, args.preprocessing)
    prepare_canonical_calibration(calibration, device)
    kmer_size = int(calibration.get('kmer_size', args.kmer_size))

    minimum_signal_overlap = required_scale_signal_overlap(
        active_pretrain_config, canonical_model.scale_model,
    )
    active_null = getattr(canonical_model, 'null_predictor', None)
    if requires_rna_structure_features(active_null):
        minimum_signal_overlap = max(minimum_signal_overlap, required_scale_signal_overlap(active_pretrain_config, active_null))
    args.step = resolve_downstream_step(
        args.size,
        args.seq_type,
        args.step,
        minimum_overlap=minimum_signal_overlap,
        context="zero-shot detect",
    )

    if args.disable_compile is False:
        canonical_model = torch.compile(canonical_model, dynamic=True)

    kmer_table = resolve_cli_preprocessing(args).kmer_table
    # An empty region selection enables the loader's sequential read-based
    # stream. Expanding it to FASTA contigs would silently exclude unmapped reads.
    chrom_info = (
        [] if args.read_based and args.chrom is None
        else shared.get_chrom_info(args.ref, args.chrom)
    )
    bam_paths = shared.expand_bam_paths(args.bam_path)
    bam_samples = [
        shared.LabelledBamSample(
            bam_path=bam_path,
            chrom_info=chrom_info,
            ref=args.ref,
            num_classes=2,
            pos_file=args.pos_file,
            motif=sample_motif,
            motif_label=args.motif_label,
            flag=args.flag,
            read_based=args.read_based,
            **pod5_options,
            polish=args.use_polished_sequence,
            polish_indel_limit=args.polish_indel_limit,
            polish_vcf_path=(
                args.polish_vcf_path if args.use_polished_sequence else None
            ),
        )
        for bam_path in bam_paths
    ]
    dataset = shared.CompleteLabelledReadIterableDataset(
        bam_list=bam_samples,
        size=args.size,
        step=args.step,
        move_downsample_factor=1,
        seq_type=args.seq_type,
        chunking_mode=resolve_chunking_mode(args), kmer_table=kmer_table,
        reads_per_chunk=args.reads_per_chunk,
        min_read_length=args.min_read_length,
        max_read_length=args.max_read_length,
        max_alignment_length=args.max_alignment_length,
        min_qscore=args.min_qscore,
        pore_type=args.pore_type,
        base_seed=args.seed,
        rank=rank,
        world_size=world_size,
        discard_empty_labels=True,
        return_pysam=False,
        return_scale_features=True,
        label_all_aligned_bases=label_all_aligned_bases,
        expected_ts_dim=active_pretrain_config.ts_dim,
        allow_pod5=True,
    )
    num_workers = shared.inference_worker_count(dataset, args.num_workers, rank, world_size)
    loader_kwargs = {
        'batch_size': None,
        'num_workers': num_workers,
        'pin_memory': device.type == 'cuda',
    }
    from porefm.workflows.canonical_null.rna_structure import append_rna_structure_features
    # One extraction feeds either or both heads; read their saved schemas.
    active_scale = canonical_model.scale_model
    loader_kwargs['collate_fn'] = partial(
        append_rna_structure_features,
        enabled=requires_rna_structure_features(active_null, active_scale),
    )
    if num_workers > 0:
        loader_kwargs.update(
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
        )
    loader = DataLoader(dataset, **loader_kwargs)

    include_context = args.output_context_radius > 0
    rows = empty_rows(include_calibration=True, include_context=include_context)
    part_idx = 0
    total_rows = 0
    start = time.time()
    batch_idx = 0
    loader_iter = iter(loader)
    try:
        while True:
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            batch_idx += 1
            batch_result = compute_distance_batch_canonical_null(
                batch,
                canonical_model,
                calibration,
                kmer_size,
                device,
                args.num_trim,
                args.batch_size,
                score_type=score_type,
                include_sequence=include_context,
            )
            batch_rows = stitch_batch_rows(
                batch_result,
                kmer_size,
                min_metric=args.min_metric,
                max_pvalue=args.max_pvalue,
                peak_only=args.peak_only,
                prune_radius=args.prune_radius,
                output_context_radius=args.output_context_radius,
                output_context_format=args.output_context_format,
            )
            append_rows(rows, batch_rows)
            total_rows += len(batch_rows['distance'])

            if batch_idx % args.log_every == 0:
                print(
                    f"[{time.time() - start:.2f}s] batches={batch_idx} rows={total_rows}",
                    flush=True,
                )
            if batch_idx % args.checkpoint_batch_step == 0:
                path = write_rows(rows, args.output_dir, part_idx,
                                  output_context_radius=args.output_context_radius,
                                  output_context_format=args.output_context_format)
                if path is not None:
                    print(f"Wrote {path}", flush=True)
                    part_idx += 1
                rows = empty_rows(include_calibration=True, include_context=include_context)
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
    finally:
        shared.shutdown_dataloader_iterator(loader_iter)

    path = write_rows(rows, args.output_dir, part_idx,
                      output_context_radius=args.output_context_radius,
                      output_context_format=args.output_context_format)
    if path is not None:
        print(f"Wrote {path}", flush=True)
    print({'total_rows': total_rows, 'output_dir': args.output_dir}, flush=True)


def parse_args(argv=None):
    """Parse calibrated detection, input selection, and output options."""
    parser = PoreFMArgumentParser(
        formatter_class=shared.DefaultsFormatter,
        description=(
            "Detect deviations from a canonical embedding reference using a unified "
            "model and empirical k-mer calibration."
        ),
    )

    from porefm.common.run_outputs import add_output_safety_argument
    add_output_safety_argument(parser)
    parser.add_argument('--bam_path', required=True)
    shared.add_pod5_arguments(parser)
    add_chunking_arguments(parser)
    parser.add_argument('--kmer_dict_path', required=True)
    parser.add_argument(
        '--ref', default=None,
        help="Reference FASTA; required unless --read_based is used without --chrom or polishing.",
    )
    parser.add_argument('--output_dir', required=True)
    parser.add_argument(
        '--chrom', nargs='+', default=None,
        help=(
            "One or more chromosomes to process fully (requires --ref). "
            "If omitted, process all reference contigs, or stream all "
            "reads including unmapped records with --read_based."
        ),
    )
    parser.add_argument(
        '--read_based', '--read-based', action='store_true',
        help=(
            "Select motifs in the called read sequence instead of the reference. "
            "Without --chrom, include unmapped reads and allow unindexed BAMs; "
            "--ref is optional. Use --motif 'N 0' to score every eligible read base."
        ),
    )
    parser.add_argument('--pos_file', default=None)
    parser.add_argument(
        '--motif',
        default=None,
        help=(
            "Motif specification supplied as one quoted shell argument, "
            "for example 'CG 0;CH 0'; use 'N 0' to score every aligned base "
            "(every eligible read base with --read_based)."
        ),
    )
    parser.add_argument('--motif_label', default='1', help='Semicolon-delimited class labels corresponding to --motif.')
    parser.add_argument('--seq_type', default='dna', choices=['dna', 'rna'])
    parser.add_argument('--pore_type', default='r10')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--reads_per_chunk', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--min_read_length', type=int, default=100)
    parser.add_argument('--max_read_length', type=float, default=5e5)
    parser.add_argument('--max_alignment_length', type=float, default=1e6)
    parser.add_argument('--min_qscore', type=float, default=10)
    parser.add_argument('--use_polished_sequence', default=False, action='store_true', help='Use reference-guided polished base sequences for query extraction.')
    parser.add_argument('--polish_indel_limit', type=int, default=5, help='Maximum unmasked insertion/deletion length corrected during polishing.')
    parser.add_argument('--polish_vcf_path', default=None, help='Optional VCF/BCF whose ALT SNPs and indel loci are preserved during polishing.')
    parser.add_argument('--num_trim', type=int, default=16)
    output_filter = parser.add_mutually_exclusive_group()
    output_filter.add_argument(
        '--min_metric', type=float, default=None,
        help='Strict raw-distance cutoff; defaults to 0.0 when --max_pvalue is not supplied.',
    )
    parser.add_argument('--flag', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint_batch_step', type=int, default=10)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--disable_compile', action='store_true')
    parser.add_argument(
        '--deterministic', action='store_true',
        help=(
            'Use deterministic numerical kernels for repeated runs with the same '
            'inputs, batching, hardware, and software. May reduce speed and change '
            'scores relative to the default policy; retains compilation and '
            'precision settings. Does not regenerate calibration.'
        ),
    )

    output_filter.add_argument(
        '--max_pvalue', type=float, default=None,
        help='Inclusive calibrated p-value cutoff in [0, 1], replacing --min_metric.',
    )
    parser.add_argument(
        '--peak_only', action='store_true',
        help="Keep stitched-read p-value minima; flat peaks use their midpoint rounded down. Requires --motif 'N 0' and no --pos_file.",
    )
    parser.add_argument(
        '--prune_radius', type=int, default=0,
        help='Suppress weaker qualifying peaks within +/- this many read bases, inclusive. Requires --peak_only; 0 disables pruning.',
    )
    parser.add_argument(
        '--output_context_radius', type=int, default=0,
        help=(
            'Add sequence_context (2R+1 model-input bases, centered on read_pos) '
            'and context_complete columns. Unavailable sequence is N-padded; '
            '0 disables context output. Uses polished sequence when enabled.'
        ),
    )
    parser.add_argument(
        '--output_context_format', choices=['packed', 'text'], default='packed',
        help='Sequence context storage: compact two-bit DNA with a lossless N mask, or readable text. Both use Zstandard compression.',
    )
    add_model_arguments(parser, required=False, local_flags=('--model_path', '--model-path'),
                        local_help='Complete inference bundle; overrides all separate model artifact paths.')
    parser.add_argument('--canonical_null_model_path', default=None)
    parser.add_argument('--calibration_path', default=None)
    parser.add_argument('--kmer_size', type=int, default=9)
    parser.add_argument(
        '--score_type',
        choices=['auto', 'cosine', 'whitening'],
        default='auto',
    )
    parser.add_argument('--whitening_artifact', default='')
    add_preprocessing_arguments(parser)
    args = parser.parse_args(argv)
    resolve_model_arguments(args, roles=('zero-shot',), parser=parser)
    try:
        apply_bundle_arguments(args, 'zero-shot')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not args.model_path and (not args.canonical_null_model_path or not args.calibration_path):
        parser.error('provide --model, --model_path/--model-path, or both --canonical_null_model_path and --calibration_path')
    shared.validate_pod5_arguments(parser, args)
    try:
        validate_output_filters(
            args.min_metric, args.max_pvalue, args.peak_only, args.prune_radius,
            calibrated=True,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.output_context_radius < 0:
        parser.error('--output_context_radius must be nonnegative')
    if args.peak_only and (not shared.is_all_base_motif(args.motif) or args.pos_file):
        parser.error("--peak_only requires --motif 'N 0' and no --pos_file")
    if args.min_metric is None and args.max_pvalue is None:
        args.min_metric = 0.0
    if not args.ref and (not args.read_based or args.chrom or args.use_polished_sequence):
        parser.error("--ref is required for reference-based mode, --chrom, or --use_polished_sequence")
    if args.read_based:
        if args.pos_file:
            parser.error("--read_based cannot be combined with --pos_file")
        if not args.motif:
            parser.error("--read_based requires --motif; use --motif 'N 0' for all read bases")
    return args


if __name__ == '__main__':
    shared.require_single_process_environment(
        "porefm zero-shot detect",
        "Zero-shot detection writes locus/read-level records in one process.",
    )
    print(
        '\n%s:\nCommand: python %s\n'
        % (str(datetime.datetime.now()), ' '.join(sys.argv)),
        flush=True,
    )
    run(parse_args())
