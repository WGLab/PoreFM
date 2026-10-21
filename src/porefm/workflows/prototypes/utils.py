"""Shared input, embedding, distance, and output helpers for prototype workflows."""

from porefm.preprocessing import validate_artifact_preprocessing
import os
from pathlib import Path

from porefm.cli._parser import PoreFMHelpFormatter
from porefm.cli.pod5_args import add_pod5_arguments, validate_pod5_arguments
from porefm.inference.pod5_input import prepare_cli_pod5_input, inference_worker_count
from porefm.workflows.prototypes.embedding_selection import EMBEDDING_KEY_CHOICES

np = None
pa = None
pq = None
pysam = None
torch = None
F = None
DataLoader = None
build_lookup_table = None
CompleteLabelledReadIterableDataset = None
LabelledBamSample = None
load_checked_model = None


def lazy_import_runtime_dependencies():
    """Import tensor, BAM, model, and parquet dependencies once on demand."""
    global np, pa, pq, pysam, torch, F, DataLoader
    global build_lookup_table, CompleteLabelledReadIterableDataset, LabelledBamSample
    global load_checked_model

    if torch is not None:
        return

    import numpy as _np
    import pyarrow as _pa
    import pyarrow.parquet as _pq
    import pysam as _pysam
    import torch as _torch
    from torch.nn import functional as _F
    from torch.utils.data import DataLoader as _DataLoader

    from porefm.data.bam import build_lookup_table as _build_lookup_table
    from porefm.data.complete_reads import CompleteLabelledReadIterableDataset as _CompleteLabelledReadIterableDataset
    from porefm.data.reader import LabelledBamSample as _LabelledBamSample
    from porefm.models.loading import load_model as _load_checked_model

    np = _np
    pa = _pa
    pq = _pq
    pysam = _pysam
    torch = _torch
    F = _F
    DataLoader = _DataLoader
    build_lookup_table = _build_lookup_table
    CompleteLabelledReadIterableDataset = _CompleteLabelledReadIterableDataset
    LabelledBamSample = _LabelledBamSample
    load_checked_model = _load_checked_model


class DefaultsFormatter(PoreFMHelpFormatter):
    """Argument formatter that displays defaults for otherwise blank help."""


def require_single_process_environment(script_name: str, reason: str):
    """Reject multi-rank execution for a single-writer comparison workflow."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size != 1:
        raise RuntimeError(
            f"{script_name} must run with a single process/GPU. {reason} "
            "Launch it without torchrun/srun multi-rank environment variables."
        )


def expand_bam_paths(bam_path):
    """Resolve one BAM or recursively list directory BAMs in natural order."""
    from natsort import natsorted

    path_obj = Path(bam_path)
    if path_obj.is_dir():
        bam_paths = natsorted([str(path) for path in path_obj.rglob('*.bam')])
    elif path_obj.is_file():
        bam_paths = [str(path_obj)]
    else:
        raise FileNotFoundError(f"BAM path does not exist: {bam_path}")

    if len(bam_paths) == 0:
        raise FileNotFoundError(f"No .bam files found under: {bam_path}")
    return bam_paths


def shutdown_dataloader_iterator(loader_iter):
    """Stop private data-loader workers when early workflow exit requires it."""
    shutdown = getattr(loader_iter, '_shutdown_workers', None)
    if shutdown is not None:
        shutdown()


def is_all_base_motif(motif):
    """Return whether a motif specification requests every canonical base."""
    return motif is not None and motif.strip().upper().split() == ['N', '0']


def ignore_edges(x, num_trim):
    """Remove a fixed count of valid bases from both ends of each mask row."""
    lens = x.sum(dim=1, keepdim=True)
    indices = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    valid_region = (indices >= num_trim) & (indices < (lens - num_trim))
    return x & valid_region


def kmer_index_to_string(kmer_idx, k=9):
    """Decode a two-bit integer k-mer representation to an A/C/G/T string."""
    bases = "ACGT"
    return "".join(bases[(int(kmer_idx) >> (2 * offset)) & 3] for offset in range(k - 1, -1, -1))


def load_pretrain_model(model_path, device):
    """Restore the query-embedding backbone in evaluation mode.

    Returns the model and its normalized configuration. Signal-to-base
    cross-attention is disabled; signal-modality dropout remains recorded in
    the config but is inactive under the inference forward default.
    """
    model = load_checked_model(
        model_path,
        model_type="backbone",
        add_signal_cross_attn=False,
        device=device,
    )
    return model, model.config


def resolve_prototype_embedding_key(prototype_metadata, requested_key=None):
    """Resolve one embedding method from prototype metadata and an optional check."""
    validate_artifact_preprocessing(prototype_metadata)
    resolved = []
    for prototype_info in prototype_metadata:
        stored_key = prototype_info['metadata'].get('embedding_key')
        if stored_key is None:
            stored_key = requested_key or 'contextual_base_seq'
        if stored_key not in EMBEDDING_KEY_CHOICES:
            raise ValueError(
                f"Prototype {prototype_info['name']!r} records unsupported "
                f"embedding_key={stored_key!r}"
            )
        if requested_key is not None and stored_key != requested_key:
            raise ValueError(
                f"Prototype {prototype_info['name']!r} records "
                f"embedding_key={stored_key!r}, which does not match requested "
                f"--embedding_key={requested_key!r}"
            )
        resolved.append(stored_key)

    unique_keys = sorted(set(resolved))
    if len(unique_keys) != 1:
        details = ', '.join(
            f"{prototype_info['name']}={embedding_key}"
            for prototype_info, embedding_key in zip(prototype_metadata, resolved)
        )
        raise ValueError(
            "Prototype files use different embedding methods; all files must "
            f"agree ({details})"
        )
    return unique_keys[0]


def get_chrom_info(ref_path, chroms):
    """Return full-length FASTA intervals for ``chroms``, or all if ``None``.

    ``chroms`` is a sequence of contig names or ``None``. Explicit selections
    retain their order and require every name to exist in the indexed FASTA;
    an empty sequence remains an empty selection. Returned intervals are
    ``(contig, 0, length)`` in zero-based half-open coordinates. The BAM loader
    handles which selected contigs actually contain aligned reads.
    """
    with pysam.FastaFile(ref_path) as fasta:
        if chroms is None:
            chroms = fasta.references
        return [(chrom, 0, fasta.get_reference_length(chrom)) for chrom in chroms]


def get_read_based_chrom_info(ref_path, chroms):
    """Resolve optional contig filtering independently of read-motif selection.

    Without ``chroms``, return an empty region list to stream all BAM records,
    including unmapped reads, even when a reference FASTA is supplied. Explicit
    contigs use indexed BAM fetching. A FASTA validates their names/lengths;
    without one, ``(contig, 0, None)`` requests the entire BAM contig.
    """
    if not chroms:
        return []
    chroms = list(dict.fromkeys(chroms))
    if ref_path:
        return get_chrom_info(ref_path, chroms)
    return [(chrom, 0, None) for chrom in chroms]


def compute_chunk_kmer_ids(base_seq_input, kmer_size):
    """Encode centered canonical k-mers for a padded model-input batch.

    Parameters
    ----------
    base_seq_input : torch.Tensor
        Integer bases with shape ``(batch, length)`` using the model's one-based
        encoding (zero is padding).
    kmer_size : int
        Odd window length.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        K-mer IDs (``-1`` when unavailable) and a boolean validity mask, each
        shaped like ``base_seq_input``.
    """
    if kmer_size % 2 != 1:
        raise ValueError(f"kmer_size must be odd, got {kmer_size}")

    half = kmer_size // 2
    bases = base_seq_input - 1
    valid_base = (bases >= 0) & (bases < 4)
    kmer_ids = torch.full(base_seq_input.shape, -1, dtype=torch.long, device=base_seq_input.device)
    kmer_valid = torch.zeros(base_seq_input.shape, dtype=torch.bool, device=base_seq_input.device)

    if base_seq_input.shape[1] < kmer_size:
        return kmer_ids, kmer_valid

    windows = bases.unfold(dimension=1, size=kmer_size, step=1)
    valid_windows = valid_base.unfold(dimension=1, size=kmer_size, step=1).all(dim=-1)

    weights = torch.tensor(
        [4 ** power for power in range(kmer_size - 1, -1, -1)],
        dtype=torch.long,
        device=base_seq_input.device,
    )
    encoded = (windows.long() * weights).sum(dim=-1)
    kmer_ids[:, half:-half] = encoded
    kmer_valid[:, half:-half] = valid_windows
    return kmer_ids, kmer_valid


def compute_distance(query_embeddings, reference_mean, reference_logvar, metric):
    """Compute row-wise L2, cosine, or diagonal-Mahalanobis distance."""
    if metric == 'l2':
        return torch.linalg.vector_norm(query_embeddings - reference_mean, ord=2, dim=-1)
    if metric == 'cosine':
        return 1.0 - F.cosine_similarity(query_embeddings, reference_mean, dim=-1)
    if metric == 'mahalanobis':
        if reference_logvar is None:
            raise ValueError("mahalanobis metric requires reference log-variance")
        logvar = torch.clamp(reference_logvar, min=-10.0, max=10.0)
        variance = torch.exp(logvar) + 1e-6
        return torch.sqrt(torch.sum((query_embeddings - reference_mean) ** 2 / variance, dim=-1))
    raise ValueError(f"Unknown metric: {metric}")


def unique_read_boundaries(readnames):
    """Group consecutive chunk indices belonging to the same read name."""
    read_mask = np.empty(len(readnames), dtype=bool)
    read_mask[0] = True
    read_mask[1:] = readnames[1:] != readnames[:-1]
    boundaries = np.where(read_mask[1:])[0] + 1
    return read_mask, np.split(np.arange(len(readnames)), boundaries)


def prototype_distance_column(prototype_name):
    """Return the parquet distance-column name for a prototype label."""
    return f'distance_{prototype_name}'


def append_rows(dst, src):
    """Append column-oriented rows from ``src`` to ``dst`` in place."""
    for key, values in src.items():
        dst[key].extend(values)


def write_prototype_rows(rows, prototype_names, output_dir, part_idx):
    """Write a Snappy parquet part, returning its path or ``None`` if empty."""
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
        'closest_prototype_index': pa.array(rows['closest_prototype_index'], pa.int32()),
        'closest_prototype': pa.array(
            rows['closest_prototype'],
            pa.dictionary(pa.int32(), pa.string()),
        ),
    }
    for prototype_name in prototype_names:
        column_name = prototype_distance_column(prototype_name)
        columns[column_name] = pa.array(rows[column_name], pa.float32())
    table = pa.table(columns)
    path = os.path.join(output_dir, f"part_{part_idx:06d}.parquet")
    pq.write_table(table, path, compression='snappy')
    return path
