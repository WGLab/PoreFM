"""Extract or merge per-k-mer prototype embeddings from aligned BAM reads."""

from porefm.preprocessing import validate_artifact_preprocessing
from porefm.cli.preprocessing_args import add_preprocessing_arguments, resolve_cli_preprocessing
from porefm.cli.chunking_args import add_chunking_arguments, resolve_chunking_mode
import argparse
import contextlib
import datetime
import json
import mmap
import os
import random
import sys
import tempfile
import time
from pathlib import Path

from porefm.cli._parser import PoreFMArgumentParser, PoreFMHelpFormatter
from porefm.common.run_outputs import add_output_safety_argument, prepare_run_outputs
from porefm.workflows.prototypes.embedding_selection import EMBEDDING_KEY_CHOICES


class DefaultsFormatter(PoreFMHelpFormatter):
    """Argument formatter that displays defaults even when help is omitted."""


def build_arg_parser():
    """Build the prototype extraction and reservoir-union CLI parser."""
    parser = PoreFMArgumentParser(
        formatter_class=DefaultsFormatter,
        description=(
            "Extract strand-agnostic exact read/reference-match k-mer embedding "
            "prototypes, optionally restricted to labelled center positions, "
            "or union previously saved reservoir prototype files."
        ),
    )
    bam_source = parser.add_mutually_exclusive_group()
    bam_source.add_argument(
        '--bam_path',
        '--bam',
        dest='bam_path',
        default='',
        help="Input BAM file or directory containing BAM files.",
    )
    bam_source.add_argument(
        '--bam_config',
        default='',
        help=(
            "JSON input config with a nonempty 'samples' list. Each sample "
            "requires 'bam_path' (a BAM or recursive BAM directory) and may "
            "provide 'ref' and 'chroms'. Missing per-sample refs inherit "
            "--ref; an explicit null ref uses MD+CIGAR matching."
        ),
    )
    add_chunking_arguments(parser)
    parser.add_argument('--kmer_dict_path', default='')
    parser.add_argument(
        '--ref',
        default='',
        help=(
            "Optional reference FASTA for --bam, or the default reference for "
            "--bam_config samples that omit 'ref'. Without a reference, "
            "exact-match filtering requires MD+CIGAR in every processed read."
        ),
    )
    parser.add_argument(
        '--chrom',
        nargs='+',
        default=None,
        help=(
            "Optional contigs to process. When omitted, use contigs present in "
            "--pos_file, or all contigs from the source FASTA/BAM header when "
            "no position file is supplied."
        ),
    )
    label_source = parser.add_mutually_exclusive_group()
    label_source.add_argument(
        '--pos_file',
        default='',
        help="Optional zero-based reference-position file; only listed centers are used.",
    )
    label_source.add_argument(
        '--motif',
        default='',
        help=(
            "Optional motif specification supplied as one quoted shell argument "
            "(for example 'CG 0;CH 0'); only motif centers are used."
        ),
    )
    parser.add_argument(
        '--read_based',
        '--read_based_motif',
        dest='read_based_motif',
        action='store_true',
        help=(
            "Find --motif in each oriented read instead of in the reference. "
            "Exact k-mer matching uses the source FASTA when provided and "
            "otherwise requires MD+CIGAR."
        ),
    )
    parser.add_argument('--output', default='')
    parser.add_argument(
        '--union_prototype_files',
        nargs='+',
        default=None,
        help=(
            "Merge saved reservoir prototype .pt files instead of extracting "
            "from BAMs. K-mers are unioned, overlapping counts/means are "
            "combined, and every saved reservoir embedding is concatenated "
            "without an additional cap. Uses memory-mapped inputs and disk-backed "
            "output to limit RAM use. Preserves the reservoir dtype (promotes "
            "mixed dtypes). Requires --output and disk space "
            "for the complete uncapped union."
        ),
    )
    parser.add_argument('--model_path', default='')
    parser.add_argument('--seq_type', default='dna', choices=['dna', 'rna'])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--disable_compile', action='store_true', help='Disable the default torch.compile(dynamic=True) model compilation.')
    parser.add_argument('--prototype_device', default='cpu')
    parser.add_argument(
        '--embedding_key',
        choices=EMBEDDING_KEY_CHOICES,
        default='contextual_base_seq',
        help="Base-aligned model representation to accumulate.",
    )
    parser.add_argument('--dry_run_no_model', action='store_true')
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--reads_per_chunk', type=int, default=20)
    parser.add_argument('--min_read_length', type=int, default=100)
    parser.add_argument('--min_qscore', type=float, default=10)
    parser.add_argument('--use_polished_sequence', default=False, action='store_true', help='Use reference-guided polished base sequences during prototype extraction.')
    parser.add_argument('--polish_indel_limit', type=int, default=5, help='Maximum unmasked insertion/deletion length corrected during polishing.')
    parser.add_argument('--polish_vcf_path', default=None, help='Optional VCF/BCF whose ALT SNPs and indel loci are preserved during polishing.')
    parser.add_argument('--min_base_quality', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--num_trim', type=int, default=16)
    parser.add_argument('--kmer_size', type=int, default=9)
    parser.add_argument('--min_count', type=int, default=1)
    parser.add_argument(
        '--reservoir_size',
        type=int,
        default=0,
        help=(
            "Maximum bfloat16 sampled embeddings saved per retained k-mer. Zero disables "
            "reservoir storage; means and total counts are always saved."
        ),
    )
    parser.add_argument('--reservoir_seed', type=int, default=0)
    parser.add_argument('--pore_type', default='r10')
    parser.add_argument('--ts_dim', type=int, default=None, help="Expected BAM mv tag stride. When set, requires mv[0] to match.")
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--log_every', type=int, default=100)
    add_output_safety_argument(parser)
    add_preprocessing_arguments(parser)
    return parser


def parse_args(argv=None):
    """Parse prototype extraction arguments from ``argv`` or ``sys.argv``."""
    return build_arg_parser().parse_args(argv)


if __name__ == '__main__' and any(arg in ('-h', '--help') for arg in sys.argv[1:]):
    parse_args()

import numpy as np
import pysam
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

from porefm.data.chunk_boundaries import validate_chunking_mode
from porefm.data.inferred_chunks import (has_inferred_boundaries, inferred_chunks,
    inferred_labelled_chunks, slice_base_values, chunk_bounds)
from porefm.data.bam import (
    Read,
    Read_Lite,
    build_lookup_table, resolve_preprocessing,
    get_pos,
    get_read_info,
    get_read_info_lite,
    get_read_motif_targets,
    parse_motif_specs,
    passes_min_qscore,
    pre_process,
    num_to_base_map,
)
from porefm.data.reader import (
    chunk_padded_components,
    format_iterable_bam_log,
    get_percentiles,
    get_polished_read_for_sample,
    get_read_chunks_labelled,
    normalize_read_signal,
    pad_read_components,
)
from porefm.data.datasets.aligned import AlignedBamSample
from porefm.data.collation import collate_modal_aligned_batch
from porefm.data.reference_match import (
    build_exact_match_kmer_arrays_from_read_arrays,
    build_md_exact_match_kmer_arrays,
)
from porefm.workflows.prototypes.embedding_selection import (
    get_embedding_dimension,
    select_base_embedding,
)
from porefm.models.loading import load_model as load_checked_model
from porefm.common.chunking import resolve_downstream_step
from porefm.common.utils import DegenerateQuantileNormalizationError


def require_single_process_environment(script_name: str, reason: str):
    """Reject multi-rank environments for workflows with shared output state."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size != 1:
        raise RuntimeError(
            f"{script_name} must run with a single process/GPU. {reason} "
            "Launch it without torchrun/srun multi-rank environment variables."
        )


def expand_bam_paths(bam_path):
    """Resolve a BAM file or recursively expand a directory in natural order."""
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


def load_bam_source_config(config_path, default_ref='', default_chroms=None):
    """Load and validate per-source BAM, reference, and contig settings.

    Relative BAM and FASTA paths are resolved against the JSON file's parent
    directory. A missing ``ref`` field inherits ``default_ref``; an explicit
    JSON ``null`` selects MD+CIGAR matching even when a default was supplied.

    Parameters
    ----------
    config_path : str
        JSON file containing a nonempty top-level ``samples`` list.
    default_ref : str, optional
        Reference inherited by entries that omit ``ref``.
    default_chroms : sequence of str, optional
        Contigs inherited by entries that omit ``chroms``.

    Returns
    -------
    list of dict
        Normalized source specifications with ``bam_path``, ``ref``, and
        ``chroms`` keys.
    """
    config_file = Path(config_path).expanduser().resolve()
    with config_file.open() as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise TypeError(f"{config_file}: top-level JSON value must be an object")
    unknown_top_level = set(config) - {'samples'}
    if unknown_top_level:
        raise ValueError(
            f"{config_file}: unsupported top-level fields: {sorted(unknown_top_level)}"
        )
    samples = config.get('samples')
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{config_file}: 'samples' must be a nonempty list")

    inherited_ref = None
    if default_ref:
        inherited_ref = str(Path(default_ref).expanduser().resolve())

    def resolve_config_path(value, field, sample_index):
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise TypeError(
                f"{config_file}: samples[{sample_index}].{field} must be a "
                "nonempty string or null"
            )
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_file.parent / path
        return str(path.resolve())

    normalized = []
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise TypeError(
                f"{config_file}: samples[{sample_index}] must be a JSON object"
            )
        unknown_fields = set(sample) - {'bam_path', 'ref', 'chroms'}
        if unknown_fields:
            raise ValueError(
                f"{config_file}: samples[{sample_index}] has unsupported fields: "
                f"{sorted(unknown_fields)}"
            )
        if 'bam_path' not in sample:
            raise ValueError(f"{config_file}: samples[{sample_index}] requires 'bam_path'")

        has_sample_ref = 'ref' in sample
        ref_value = sample.get('ref')
        chroms = sample['chroms'] if 'chroms' in sample else default_chroms
        if chroms is not None:
            if (
                not isinstance(chroms, list)
                or not chroms
                or any(not isinstance(chrom, str) or not chrom for chrom in chroms)
            ):
                raise TypeError(
                    f"{config_file}: samples[{sample_index}].chroms must be a "
                    "nonempty list of strings"
                )
            chroms = list(chroms)

        normalized.append({
            'bam_path': resolve_config_path(sample['bam_path'], 'bam_path', sample_index),
            'ref': (
                resolve_config_path(ref_value, 'ref', sample_index)
                if has_sample_ref
                else inherited_ref
            ),
            'chroms': chroms,
        })
    return normalized


def shutdown_dataloader_iterator(loader_iter):
    """Promptly stop private data-loader workers when the iterator supports it."""
    shutdown = getattr(loader_iter, '_shutdown_workers', None)
    if shutdown is not None:
        shutdown()


def get_chrom_info(ref_path, chroms=None, pos_file=''):
    """Return ``(contig, 0, length)`` regions selected from FASTA metadata."""
    if not chroms and pos_file:
        chroms = list(get_pos(pos_file))
    with pysam.FastaFile(ref_path) as fasta:
        if not chroms:
            chroms = list(fasta.references)
        return [(chrom, 0, fasta.get_reference_length(chrom)) for chrom in chroms]


def get_bam_chrom_info(bam_path, chroms=None):
    """Return full contig regions from a BAM header for MD-backed matching."""
    with pysam.AlignmentFile(bam_path, 'rb', check_sq=False) as bam:
        lengths = dict(zip(bam.references, bam.lengths))
        selected = list(chroms) if chroms else list(bam.references)
    missing = [chrom for chrom in selected if chrom not in lengths]
    if missing:
        raise ValueError(f"{bam_path}: contigs absent from BAM header: {missing}")
    if not selected:
        raise ValueError(f"{bam_path}: BAM header contains no reference contigs")
    return [(chrom, 0, lengths[chrom]) for chrom in selected]


def resolve_bam_samples(args):
    """Expand CLI or JSON sources into per-BAM extraction descriptors.

    Sources with a FASTA retain the established FASTA comparison path. Sources
    without one use BAM header coordinates and are marked for MD+CIGAR matching.
    Reference-position labels, reference motifs, and polishing require a FASTA.

    Returns
    -------
    tuple
        ``(samples, source_metadata)`` where ``samples`` contains one
        :class:`AlignedBamSample` per expanded BAM and ``source_metadata`` is a
        JSON-serializable description of the resolved matching inputs.
    """
    if args.bam_config:
        source_specs = load_bam_source_config(
            args.bam_config,
            default_ref=args.ref,
            default_chroms=args.chrom,
        )
    else:
        source_specs = [{
            'bam_path': args.bam_path,
            'ref': args.ref or None,
            'chroms': args.chrom,
        }]

    samples = []
    source_metadata = []
    for source_index, source in enumerate(source_specs):
        ref_path = source['ref']
        chroms = source['chroms']
        if ref_path is None:
            if args.pos_file or (args.motif and not args.read_based_motif):
                raise ValueError(
                    f"Input source {source_index} ({source['bam_path']}) has no reference "
                    "FASTA; --pos_file and reference-based --motif require a FASTA"
                )
            if args.use_polished_sequence:
                raise ValueError(
                    f"Input source {source_index} ({source['bam_path']}) has no reference "
                    "FASTA; --use_polished_sequence requires a FASTA"
                )

        for bam_path in expand_bam_paths(source['bam_path']):
            chrom_info = (
                get_chrom_info(ref_path, chroms, args.pos_file)
                if ref_path is not None
                else get_bam_chrom_info(bam_path, chroms)
            )
            samples.append(AlignedBamSample(
                bam_path=bam_path,
                chrom_info=chrom_info,
                ref=ref_path,
                pos_file=args.pos_file,
                motif=args.motif,
                motif_label='0',
                flag=0,
                polish=args.use_polished_sequence,
                polish_indel_limit=args.polish_indel_limit,
                polish_vcf_path=(
                    args.polish_vcf_path if args.use_polished_sequence else None
                ),
            ))
            source_metadata.append({
                'bam_path': bam_path,
                'ref': ref_path,
                'chroms': [chrom for chrom, _start, _end in chrom_info],
                'reference_match_source': 'fasta' if ref_path is not None else 'md_cigar',
            })
    return samples, source_metadata


def ignore_edges(x, num_trim):
    """Remove ``num_trim`` valid positions from both ends of each mask row."""
    lens = x.sum(dim=1, keepdim=True)
    indices = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    valid_region = (indices >= num_trim) & (indices < (lens - num_trim))
    return x & valid_region


def phred_threshold_to_error_prob(min_base_quality):
    """Convert a Phred threshold to error probability, or disable filtering."""
    if min_base_quality is None or min_base_quality <= 0:
        return None
    return 10 ** (-float(min_base_quality) / 10.0)


def build_reference_label_center_mask(read, labelled_arrays):
    """Return read-coordinate centers selected on this read's reference strand."""
    center_mask = np.zeros(read.length, dtype=np.bool_)
    ref_labels = labelled_arrays.get(read.ref_name)
    if ref_labels is None:
        return center_mask

    target_positions = ref_labels[read.is_reverse][0]
    if len(target_positions) == 0:
        return center_mask

    aligned_pairs = read.aligned_pairs
    aligned = (aligned_pairs[:, 0] >= 0) & (aligned_pairs[:, 1] >= 0)
    aligned_read_positions = aligned_pairs[aligned, 0]
    aligned_ref_positions = aligned_pairs[aligned, 1]
    selected = np.isin(aligned_ref_positions, target_positions)
    center_mask[aligned_read_positions[selected]] = True
    return center_mask


def build_read_motif_center_mask(read, parsed_motifs):
    """Return read-coordinate centers selected by motifs in the oriented read."""
    center_mask = np.zeros(read.length, dtype=np.bool_)
    read_sequence = ''.join(read.get_base_seq(string=True).tolist())
    target_positions, _ = get_read_motif_targets(
        read_sequence=read_sequence,
        parsed_motifs=parsed_motifs,
        num_classes=1,
    )
    center_mask[target_positions] = True
    return center_mask


def build_md_backed_read(bam_read, seq_type, kmer_table, expected_ts_dim=None, chunking_mode="auto"):
    """Decode a signal read and attach alignment metadata without a FASTA.

    Sequence, signal, and expected-current fields come from
    :func:`get_read_info_lite`. Reference coordinates come from CIGAR through
    pysam, while exact reference bases remain the responsibility of the MD
    matching helper. Query coordinates and qualities are reversed for reverse
    alignments so all arrays follow the model's read-oriented convention.

    Parameters
    ----------
    bam_read : pysam.AlignedSegment
        Mapped signal-BAM record with query sequence, CIGAR, MD, and signal
        tags required by :func:`get_read_info_lite`.
    seq_type : str
        Sequence type passed to the signal decoder.
    kmer_table : numpy.ndarray
        Expected-current lookup table.
    expected_ts_dim : int or None
        Optional required move-table stride.

    Returns
    -------
    Read_Lite
        Read-oriented signal object augmented with alignment coordinates,
        base error probabilities, strand, and reference span metadata.
    """
    if bam_read.is_unmapped:
        raise ValueError(
            f"Read {bam_read.query_name!r} must be mapped for prototype extraction"
        )
    read = Read_Lite(*get_read_info_lite(
        bam_read,
        seq_type,
        kmer_table,
        expected_ts_dim=expected_ts_dim, chunking_mode=chunking_mode,
    ))

    aligned_pairs = np.asarray([
        (
            -1 if query_pos is None else int(query_pos),
            -1 if ref_pos is None else int(ref_pos),
        )
        for query_pos, ref_pos in bam_read.get_aligned_pairs(matches_only=False)
    ], dtype=np.int32)
    if bam_read.is_reverse:
        aligned_pairs = aligned_pairs[::-1].copy()
        query_mask = aligned_pairs[:, 0] >= 0
        aligned_pairs[query_mask, 0] = read.length - aligned_pairs[query_mask, 0] - 1

    query_qualities = bam_read.query_qualities
    if query_qualities is None:
        base_qual = np.ones(read.length, dtype=np.float32)
    else:
        phred = np.asarray(query_qualities, dtype=np.float32)
        if bam_read.is_reverse:
            phred = phred[::-1].copy()
        base_qual = np.power(10.0, -phred / 10.0).astype(np.float32)

    read.base_qual = base_qual
    read.aligned_pairs = aligned_pairs
    read.ref_name = bam_read.reference_name
    read.flag = bam_read.flag
    read.is_forward = bam_read.is_forward
    read.is_reverse = bam_read.is_reverse
    read.ref_start = bam_read.reference_start
    read.ref_end = bam_read.reference_end
    read.cigarstring = bam_read.cigarstring
    return read


def build_exact_match_kmer_arrays(
    read,
    k=9,
    min_base_quality=0,
    eligible_center_mask=None,
):
    """Encode exact read/reference k-mers at eligible read-coordinate centers.

    A center is valid only when its odd-length window contains canonical bases,
    every base exactly matches the reference, adjacent reference coordinates
    differ by one in either strand direction, and all quality values meet the
    requested threshold.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Per-read integer k-mer IDs (``-1`` when invalid) and a boolean validity
        mask, both with shape ``(read.length,)``.
    """
    if k % 2 != 1:
        raise ValueError(f"k must be odd so each k-mer has one center base, got {k}")

    n = read.length
    kmer_ids = np.full(n, -1, dtype=np.int64)
    kmer_valid = np.zeros(n, dtype=np.bool_)
    if n < k:
        return kmer_ids, kmer_valid

    ref_pos_by_read = np.full(n, -1, dtype=np.int64)
    exact_by_read = np.zeros(n, dtype=np.bool_)

    aligned_pairs = read.aligned_pairs
    read_mask = aligned_pairs[:, 0] >= 0
    read_pos = aligned_pairs[read_mask, 0]
    ref_pos = aligned_pairs[read_mask, 1]
    aligned_seq = read.full_seq[read_mask]

    exact_base = (
        (ref_pos >= 0)
        & (aligned_seq[:, 0] < 4)
        & (aligned_seq[:, 1] < 4)
        & (aligned_seq[:, 0] == aligned_seq[:, 1])
    )
    ref_pos_by_read[read_pos] = ref_pos
    exact_by_read[read_pos] = exact_base

    base_seq = read.get_base_seq(expanded=False)
    canonical_base = base_seq < 4

    max_error_prob = phred_threshold_to_error_prob(min_base_quality)
    if max_error_prob is None:
        quality_ok = np.ones(n, dtype=np.bool_)
    else:
        quality_ok = read.base_qual <= max_error_prob

    return build_exact_match_kmer_arrays_from_read_arrays(
        base_seq,
        ref_pos_by_read,
        exact_by_read & canonical_base,
        k=k,
        quality_ok=quality_ok,
        eligible_center_mask=eligible_center_mask,
    )


def build_polished_exact_match_kmer_arrays(
    read,
    polished_read,
    k=9,
    min_base_quality=0,
    eligible_center_mask=None,
):
    """Encode exact k-mers in polished-sequence coordinates.

    Polished bases are compared with reference bases from the original
    alignment. Quality constraints are transferred through ``p_from_read``;
    polished positions without an originating read base fail an enabled
    quality threshold.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        K-mer IDs and validity mask aligned to the polished sequence.
    """
    if k % 2 != 1:
        raise ValueError(f"k must be odd so each k-mer has one center base, got {k}")

    base_seq = polished_read.sequence
    ref_positions = polished_read.p_to_ref
    n = len(base_seq)
    kmer_ids = np.full(n, -1, dtype=np.int64)
    kmer_valid = np.zeros(n, dtype=np.bool_)
    if n < k:
        return kmer_ids, kmer_valid

    ref_base_by_position = {}
    for aligned_pair, aligned_seq in zip(read.aligned_pairs, read.full_seq):
        ref_position = int(aligned_pair[1])
        ref_base = int(aligned_seq[1])
        if ref_position >= 0 and ref_base < 4:
            ref_base_by_position[ref_position] = ref_base

    exact = np.zeros(n, dtype=np.bool_)
    for idx, (base, ref_position) in enumerate(zip(base_seq, ref_positions)):
        exact[idx] = (
            int(base) < 4
            and int(ref_position) in ref_base_by_position
            and int(base) == ref_base_by_position[int(ref_position)]
        )

    max_error_prob = phred_threshold_to_error_prob(min_base_quality)
    quality_ok = np.ones(n, dtype=np.bool_)
    if max_error_prob is not None:
        quality_ok[:] = False
        for read_idx, polished_idx in enumerate(polished_read.p_from_read):
            if polished_idx >= 0:
                quality_ok[int(polished_idx)] = read.base_qual[read_idx] <= max_error_prob

    return build_exact_match_kmer_arrays_from_read_arrays(
        base_seq,
        ref_positions,
        exact,
        k=k,
        quality_ok=quality_ok,
        eligible_center_mask=eligible_center_mask,
    )


def chunk_read_base_array(read, values, padding_needed, move_table_chunks, size, step, pad_value):
    """Project a read-coordinate array into base-aligned overlapping chunks."""
    if has_inferred_boundaries(read):
        return slice_base_values(read, values, size, step)
    expanded_values = torch.from_numpy(values[read.indexes])
    expanded_values = F.pad(expanded_values, (0, padding_needed), mode='constant', value=pad_value)
    value_chunks = expanded_values.unfold(dimension=0, size=size, step=step)
    return [value_chunks[i][move_table_chunks[i]] for i in range(len(move_table_chunks))]


def get_read_chunks_with_kmers(
    read,
    size,
    step,
    kmer_size=9,
    min_base_quality=0,
    eligible_center_mask=None,
    polished_read=None,
    md_bam_read=None,
):
    """Chunk a read and attach exact-match k-mer IDs to each base position.

    The raw-read path expands values through the signal-to-base index map. The
    polished path instead uses the polished-to-read/reference mappings returned
    by labelled chunking. When ``md_bam_read`` is provided, exact-match k-mers
    are decoded from that record's MD+CIGAR rather than from FASTA bases stored
    on ``read``.

    Returns
    -------
    tuple
        Signal, sequence, move-table, reference-position, k-mer-ID, and
        k-mer-valid chunk collections in matching order.
    """
    if polished_read is not None:
        if md_bam_read is not None:
            raise ValueError("Polished prototype extraction requires a reference FASTA")
        signal_chunks, base_seq_chunks, init_move_table_chunks, pos_padded, read_pos_padded = get_read_chunks_labelled(
            read, size, step, polished_read=polished_read,
        )
        kmer_ids, kmer_valid = build_polished_exact_match_kmer_arrays(
            read,
            polished_read,
            k=kmer_size,
            min_base_quality=min_base_quality,
            eligible_center_mask=eligible_center_mask,
        )
        kmer_ids = torch.from_numpy(kmer_ids)
        kmer_valid = torch.from_numpy(kmer_valid)
        pos_chunks = []
        kmer_id_chunks = []
        kmer_valid_chunks = []
        for chunk_idx, seq_chunk in enumerate(base_seq_chunks):
            chunk_len = len(seq_chunk)
            polished_positions = read_pos_padded[chunk_idx, :chunk_len]
            pos_chunks.append(pos_padded[chunk_idx, :chunk_len])
            kmer_id_chunks.append(kmer_ids[polished_positions])
            kmer_valid_chunks.append(kmer_valid[polished_positions])
        return (
            signal_chunks.transpose(1, 2),
            base_seq_chunks,
            init_move_table_chunks,
            pos_chunks,
            kmer_id_chunks,
            kmer_valid_chunks,
        )

    if has_inferred_boundaries(read):
        signal_chunks, base_seq_chunks, pos_chunks, _ = inferred_chunks(read, size, step)
        init_move_table_chunks = move_table_chunks = None
        padding_needed = 0
    else:
        signal, base_seq, move_table, padding_needed = pad_read_components(read, size, step)
        signal_chunks, base_seq_chunks, init_move_table_chunks, move_table_chunks = chunk_padded_components(
            signal, base_seq, move_table, size, step
        )

        aligned_pairs = read.aligned_pairs
        read_wise_aligned_pos = aligned_pairs[aligned_pairs[:, 0] >= 0][:, 1]
        read_wise_aligned_pos = torch.from_numpy(read_wise_aligned_pos[read.indexes])
        read_wise_aligned_pos = F.pad(read_wise_aligned_pos, (0, padding_needed), mode='constant', value=0)
        read_wise_aligned_pos_chunks = read_wise_aligned_pos.unfold(dimension=0, size=size, step=step)
        pos_chunks = [read_wise_aligned_pos_chunks[i][move_table_chunks[i]] for i in range(len(move_table_chunks))]

    if md_bam_read is None:
        kmer_ids, kmer_valid = build_exact_match_kmer_arrays(
            read,
            k=kmer_size,
            min_base_quality=min_base_quality,
            eligible_center_mask=eligible_center_mask,
        )
    else:
        max_error_prob = phred_threshold_to_error_prob(min_base_quality)
        quality_ok = (
            None
            if max_error_prob is None
            else read.base_qual <= max_error_prob
        )
        kmer_ids, kmer_valid = build_md_exact_match_kmer_arrays(
            md_bam_read,
            read.get_base_seq(expanded=False),
            k=kmer_size,
            quality_ok=quality_ok,
            eligible_center_mask=eligible_center_mask,
        )
    kmer_id_chunks = chunk_read_base_array(
        read, kmer_ids, padding_needed, move_table_chunks, size, step, pad_value=-1
    )
    kmer_valid_chunks = chunk_read_base_array(
        read, kmer_valid, padding_needed, move_table_chunks, size, step, pad_value=False
    )

    return (
        signal_chunks.transpose(1, 2),
        base_seq_chunks,
        init_move_table_chunks,
        pos_chunks,
        kmer_id_chunks,
        kmer_valid_chunks,
    )


class KmerPrototypeAlignedBamIterableDataset(IterableDataset):
    """Stream aligned read chunks whose centers have exact reference k-mers.

    Reads are normalized and chunked using the aligned-data pipeline. Optional
    reference-position or motif restrictions determine eligible centers, while
    exact-match and quality filters determine whether each center contributes
    to a prototype. Iteration yields buffered modal batches for collation.
    """
    def __init__(
        self,
        *args,
        kmer_size=9,
        min_base_quality=0,
        read_based_motif=False,
        **kwargs,
    ):
        super().__init__()
        self.bam_list = kwargs.pop('bam_list') if 'bam_list' in kwargs else args[0]
        self.size = kwargs.pop('size')
        self.step = kwargs.pop('step')
        self.preprocessing = resolve_preprocessing(
            preprocessing=kwargs.pop('preprocessing', None), chemistry=kwargs.pop('chemistry', None),
            seq_type=kwargs.pop('seq_type', None), pore_type=kwargs.pop('pore_type', None),
            kmer_table=kwargs.pop('kmer_table', None),
        )
        self.seq_type = self.preprocessing.seq_type
        self.kmer_table = self.preprocessing.kmer_table
        self.reads_per_chunk = kwargs.pop('reads_per_chunk', 1000)
        self.min_read_length = kwargs.pop('min_read_length', 100)
        self.min_qscore = kwargs.pop('min_qscore', 10)
        self.pore_type = self.preprocessing.pore_type
        self.base_seed = kwargs.pop('base_seed', 1234)
        self.rank = kwargs.pop('rank', 0)
        self.world_size = kwargs.pop('world_size', 1)
        self.chunking_mode = validate_chunking_mode(kwargs.pop('chunking_mode', 'auto'))
        self.expected_ts_dim = kwargs.pop('expected_ts_dim', None)
        if kwargs:
            raise TypeError(f"Unexpected dataset kwargs: {sorted(kwargs)}")
        self.kmer_size = kmer_size
        self.min_base_quality = min_base_quality
        self.read_based_motif = read_based_motif
        self.parsed_read_motifs = None
        if self.read_based_motif:
            motif = self.bam_list[0].motif
            self.parsed_read_motifs = parse_motif_specs(
                motif=motif,
                motif_label='0',
                num_classes=1,
            )

    def _new_buffer(self):
        return {
            'signal': [],
            'seq': [],
            'read_name': [],
            'position': [],
            'strand': [],
            'kmer_id': [],
            'kmer_valid': [],
        }

    def _yield_buffer(self, data):
        signals = torch.cat(data['signal'], dim=0)
        percentiles = get_percentiles(signals)

        if data['position']:
            pos_cat = torch.cat(data['position'])
            valid_pos = pos_cat[pos_cat > 0]
            min_start = valid_pos.min().item() if valid_pos.numel() > 0 else 1e12
        else:
            min_start = 1e12

        for i in range(len(data['seq'])):
            if len(data['seq'][i]) >= 64:
                yield (
                    data['seq'][i],
                    signals[i].to(dtype=torch.float16),
                    percentiles[i].to(dtype=torch.float16),
                    data['read_name'][i],
                    data['position'][i],
                    min_start,
                    data['strand'][i],
                    data['kmer_id'][i],
                    data['kmer_valid'][i],
                )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        global_rank = self.rank
        global_world_size = self.world_size
        global_worker_id = global_rank * num_workers + worker_id
        global_worker_count = max(global_world_size * num_workers, 1)

        seed = self.base_seed + 1000003 * (global_worker_id + 1)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)

        my_files = []
        for bam_idx in range(global_worker_id, len(self.bam_list), global_worker_count):
            my_files.append(self.bam_list[bam_idx])
        if len(my_files) == 0:
            return

        preprocess_cache = {}
        polish_variant_mask_cache = {}
        data = self._new_buffer()

        for bam_sample in my_files:
            bam_path, chrom_info = bam_sample.bam_path, bam_sample.chrom_info
            flag_info = bam_sample.flag
            chroms = tuple(x[0] for x in chrom_info)

            with pysam.AlignmentFile(bam_path, 'rb', check_sq=False) as bam:
                print(
                    f'{str(datetime.datetime.now())}: '
                    f'{format_iterable_bam_log(self, global_rank, worker_id, bam_path)}',
                    flush=True,
                )

                if bam_sample.ref is None:
                    ref_seq_dict, labelled_arrays = None, {}
                else:
                    preprocess_args = (
                        bam_sample.ref,
                        self.seq_type,
                        '' if self.read_based_motif else bam_sample.pos_file,
                        '' if self.read_based_motif else bam_sample.motif,
                        bam_sample.motif_label,
                        chroms,
                    )
                    if preprocess_args in preprocess_cache:
                        ref_seq_dict, labelled_arrays = preprocess_cache[preprocess_args]
                    else:
                        ref_seq_dict, labelled_arrays = pre_process(*preprocess_args)
                        preprocess_cache[preprocess_args] = (ref_seq_dict, labelled_arrays)

                for contig, start, end in chrom_info:
                    for bam_read in bam.fetch(contig, start=start, end=end, until_eof=True):
                        if (
                            bam_read.flag & (0x900 | flag_info) == 0
                            and passes_min_qscore(bam_read, self.min_qscore)
                            and len(bam_read.seq) >= self.min_read_length
                        ):
                            if ref_seq_dict is None:
                                read = build_md_backed_read(
                                    bam_read,
                                    self.seq_type,
                                    self.kmer_table,
                                    expected_ts_dim=self.expected_ts_dim, chunking_mode=self.chunking_mode,
                                )
                            else:
                                read_info = get_read_info(
                                    bam_read, ref_seq_dict, self.seq_type, self.kmer_table,
                                    expected_ts_dim=self.expected_ts_dim, chunking_mode=self.chunking_mode,
                                )
                                read = Read(*read_info)

                            polished_read = (
                                get_polished_read_for_sample(
                                    bam_read, bam_sample, polish_variant_mask_cache,
                                )
                                if bam_sample.polish
                                else None
                            )

                            if polished_read is not None and self.read_based_motif:
                                polished_sequence = ''.join(
                                    num_to_base_map.get(int(base), 'N')
                                    for base in polished_read.sequence
                                )
                                target_positions, _ = get_read_motif_targets(
                                    read_sequence=polished_sequence,
                                    parsed_motifs=self.parsed_read_motifs,
                                    num_classes=1,
                                )
                                eligible_center_mask = np.zeros(
                                    len(polished_read.sequence), dtype=np.bool_,
                                )
                                eligible_center_mask[target_positions] = True
                            elif polished_read is not None and (bam_sample.pos_file or bam_sample.motif):
                                target_positions, _ = labelled_arrays[read.ref_name][read.is_reverse]
                                eligible_center_mask = np.isin(
                                    polished_read.p_to_ref, target_positions,
                                )
                            elif self.read_based_motif:
                                eligible_center_mask = build_read_motif_center_mask(
                                    read,
                                    self.parsed_read_motifs,
                                )
                            elif bam_sample.pos_file or bam_sample.motif:
                                eligible_center_mask = build_reference_label_center_mask(
                                    read,
                                    labelled_arrays,
                                )
                            else:
                                eligible_center_mask = None

                            if eligible_center_mask is not None and not eligible_center_mask.any():
                                continue

                            try:
                                read.signal = normalize_read_signal(read, self.pore_type)
                            except DegenerateQuantileNormalizationError:
                                continue
                            chunk_result = get_read_chunks_with_kmers(
                                read,
                                self.size,
                                self.step,
                                kmer_size=self.kmer_size,
                                min_base_quality=self.min_base_quality,
                                eligible_center_mask=eligible_center_mask,
                                polished_read=polished_read,
                                md_bam_read=bam_read if ref_seq_dict is None else None,
                            )
                            signal_chunks, base_seq_chunks, _, pos_chunks, kmer_id_chunks, kmer_valid_chunks = chunk_result

                            if start is not None and end is not None:
                                pos_tensor = pad_sequence(pos_chunks, batch_first=True, padding_value=-1)
                                keep = torch.any((pos_tensor >= start) & (pos_tensor <= end), dim=1)
                            else:
                                keep = torch.ones(len(base_seq_chunks), dtype=torch.bool)

                            if eligible_center_mask is not None:
                                keep = keep & torch.tensor(
                                    [chunk.any().item() for chunk in kmer_valid_chunks],
                                    dtype=torch.bool,
                                )

                            if keep.any():
                                keep_idx = keep.nonzero(as_tuple=True)[0].tolist()
                                data['signal'].append(signal_chunks[keep])
                                data['seq'].extend([base_seq_chunks[i] for i in keep_idx])
                                data['read_name'].extend([read.read_name] * len(keep_idx))
                                data['position'].extend([pos_chunks[i] for i in keep_idx])
                                data['strand'].extend([0 if read.is_forward else 1] * len(keep_idx))
                                data['kmer_id'].extend([kmer_id_chunks[i] for i in keep_idx])
                                data['kmer_valid'].extend([kmer_valid_chunks[i] for i in keep_idx])

                            if len(data['signal']) == self.reads_per_chunk:
                                yield from self._yield_buffer(data)
                                data = self._new_buffer()

        if len(data['signal']) > 0:
            yield from self._yield_buffer(data)


def kmer_prototype_collate_fn(batch):
    """Pad modal chunk batches and their aligned k-mer ID/validity tensors."""
    seqs_batch, signals_batch, percentiles_batch = [], [], []
    read_name_batch, positions_batch, strand_batch = [], [], []
    kmer_id_batch, kmer_valid_batch = [], []
    min_start = 1e12

    for item in batch:
        item_seqs, item_signals, item_percentiles, item_read_names, item_positions, item_min_start, item_strand, item_kmer_ids, item_kmer_valid = item
        min_start = min(min_start, item_min_start)
        seqs_batch.append(item_seqs)
        signals_batch.append(item_signals)
        percentiles_batch.append(item_percentiles)
        read_name_batch.append(item_read_names)
        positions_batch.append(item_positions)
        strand_batch.append(item_strand)
        kmer_id_batch.append(item_kmer_ids)
        kmer_valid_batch.append(item_kmer_valid)

    batch_out = collate_modal_aligned_batch(
        seqs_batch,
        signals_batch,
        percentiles_batch,
        read_name_batch,
        positions_batch,
        strand_batch,
        min_start,
    )
    seqs_batch_padded = batch_out[0]

    kmer_id_batch_padded = pad_sequence(kmer_id_batch, batch_first=True, padding_value=-1)
    kmer_valid_batch_padded = pad_sequence(kmer_valid_batch, batch_first=True, padding_value=False)
    pad_needed = seqs_batch_padded.shape[1] - kmer_id_batch_padded.shape[1]
    if pad_needed > 0:
        kmer_id_batch_padded = F.pad(kmer_id_batch_padded, (0, pad_needed), value=-1)
        kmer_valid_batch_padded = F.pad(kmer_valid_batch_padded, (0, pad_needed), value=False)

    return batch_out + (
        kmer_id_batch_padded.to(torch.long),
        kmer_valid_batch_padded.to(torch.bool),
    )


def kmer_index_to_string(kmer_idx, k=9):
    """Decode a two-bit integer k-mer representation to an A/C/G/T string."""
    bases = "ACGT"
    return "".join(bases[(int(kmer_idx) >> (2 * offset)) & 3] for offset in range(k - 1, -1, -1))


class KmerPrototypeAccumulator:
    """Accumulate per-k-mer embedding means and optional reservoir samples.

    Float32 sums and total counts live on ``device``. Reservoir samples live on
    CPU in bfloat16 and use random priorities to maintain an unbiased fixed-size
    sample for each k-mer independently. Finalization consumes the reservoir.
    """
    def __init__(self, dim, k=9, device='cpu', reservoir_size=0, reservoir_seed=0):
        if reservoir_size < 0:
            raise ValueError(f"reservoir_size must be non-negative, got {reservoir_size}")
        self.dim = dim
        self.k = k
        self.num_kmers = 4 ** k
        self.device = torch.device(device)
        self.sum = torch.zeros((self.num_kmers, dim), dtype=torch.float32, device=self.device)
        self.count = torch.zeros(self.num_kmers, dtype=torch.long, device=self.device)
        self.reservoir_size = int(reservoir_size)
        self.reservoir_generator = torch.Generator(device='cpu')
        self.reservoir_generator.manual_seed(int(reservoir_seed))
        self._reservoir = {}
        self._finalized = False

    def update(self, kmer_ids, embeddings):
        """Add ``(n,)`` k-mer IDs and corresponding ``(n, dim)`` embeddings."""
        if self._finalized:
            raise RuntimeError("Cannot update a finalized prototype accumulator")
        if kmer_ids.numel() == 0:
            return
        kmer_ids = kmer_ids.to(self.device, non_blocking=True).long()
        sum_embeddings = embeddings.to(self.device, non_blocking=True).float()
        self.sum.index_add_(0, kmer_ids, sum_embeddings)
        del sum_embeddings
        self.count.index_add_(0, kmer_ids, torch.ones_like(kmer_ids, dtype=torch.long, device=self.device))
        if self.reservoir_size > 0:
            self._update_reservoir(kmer_ids, embeddings)

    def _update_reservoir(self, kmer_ids, embeddings):
        """Keep an unbiased fixed-size sample using per-observation priorities."""
        ids_cpu = kmer_ids.detach().to(device='cpu', dtype=torch.long).reshape(-1)
        embeddings_cpu = embeddings.detach().to(device='cpu', dtype=torch.bfloat16).reshape(-1, self.dim)
        priorities = torch.rand(
            len(ids_cpu),
            dtype=torch.float64,
            generator=self.reservoir_generator,
        )

        order = torch.argsort(ids_cpu, stable=True)
        ids_sorted = ids_cpu[order]
        embeddings_sorted = embeddings_cpu[order]
        priorities_sorted = priorities[order]
        unique_ids, group_counts = torch.unique_consecutive(ids_sorted, return_counts=True)

        offset = 0
        for kmer_id, group_count in zip(unique_ids.tolist(), group_counts.tolist()):
            group_slice = slice(offset, offset + group_count)
            candidate_embeddings = embeddings_sorted[group_slice]
            candidate_priorities = priorities_sorted[group_slice]
            offset += group_count

            existing = self._reservoir.get(kmer_id)
            if existing is not None:
                existing_priorities, existing_embeddings = existing
                candidate_priorities = torch.cat((existing_priorities, candidate_priorities))
                candidate_embeddings = torch.cat((existing_embeddings, candidate_embeddings))

            if len(candidate_priorities) > self.reservoir_size:
                keep = torch.topk(
                    candidate_priorities,
                    k=self.reservoir_size,
                    sorted=False,
                ).indices
                candidate_priorities = candidate_priorities[keep]
                candidate_embeddings = candidate_embeddings[keep]
            elif existing is None:
                # Slices otherwise retain the entire sorted batch allocation,
                # so rare k-mers could pin far more memory than their samples.
                candidate_priorities = candidate_priorities.clone()
                candidate_embeddings = candidate_embeddings.clone()

            self._reservoir[kmer_id] = (candidate_priorities, candidate_embeddings)

    def finalize(self, min_count=1):
        """Materialize prototypes whose total observation count meets a limit.

        Returns a mapping containing IDs, strings, counts, and float32 means.
        When reservoir sampling is enabled, bfloat16 samples and per-k-mer
        offsets/counts are included as well. This is a terminal operation:
        reservoir buffers are consumed, and later updates/finalizations raise.
        The output is allocated once and filled incrementally while releasing
        source buffers, avoiding a full live reservoir plus concatenation copy.
        """
        if self._finalized:
            raise RuntimeError("Prototype accumulator has already been finalized")
        self._finalized = True
        counts = self.count.cpu()
        keep = counts >= min_count
        kmer_ids = keep.nonzero(as_tuple=True)[0]
        kept_counts = counts[kmer_ids]
        sums = self.sum.cpu()[kmer_ids]
        mean_embeddings = sums / torch.clamp_min(kept_counts, 1).unsqueeze(1)
        result = {
            'kmer_ids': kmer_ids,
            'kmers': np.asarray([kmer_index_to_string(i, self.k) for i in kmer_ids.tolist()], dtype=object),
            'counts': kept_counts,
            'mean_embeddings': mean_embeddings,
        }
        if self.reservoir_size > 0:
            retained_ids = kmer_ids.tolist()
            retained_set = set(retained_ids)
            for kmer_id in list(self._reservoir):
                if kmer_id not in retained_set:
                    del self._reservoir[kmer_id]
            reservoir_counts = torch.tensor([
                len(self._reservoir[kmer_id][1]) if kmer_id in self._reservoir else 0
                for kmer_id in retained_ids
            ], dtype=torch.long)
            reservoir_offsets = torch.zeros(len(retained_ids) + 1, dtype=torch.long)
            torch.cumsum(reservoir_counts, dim=0, out=reservoir_offsets[1:])
            # Empty allocation avoids touching every output page up front.
            # Copy one independently owned buffer at a time and release it
            # before touching the next output slice.
            reservoir_embeddings = torch.empty(
                (int(reservoir_offsets[-1]), self.dim), dtype=torch.bfloat16,
            )
            for index, kmer_id in enumerate(retained_ids):
                entry = self._reservoir.pop(kmer_id, None)
                if entry is not None:
                    start = int(reservoir_offsets[index])
                    end = int(reservoir_offsets[index + 1])
                    reservoir_embeddings[start:end].copy_(entry[1])
                    del entry
            result.update({
                'reservoir_embeddings': reservoir_embeddings,
                'reservoir_offsets': reservoir_offsets,
                'reservoir_counts': reservoir_counts,
            })
        return result


UNION_CHUNK_BYTES = 16 * 1024 * 1024


def _embedding_row_chunks(embeddings):
    """Yield row ranges with at most roughly 16 MiB of float32 values."""
    row_bytes = max(1, embeddings.shape[1]) * max(4, embeddings.element_size())
    rows = max(1, UNION_CHUNK_BYTES // row_bytes)
    for start in range(0, len(embeddings), rows):
        yield start, min(start + rows, len(embeddings))


class _UnionProgress:
    """Print flushed stage progress at start, completion, and every 10 seconds."""

    def __init__(self, label, total, unit, enabled):
        self.label, self.total, self.unit = label, total, unit
        self.enabled = enabled
        self.started = time.monotonic()
        self.last_print = self.started
        if enabled:
            self.update(0, force=True)

    def update(self, completed, force=False):
        """Report completed items, throttling intermediate updates by elapsed time."""
        if not self.enabled:
            return
        now = time.monotonic()
        if force or completed == self.total or now - self.last_print >= 10:
            percent = 100 * completed / self.total if self.total else 100
            print(
                f"[union] {self.label}: {completed:,}/{self.total:,} {self.unit} "
                f"({percent:.1f}%), elapsed {now - self.started:.1f}s",
                flush=True,
            )
            self.last_print = now


def _embeddings_are_finite(embeddings, progress=None):
    """Check all values without allocating a reservoir-sized boolean tensor."""
    for start, end in _embedding_row_chunks(embeddings):
        if not bool(torch.isfinite(embeddings[start:end]).all()):
            return False
        if progress is not None:
            progress.update(end)
    return True


def _load_union_input(path):
    """Map a torch.save ZIP payload without eagerly loading its tensor storages."""
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def validate_union_prototype_payload(payload, path, progress=False):
    """Validate and normalize one reservoir-prototype payload for unioning.

    Returns normalized tensors plus k-mer size, embedding key, and embedding
    dimension. Shape, offset, count, and metadata inconsistencies raise
    ``ValueError`` identifying ``path``. With ``progress=True``, report reservoir
    validation progress in embedding rows.
    """
    required = {
        'kmer_ids',
        'counts',
        'mean_embeddings',
        'reservoir_embeddings',
        'reservoir_offsets',
        'reservoir_counts',
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            f"Prototype file {path} is not a saved reservoir prototype; "
            f"missing keys: {', '.join(missing)}"
        )

    kmer_ids = payload['kmer_ids'].long().reshape(-1)
    counts = payload['counts'].long().reshape(-1)
    means = payload['mean_embeddings'].float()
    # Keep the on-disk dtype; convert only the slice being copied.
    reservoir_embeddings = payload['reservoir_embeddings']
    reservoir_offsets = payload['reservoir_offsets'].long().reshape(-1)
    reservoir_counts = payload['reservoir_counts'].long().reshape(-1)
    metadata = payload.get('metadata', {})
    kmer_size = int(metadata.get('kmer_size', 9))
    embedding_key = metadata.get('embedding_key', 'contextual_base_seq')

    if embedding_key not in EMBEDDING_KEY_CHOICES:
        raise ValueError(
            f"Prototype file {path} records unsupported "
            f"embedding_key={embedding_key!r}"
        )
    if means.ndim != 2:
        raise ValueError(
            f"Prototype file {path} mean_embeddings must be two-dimensional, "
            f"got shape {tuple(means.shape)}"
        )
    if reservoir_embeddings.ndim != 2:
        raise ValueError(
            f"Prototype file {path} reservoir_embeddings must be two-dimensional, "
            f"got shape {tuple(reservoir_embeddings.shape)}"
        )
    if len(kmer_ids) != len(counts) or len(kmer_ids) != len(means):
        raise ValueError(
            f"Prototype file {path} has inconsistent kmer_ids, counts, and "
            "mean_embeddings lengths"
        )
    if len(reservoir_counts) != len(kmer_ids) or len(reservoir_offsets) != len(kmer_ids) + 1:
        raise ValueError(
            f"Prototype file {path} has inconsistent reservoir offsets/counts"
        )
    if reservoir_embeddings.shape[1] != means.shape[1]:
        raise ValueError(
            f"Prototype file {path} reservoir dimension "
            f"{reservoir_embeddings.shape[1]} does not match mean dimension "
            f"{means.shape[1]}"
        )
    if len(torch.unique(kmer_ids)) != len(kmer_ids):
        raise ValueError(f"Prototype file {path} contains duplicate k-mer IDs")
    if (counts <= 0).any():
        raise ValueError(f"Prototype file {path} contains a non-positive total count")
    if (reservoir_counts < 0).any() or (reservoir_counts > counts).any():
        raise ValueError(
            f"Prototype file {path} has a reservoir count outside [0, total count]"
        )
    if reservoir_offsets[0].item() != 0:
        raise ValueError(f"Prototype file {path} reservoir offsets must start at zero")
    if (reservoir_offsets[1:] < reservoir_offsets[:-1]).any():
        raise ValueError(
            f"Prototype file {path} reservoir offsets must be non-decreasing"
        )
    if reservoir_offsets[-1].item() != len(reservoir_embeddings):
        raise ValueError(
            f"Prototype file {path} final reservoir offset does not match the "
            "number of reservoir embeddings"
        )
    if not torch.equal(
        reservoir_offsets[1:] - reservoir_offsets[:-1],
        reservoir_counts,
    ):
        raise ValueError(
            f"Prototype file {path} reservoir_counts do not match its offsets"
        )
    validation_progress = _UnionProgress(
        f"Validating reservoir {Path(path).name}", len(reservoir_embeddings),
        'embedding rows', progress,
    )
    if not _embeddings_are_finite(means) or not _embeddings_are_finite(
        reservoir_embeddings, validation_progress,
    ):
        raise ValueError(f"Prototype file {path} contains non-finite embeddings")

    num_possible_kmers = 4 ** kmer_size
    if ((kmer_ids < 0) | (kmer_ids >= num_possible_kmers)).any():
        raise ValueError(
            f"Prototype file {path} contains a k-mer ID outside "
            f"[0, {num_possible_kmers}) for kmer_size={kmer_size}"
        )
    recorded_dim = metadata.get('embedding_dim')
    if recorded_dim is not None and int(recorded_dim) != means.shape[1]:
        raise ValueError(
            f"Prototype file {path} records embedding_dim={recorded_dim}, "
            f"but its embeddings have dimension {means.shape[1]}"
        )

    return {
        'kmer_ids': kmer_ids,
        'counts': counts,
        'mean_embeddings': means,
        'reservoir_embeddings': reservoir_embeddings,
        'reservoir_offsets': reservoir_offsets,
        'reservoir_counts': reservoir_counts,
        'metadata': metadata,
        'kmer_size': kmer_size,
        'embedding_key': embedding_key,
        'embedding_dim': means.shape[1],
    }


def union_reservoir_prototype_files(paths, min_count=1, output_path=None, progress=False):
    """Union counts, weighted means, and all reservoir rows in source order.

    Inputs must be memory-mappable torch.save ZIP files. Reservoir validation
    and dtype conversion use bounded row chunks. If ``output_path`` is supplied,
    write reservoirs directly into that .pt file via a shared mapping;
    otherwise return an in-memory reservoir. Small statistics scale with the
    number of distinct k-mers, while mapped pages are reclaimable by the OS.
    Reservoirs retain their input dtype; mixed input dtypes use torch's common
    promoted dtype to avoid downcasting. Means accumulate in float64 and are
    saved as float32 independently of the reservoir dtype.
    Set ``progress=True`` for flushed, periodic validation and per-file k-mer
    progress. K-mer totals refer to source entries, including filtered entries
    during copying, rather than distinct k-mers across all inputs.
    The output path must differ from every input; callers should use a temporary
    path and publish it only after success. Returned disk-backed tensors retain
    their mapping. Files created with skip_data are readable by torch.load but
    do not contain valid ZIP CRC checksums for subsequently populated storages.
    """
    if len(paths) == 0:
        raise ValueError("At least one prototype file is required for union")
    if min_count < 1:
        raise ValueError(f"min_count must be at least one, got {min_count}")

    resolved_paths = [str(Path(path).resolve(strict=True)) for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("--union_prototype_files must not contain duplicate paths")

    if output_path is not None and str(Path(output_path).resolve()) in resolved_paths:
        raise ValueError("Union output must not overwrite an input prototype file")

    expected_kmer_size = None
    expected_embedding_key = None
    expected_embedding_dim = None
    reservoir_dtype = None
    total_counts = {}
    weighted_sums = {}
    merged_reservoir_counts = {}
    source_summaries = []

    for file_index, path in enumerate(resolved_paths, 1):
        if progress:
            print(f"[union] Reading input {file_index}/{len(resolved_paths)}: {path}", flush=True)
        payload = _load_union_input(path)
        info = validate_union_prototype_payload(payload, path, progress=progress)
        source_dtype = info['reservoir_embeddings'].dtype
        reservoir_dtype = (
            source_dtype if reservoir_dtype is None
            else torch.promote_types(reservoir_dtype, source_dtype)
        )
        if expected_kmer_size is None:
            expected_kmer_size = info['kmer_size']
            expected_embedding_key = info['embedding_key']
            expected_embedding_dim = info['embedding_dim']
        elif info['kmer_size'] != expected_kmer_size:
            raise ValueError(
                f"Prototype file {path} uses kmer_size={info['kmer_size']}, "
                f"expected {expected_kmer_size}"
            )
        elif info['embedding_key'] != expected_embedding_key:
            raise ValueError(
                f"Prototype file {path} uses embedding_key="
                f"{info['embedding_key']!r}, expected {expected_embedding_key!r}"
            )
        elif info['embedding_dim'] != expected_embedding_dim:
            raise ValueError(
                f"Prototype file {path} uses embedding dimension "
                f"{info['embedding_dim']}, expected {expected_embedding_dim}"
            )

        mean_progress = _UnionProgress(
            f"Combining counts/means, input {file_index}/{len(resolved_paths)}",
            len(info['kmer_ids']), 'k-mers', progress,
        )
        for row_index, kmer_id in enumerate(info['kmer_ids'].tolist()):
            count = int(info['counts'][row_index].item())
            contribution = info['mean_embeddings'][row_index].double() * count
            if kmer_id in total_counts:
                total_counts[kmer_id] += count
                weighted_sums[kmer_id].add_(contribution)
                merged_reservoir_counts[kmer_id] += int(
                    info['reservoir_counts'][row_index].item()
                )
            else:
                total_counts[kmer_id] = count
                weighted_sums[kmer_id] = contribution.clone()
                merged_reservoir_counts[kmer_id] = int(
                    info['reservoir_counts'][row_index].item()
                )
            mean_progress.update(row_index + 1)

        source_summaries.append({
            'path': path,
            'saved_kmer_count': len(info['kmer_ids']),
            'total_observation_count': int(info['counts'].sum().item()),
            'saved_reservoir_embedding_count': len(info['reservoir_embeddings']),
            'metadata': info['metadata'],
        })
        del payload, info

    kept_kmer_ids = sorted(
        kmer_id
        for kmer_id, count in total_counts.items()
        if count >= min_count
    )
    if len(kept_kmer_ids) == 0:
        raise ValueError(f"No unioned k-mers meet min_count={min_count}")

    kmer_ids = torch.tensor(kept_kmer_ids, dtype=torch.long)
    counts = torch.tensor(
        [total_counts[kmer_id] for kmer_id in kept_kmer_ids],
        dtype=torch.long,
    )
    mean_embeddings = torch.stack([
        (weighted_sums[kmer_id] / total_counts[kmer_id]).float()
        for kmer_id in kept_kmer_ids
    ])
    reservoir_counts = torch.tensor(
        [merged_reservoir_counts[kmer_id] for kmer_id in kept_kmer_ids],
        dtype=torch.long,
    )
    reservoir_offsets = torch.zeros(len(kmer_ids) + 1, dtype=torch.long)
    reservoir_offsets[1:] = torch.cumsum(reservoir_counts, dim=0)
    reservoir_embeddings = torch.empty(
        (int(reservoir_offsets[-1].item()), expected_embedding_dim),
        dtype=reservoir_dtype,
    )

    # Preserve verified provenance only when every union input identifies the
    # same checkpoint; numerical union behavior is unchanged for older files.
    checkpoint_hashes = {s['metadata'].get('checkpoint_sha256') for s in source_summaries}
    checkpoint_sha256 = next(iter(checkpoint_hashes)) if len(checkpoint_hashes) == 1 else None

    profile = validate_artifact_preprocessing(source_summaries)
    result = {
        'kmer_ids': kmer_ids,
        'kmers': np.asarray([
            kmer_index_to_string(kmer_id, expected_kmer_size)
            for kmer_id in kept_kmer_ids
        ], dtype=object),
        'counts': counts,
        'mean_embeddings': mean_embeddings,
        'reservoir_embeddings': reservoir_embeddings,
        'reservoir_offsets': reservoir_offsets,
        'reservoir_counts': reservoir_counts,
        'metadata': {
            'preprocessing': profile.to_dict() if profile is not None else None,
            'operation': 'uncapped_reservoir_union',
            'checkpoint_sha256': checkpoint_sha256,
            'source_prototype_files': resolved_paths,
            'source_summaries': source_summaries,
            'source_file_count': len(resolved_paths),
            'kmer_size': expected_kmer_size,
            'min_count': min_count,
            'reservoir_size': None,
            'reservoir_uncapped_union': True,
            'embedding_key': expected_embedding_key,
            'embedding_dim': expected_embedding_dim,
            'total_updates': int(counts.sum().item()),
            'saved_reservoir_embedding_count': len(reservoir_embeddings),
        },
    }

    del weighted_sums
    if output_path is not None:
        if progress:
            gib = reservoir_embeddings.numel() * reservoir_embeddings.element_size() / 1024**3
            print(f"[union] Preparing output: {len(kmer_ids):,} k-mers, "
                  f"{gib:.2f} GiB reservoir ({reservoir_dtype})", flush=True)
        # Reserve tensor storage in the final archive without touching the large
        # empty reservoir allocation. Populate small tensors skipped by save too.
        with torch.serialization.skip_data():
            torch.save(result, output_path)
        with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
            mapped = _load_union_input(output_path)
        for key, value in result.items():
            if isinstance(value, torch.Tensor) and key != 'reservoir_embeddings':
                mapped[key].copy_(value)
        result = mapped
        reservoir_embeddings = result['reservoir_embeddings']

    output_index_by_kmer = {
        kmer_id: output_index
        for output_index, kmer_id in enumerate(kept_kmer_ids)
    }
    write_offsets = reservoir_offsets[:-1].clone()
    for file_index, path in enumerate(resolved_paths, 1):
        payload = _load_union_input(path)
        source_kmer_ids = payload['kmer_ids'].long().reshape(-1)
        source_offsets = payload['reservoir_offsets'].long().reshape(-1)
        source_embeddings = payload['reservoir_embeddings']
        copy_progress = _UnionProgress(
            f"Copying reservoirs, input {file_index}/{len(resolved_paths)}",
            len(source_kmer_ids), 'source k-mers processed (including filtered)', progress,
        )
        for source_index, kmer_id in enumerate(source_kmer_ids.tolist()):
            output_index = output_index_by_kmer.get(kmer_id)
            if output_index is None:
                copy_progress.update(source_index + 1)
                continue
            source_start = int(source_offsets[source_index].item())
            source_end = int(source_offsets[source_index + 1].item())
            row_count = source_end - source_start
            destination_start = int(write_offsets[output_index].item())
            destination_end = destination_start + row_count
            source_rows = source_embeddings[source_start:source_end]
            for start, end in _embedding_row_chunks(source_rows):
                reservoir_embeddings[destination_start + start:destination_start + end].copy_(
                    source_rows[start:end]
                )
            del source_rows
            write_offsets[output_index] = destination_end
            copy_progress.update(source_index + 1)
        del payload, source_kmer_ids, source_offsets, source_embeddings

    if not torch.equal(write_offsets, reservoir_offsets[1:]):
        raise RuntimeError("Internal error while concatenating unioned reservoirs")

    return result


def _union_abundance_summary(payload):
    """Summarize observation counts for saved k-mers, after min_count filtering.

    Counts include all observed embeddings, not just retained reservoir rows.
    Quantiles use linear interpolation. Abundance ties use ascending k-mer ID;
    each ranked list contains up to five entries. Only count arrays are read.
    """
    counts = payload['counts'].tolist()
    ids = payload['kmer_ids'].tolist()
    kmers = payload['kmers']
    quantiles = [1, 5, 25, 50, 75, 95, 99]
    values = np.percentile(np.asarray(counts, dtype=np.float64), quantiles)

    def ranked(reverse):
        order = sorted(range(len(counts)), key=lambda i: (
            -counts[i] if reverse else counts[i], ids[i],
        ))[:5]
        return [{'kmer': str(kmers[i]), 'count': counts[i]} for i in order]

    return {
        'scope': 'saved k-mers after min_count filtering; total observations, not reservoir sizes',
        'mean': sum(counts) / len(counts),
        'median': float(values[3]),
        'min': min(counts),
        'max': max(counts),
        'quantiles': {f'p{q:02d}': float(v) for q, v in zip(quantiles, values)},
        'most_abundant': ranked(True),
        'least_abundant': ranked(False),
    }


def run_union(args):
    """Union reservoir prototype files, save the payload, and print a summary."""
    output = Path(args.output).resolve()
    inputs = {Path(path).resolve() for path in args.union_prototype_files}
    if output in inputs:
        raise ValueError("Union output must not overwrite an input prototype file")
    prepare_run_outputs(args, command="prototype extract-kmer-prototypes union", files={"output": args.output})
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep an existing output intact if validation or copying fails. Only one
    # output-sized file is created; no extra reservoir scratch copy is needed.
    with tempfile.TemporaryDirectory(prefix='.prototype-union-', dir=output.parent) as workdir:
        staged = Path(workdir) / 'union.pt'
        payload = union_reservoir_prototype_files(
            args.union_prototype_files,
            min_count=args.min_count,
            output_path=staged,
            progress=True,
        )
        print('[union] Flushing output to disk...', flush=True)
        with staged.open('rb') as handle:
            os.fsync(handle.fileno())
        os.replace(staged, output)
    print(
        json.dumps({
            'mode': 'uncapped_reservoir_union',
            'source_file_count': len(args.union_prototype_files),
            'saved_kmers': len(payload['kmer_ids']),
            'total_observation_count': int(payload['counts'].sum().item()),
            'saved_reservoir_embedding_count': len(payload['reservoir_embeddings']),
            'embedding_key': payload['metadata']['embedding_key'],
            'output': args.output,
            'observations_per_kmer': _union_abundance_summary(payload),
        }, indent=2),
        flush=True,
    )
    return payload


def autocast_context(device):
    """Return bfloat16 CUDA autocast or a no-op context on other devices."""
    if torch.device(device).type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def process_model_batch(batch, model, accumulator, device, embedding_key, num_trim):
    """Infer selected embeddings for valid k-mer centers and accumulate them.

    Returns the number of centers added after padding, edge, and k-mer masks.
    """
    dna_input, ts_input, percentiles_batch, pad_base_seq, pad_ts, _read_name, _positions, _strand, _min_start, kmer_ids, kmer_valid = batch
    filter_mask_cpu = ignore_edges(pad_base_seq.bool(), num_trim) & kmer_valid.bool() & (kmer_ids >= 0)
    if not filter_mask_cpu.any():
        return 0

    device = torch.device(device)
    dna_input = dna_input.to(device, non_blocking=True)
    ts_input = ts_input.to(device, non_blocking=True)
    percentiles_batch = percentiles_batch.to(device, non_blocking=True)
    pad_base_seq = pad_base_seq.to(device, non_blocking=True)
    pad_ts = pad_ts.to(device, non_blocking=True)
    filter_mask = filter_mask_cpu.to(device, non_blocking=True)

    with torch.inference_mode(), autocast_context(device):
        out = model(
            dna_input,
            ts_input,
            percentiles_input=percentiles_batch,
            pad_base_seq=pad_base_seq,
            pad_ts=pad_ts,
        )

    embeddings = select_base_embedding(out, embedding_key)
    accumulator.update(kmer_ids[filter_mask_cpu], embeddings[filter_mask])
    return int(filter_mask_cpu.sum().item())


def process_dummy_batch(batch, accumulator, num_trim):
    """Accumulate one-hot base vectors for a model-free pipeline dry run."""
    dna_input, _ts_input, _percentiles_batch, pad_base_seq, _pad_ts, _read_name, _positions, _strand, _min_start, kmer_ids, kmer_valid = batch
    filter_mask = ignore_edges(pad_base_seq.bool(), num_trim) & kmer_valid.bool() & (kmer_ids >= 0)
    if not filter_mask.any():
        return 0
    bases = torch.clamp(dna_input - 1, min=0, max=3)
    dummy_embeddings = F.one_hot(bases, num_classes=4).float()
    accumulator.update(kmer_ids[filter_mask], dummy_embeddings[filter_mask])
    return int(filter_mask.sum().item())


def load_model(model_path, device, compile_model=True):
    """Restore a frozen backbone, compiling it unless explicitly disabled."""
    model = load_checked_model(
        model_path,
        model_type="backbone",
        add_signal_cross_attn=False,
        device=device,
    )
    return torch.compile(model, dynamic=True) if compile_model else model


def validate_args(args):
    """Validate extraction inputs or the alternative reservoir-union mode."""
    if args.read_based_motif and not args.motif:
        raise ValueError("--read_based_motif requires --motif")
    if args.reservoir_size < 0:
        raise ValueError(f"--reservoir_size must be non-negative, got {args.reservoir_size}")
    union_files = getattr(args, 'union_prototype_files', None)
    if union_files:
        if not getattr(args, 'output', ''):
            raise ValueError("--output is required with --union_prototype_files")
        return
    bam_path = getattr(args, 'bam_path', '')
    bam_config = getattr(args, 'bam_config', '')
    if bool(bam_path) == bool(bam_config):
        raise ValueError("Extraction mode requires exactly one of --bam_path or --bam_config")


def run(args):
    """Extract prototype statistics from BAMs or dispatch reservoir unioning.

    Extraction streams exact-match centers through the selected pretrained
    representation, accumulates k-mer means and optional reservoirs, and saves
    a self-describing PyTorch payload at ``args.output``.
    """
    validate_args(args)
    if args.union_prototype_files:
        return run_union(args)
    resolve_cli_preprocessing(args)
    if args.output:
        prepare_run_outputs(args, command="prototype extract-kmer-prototypes", files={"output": args.output})
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.step = resolve_downstream_step(
        args.size,
        args.seq_type,
        args.step,
        context="k-mer prototype extraction",
    )

    kmer_table = resolve_cli_preprocessing(args).kmer_table

    bam_config_list, source_metadata = resolve_bam_samples(args)
    bam_paths = [source['bam_path'] for source in source_metadata]
    resolved_chroms = list(dict.fromkeys(
        chrom
        for source in source_metadata
        for chrom in source['chroms']
    ))
    dataset = KmerPrototypeAlignedBamIterableDataset(
        bam_list=bam_config_list,
        size=args.size,
        step=args.step,
        seq_type=args.seq_type,
        chunking_mode=resolve_chunking_mode(args), kmer_table=kmer_table,
        reads_per_chunk=args.reads_per_chunk,
        min_read_length=args.min_read_length,
        min_qscore=args.min_qscore,
        pore_type=args.pore_type,
        base_seed=0,
        rank=0,
        world_size=1,
        expected_ts_dim=args.ts_dim,
        kmer_size=args.kmer_size,
        min_base_quality=args.min_base_quality,
        read_based_motif=args.read_based_motif,
    )
    loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=kmer_prototype_collate_fn,
        num_workers=args.num_workers,
        pin_memory=(torch.device(args.device).type == 'cuda' and not args.dry_run_no_model),
        drop_last=False,
    )
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    checkpoint_sha256 = None
    if args.dry_run_no_model:
        model = None
        accumulator = KmerPrototypeAccumulator(
            dim=4,
            k=args.kmer_size,
            device='cpu',
            reservoir_size=args.reservoir_size,
            reservoir_seed=args.reservoir_seed,
        )
    else:
        if not args.model_path:
            raise ValueError("--model_path is required unless --dry_run_no_model is set")
        from .effects import checkpoint_digest
        checkpoint_sha256 = checkpoint_digest(args.model_path)
        model = load_model(args.model_path, args.device, compile_model=not args.disable_compile)
        accumulator = KmerPrototypeAccumulator(
            dim=get_embedding_dimension(model.config, args.embedding_key),
            k=args.kmer_size,
            device=args.prototype_device,
            reservoir_size=args.reservoir_size,
            reservoir_seed=args.reservoir_seed,
        )

    total_updates = 0
    start = time.time()
    batches_seen = 0
    loader_iter = iter(loader)
    try:
        while True:
            try:
                batch = next(loader_iter)
            except StopIteration:
                break

            batches_seen += 1
            if args.dry_run_no_model:
                total_updates += process_dummy_batch(batch, accumulator, args.num_trim)
            else:
                total_updates += process_model_batch(batch, model, accumulator, args.device, args.embedding_key, args.num_trim)

            if batches_seen % args.log_every == 0:
                covered_kmers = int((accumulator.count >= args.min_count).sum().item())
                print(
                    f"[{time.time() - start:.2f}s] batches={batches_seen} "
                    f"accepted_kmers={total_updates} "
                    f"covered_kmers={covered_kmers}/{accumulator.num_kmers} at min_count={args.min_count}",
                    flush=True,
                )
            if args.max_batches is not None and batches_seen >= args.max_batches:
                break
    finally:
        shutdown_dataloader_iterator(loader_iter)

    prototypes = accumulator.finalize(min_count=args.min_count)
    payload = {
        **prototypes,
        'metadata': {
            'preprocessing': args.preprocessing,
            'bam_path': args.bam_path,
            'bam_config': args.bam_config or None,
            'bam_paths': bam_paths,
            'bam_sources': source_metadata,
            'chroms': resolved_chroms,
            'ref': args.ref or None,
            'pos_file': args.pos_file,
            'motif': args.motif,
            'read_based_motif': args.read_based_motif,
            'label_center_mode': (
                'read_motif'
                if args.read_based_motif
                else 'reference_positions'
                if args.pos_file
                else 'reference_motif'
                if args.motif
                else 'all_exact_match_centers'
            ),
            'kmer_size': args.kmer_size,
            'min_base_quality': args.min_base_quality,
            'num_trim': args.num_trim,
            'min_count': args.min_count,
            'reservoir_size': args.reservoir_size,
            'reservoir_seed': args.reservoir_seed,
            'embedding_key': args.embedding_key,
            'checkpoint_sha256': checkpoint_sha256,
            'model_path': args.model_path,
            'embedding_dim': accumulator.dim,
            'dry_run_no_model': args.dry_run_no_model,
            'total_updates': total_updates,
            'batches_seen': batches_seen,
        },
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        torch.save(payload, args.output)

    print(
        {
            'batches_seen': batches_seen,
            'total_updates': total_updates,
            'observed_kmers': int((accumulator.count > 0).sum().item()),
            'covered_kmers': int((accumulator.count >= args.min_count).sum().item()),
            'total_possible_kmers': accumulator.num_kmers,
            'min_count': args.min_count,
            'saved_kmers': int(len(payload['kmer_ids'])),
            'output': args.output,
        },
        flush=True,
    )
    return payload


if __name__ == '__main__':
    require_single_process_environment(
        "extract_kmer_embedding_prototypes.py",
        "K-mer prototype extraction/union accumulates global embedding data in one process.",
    )
    print('\n%s:\nCommand: python %s\n' % (str(datetime.datetime.now()), ' '.join(sys.argv)), flush=True)
    run(parse_args())
