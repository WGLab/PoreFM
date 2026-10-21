"""Pairwise midpoint-owned control/test embedding comparisons.

This module deliberately keeps all derived comparison behavior outside the
pretrained backbone.  BAM reads are chunked through the existing complete-read
dataset, each read position is assigned to its most central eligible chunk,
and the resulting canonical observations feed lightweight site summaries,
per-read control-mean scores, or full control-null anomaly scoring.
"""

from __future__ import annotations

from porefm.cli.preprocessing_args import add_preprocessing_arguments, resolve_cli_preprocessing
from porefm.cli.chunking_args import add_chunking_arguments, resolve_chunking_mode
import argparse
import sys

from porefm.cli._parser import PoreFMArgumentParser
from porefm.cli.dual_stream_args import add_exclude_bed_argument
from porefm.common.chunking import resolve_downstream_step
from porefm.workflows.prototypes.embedding_selection import EMBEDDING_KEY_CHOICES


ANALYSIS_FLAGS = (
    "sitewise_embedding_shift",
    "sitewise_anomaly_scoring",
    "readwise_anomaly_scoring",
)
BLOCK_SCHEMA_VERSION = 2
OWNERSHIP_POLICY = "central_eligible_chunk_v1"


def _add_analysis_arguments(parser):
    parser.add_argument(
        "--sitewise-embedding-shift",
        action="store_true",
        help="Write per-site control/test mean-embedding MSE and cosine distance.",
    )
    parser.add_argument(
        "--sitewise-anomaly-scoring",
        action="store_true",
        help=(
            "Fit a control null independently at each site and write per-test-read "
            "MSE, cosine, retained PC coordinates, squared Mahalanobis distance, "
            "and p-value, plus compact per-site control-null statistics."
        ),
    )
    parser.add_argument(
        "--readwise-anomaly-scoring",
        action="store_true",
        help=(
            "Compare each test-read embedding with the full control mean at its "
            "site, writing MSE and cosine distance without retaining control rows."
        ),
    )
    parser.add_argument(
        "--store-embeddings",
        action="store_true",
        help="Write versioned midpoint-owned control/test embedding blocks.",
    )
    parser.add_argument(
        "--min-control-coverage",
        type=int,
        default=15,
        help="Required control coverage for every comparison mode.",
    )
    parser.add_argument(
        "--min-test-coverage",
        type=int,
        default=15,
        help=(
            "Required test coverage for sitewise modes; readwise scoring can "
            "score a single test read."
        ),
    )
    parser.add_argument("--block-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max-fit-coverage", type=int, default=256)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--anomaly-batch-size", type=int, default=2000)
    parser.add_argument("--score-batch-size", type=int, default=None)
    parser.add_argument("--max-score-coverage", type=int, default=None)
    parser.add_argument(
        "--pvalue-method", choices=("chi2", "empirical"), default="chi2"
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument(
        "--anomaly-device",
        default=None,
        help="Dense anomaly-scoring device; defaults to the model device or auto.",
    )


def _add_output_arguments(parser):
    parser.add_argument("--output", required=True, help="Run output directory.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when the existing run manifest matches this request.",
    )
    parser.add_argument(
        "--annotated-bam-dir",
        default=None,
        help="Optional directory receiving one annotated BAM per test input.",
    )
    parser.add_argument(
        "--bam-score",
        choices=("anomaly-phred", "control-mean-cosine"),
        default=None,
        help=(
            "Score copied to BAM. anomaly-phred scales Phred values using "
            "--bam-phred-max and requires sitewise anomaly scoring; "
            "control-mean-cosine requires either per-read scoring mode."
        ),
    )
    parser.add_argument(
        "--bam-base",
        default=None,
        help=(
            "Fundamental base for MM output. The supplied base is uppercased "
            "but is not rewritten based on --seq-type."
        ),
    )
    parser.add_argument("--bam-mod-code", default=None)
    parser.add_argument("--bam-score-tag", default="ES")
    parser.add_argument("--bam-coverage-tag", default="CV")
    parser.add_argument(
        "--bam-phred-max",
        type=float,
        default=40.0,
        help=(
            "Anomaly Phred value mapped to BAM byte 255; values at or above "
            "this limit saturate. Set 255 to reproduce the former raw-Phred bytes."
        ),
    )
    parser.add_argument("--bam-cosine-scale", type=float, default=2.0)
    parser.add_argument("--bam-threads", type=int, default=4)
    parser.add_argument(
        "--drop-tags", nargs="+", default=(),
        help="Auxiliary tags to remove from annotated BAM output after scoring, for example SG mv.",
    )


def _add_bam_processing_arguments(parser):
    parser.add_argument("--seq-type", choices=("dna", "rna"), required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--kmer-dict-path", required=True)
    parser.add_argument("--chrom-list", nargs="+", default=None)
    add_exclude_bed_argument(parser)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--motif-string",
        default="",
        help="Reference motif specification such as 'CG 0;CH 0'.",
    )
    selector.add_argument("--pos-file", default="")

    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--reads-per-chunk", type=int, default=20)
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Loader processes per population, capped at its BAM count; 0 reads in the main process.",
    )
    parser.add_argument(
        "--control-num-workers", type=int, default=None,
        help="Override --num-workers for the control population.",
    )
    parser.add_argument(
        "--test-num-workers", type=int, default=None,
        help="Override --num-workers for the test population.",
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=2,
        help="Complete-read batches prefetched per loader process.",
    )
    parser.add_argument("--forward-batch-size", type=int, default=256)
    parser.add_argument("--min-read-length", type=int, default=100)
    parser.add_argument("--max-read-length", type=int, default=500_000)
    parser.add_argument("--min-qscore", type=float, default=10)
    parser.add_argument("--num-trim", type=int, default=16)
    parser.add_argument("--pore-type", default="r10")
    parser.add_argument("--device", default="auto")
    compile_options = parser.add_mutually_exclusive_group()
    compile_options.add_argument(
        "--compile-model", dest="compile_model", action="store_true", default=True,
        help="Compile the model with torch.compile(dynamic=True); enabled by default.",
    )
    compile_options.add_argument(
        "--disable-compile", dest="compile_model", action="store_false",
        default=argparse.SUPPRESS,
        help="Disable model compilation for diagnostics.",
    )

    parser.add_argument("--test-use-polished-sequence", action="store_true")
    parser.add_argument("--control-use-polished-sequence", action="store_true")
    parser.add_argument("--polish-indel-limit", type=int, default=5)
    parser.add_argument("--polish-vcf-path", default=None)


def build_parser():
    """Build the BAM-only ``pairwise-comparison`` command parser."""
    parser = PoreFMArgumentParser(
        description="Compare control and test BAM samples using midpoint-owned embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "BAM support: sitewise-embedding-shift has no per-read score and cannot "
            "annotate BAM. sitewise-anomaly-scoring supports anomaly-phred or "
            "control-mean-cosine. readwise-anomaly-scoring supports only "
            "control-mean-cosine. MSE is retained in Parquet only."
        ),
    )
    from porefm.cli.model_args import add_model_arguments
    add_model_arguments(parser, local_help='Local pretrained backbone or classifier checkpoint.')
    parser.add_argument(
        "--embedding-key", choices=EMBEDDING_KEY_CHOICES, default="contextual_base_seq"
    )
    parser.add_argument("--control-bam", nargs="+", required=True)
    parser.add_argument("--test-bam", nargs="+", required=True)
    parser.add_argument(
        "--control-layout", choices=("ordered-shards", "independent-streams")
    )
    parser.add_argument(
        "--test-layout", choices=("ordered-shards", "independent-streams")
    )
    for population in ("control", "test"):
        parser.add_argument(
            f"--{population}-filter-class-index", type=int, nargs="+", default=None,
            metavar="INDEX",
            help=(
                f"Allowed zero-based classifier indices for {population} positions. "
                "Keep a position if any listed class meets its paired threshold."
            ),
        )
        parser.add_argument(
            f"--{population}-filter-min-probability", type=float, nargs="+", default=None,
            metavar="PROBABILITY",
            help=(
                f"Minimum probabilities for --{population}-filter-class-index, "
                "in the same order; supply one threshold per index."
            ),
        )
    add_chunking_arguments(parser)
    _add_bam_processing_arguments(parser)
    _add_analysis_arguments(parser)
    _add_output_arguments(parser)

    add_preprocessing_arguments(parser)
    return parser


if __name__ == "__main__" and any(
    arg in ("-h", "--help") for arg in sys.argv[1:]
):
    build_parser().parse_args()


import array
import collections
import dataclasses
import datetime
import heapq
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from natsort import natsorted
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

from porefm.common.tensor_masks import ignore_edges
from porefm.common.utils import require_single_process_execution
from porefm.data.bam import build_lookup_table, revcomp
from porefm.data.bam_headers import add_pg_to_header
from porefm.data.complete_reads import (
    CompleteLabelledReadIterableDataset,
    make_polished_read_dict,
)
from porefm.data.reader import LabelledBamSample, get_polished_read_for_sample
from porefm.data.excluded_regions import get_excluded_regions
from porefm.models.checkpointing import (
    build_repository_provenance,
)
from porefm.models.loading import load_model as load_checked_model
from porefm.workflows.anomaly.detection import (
    process_block_anomaly_detection_vectorized,
)
from porefm.workflows.prototypes.embedding_selection import (
    get_embedding_dimension,
    select_base_embedding,
)


@dataclasses.dataclass(frozen=True)
class ComparisonPlan:
    """Resolved analysis choices and their population-retention requirements."""

    sitewise_embedding_shift: bool = False
    sitewise_anomaly_scoring: bool = False
    readwise_anomaly_scoring: bool = False
    store_embeddings: bool = False

    @property
    def control_rows(self) -> bool:
        return self.sitewise_anomaly_scoring or self.store_embeddings

    @property
    def test_rows(self) -> bool:
        return (
            self.sitewise_anomaly_scoring
            or self.readwise_anomaly_scoring
            or self.store_embeddings
        )

    @property
    def control_moments(self) -> bool:
        # Full-precision control means are scientific inputs to every scoring
        # mode, independently of whether lower-precision rows are also kept.
        return (
            self.sitewise_embedding_shift
            or self.sitewise_anomaly_scoring
            or self.readwise_anomaly_scoring
        )

    @property
    def test_moments(self) -> bool:
        return self.sitewise_embedding_shift

    @property
    def writes_read_scores(self) -> bool:
        return self.sitewise_anomaly_scoring or self.readwise_anomaly_scoring


@dataclasses.dataclass
class ObservationRows:
    """Canonical per-read observations retained for one chromosome block."""

    positions: torch.Tensor
    read_positions: torch.Tensor
    strands: torch.Tensor
    readnames: np.ndarray
    embeddings: torch.Tensor

    def as_anomaly_dict(self):
        return {
            "positions": self.positions,
            "strands": self.strands,
            "readnames": self.readnames,
            "embeddings": self.embeddings,
        }


@dataclasses.dataclass
class SiteSummary:
    """Sorted strand/site counts and full-population mean embeddings."""

    positions: torch.Tensor
    strands: torch.Tensor
    counts: torch.Tensor
    means: torch.Tensor

    @property
    def keys(self):
        return torch.stack((self.strands.to(torch.long), self.positions), dim=1)


@dataclasses.dataclass
class CompiledPopulationBlock:
    """Rows and/or moments for one population and genomic block."""

    rows: ObservationRows | None
    summary: SiteSummary | None


class _ActivePopulationBlock:
    """Mutable row lists or dense moments retained until a block is safe."""

    def __init__(self, dim, block_size, keep_rows, keep_moments):
        self.dim = dim
        self.block_size = block_size
        self.keep_rows = keep_rows
        self.keep_moments = keep_moments
        self.positions = []
        self.read_positions = []
        self.strands = []
        self.readnames = []
        self.embeddings = []
        self.sum = (
            torch.zeros((2, block_size, dim), dtype=torch.float32)
            if keep_moments
            else None
        )
        self.count = (
            torch.zeros((2, block_size), dtype=torch.int32)
            if keep_moments
            else None
        )

    def add(self, block_start, positions, read_positions, strands, readnames, embeddings):
        local = positions - int(block_start)
        if self.keep_rows:
            self.positions.append(positions.clone())
            self.read_positions.append(read_positions.clone())
            self.strands.append(strands.clone())
            self.readnames.append(np.asarray(readnames, dtype=object).copy())
            # Keep scoring precision unchanged. Reduction to bfloat16 happens
            # only when an embedding block is serialized.
            self.embeddings.append(embeddings.clone())

        if self.keep_moments:
            ones = torch.ones_like(local, dtype=torch.int32)
            for strand in (0, 1):
                mask = strands == strand
                if mask.any():
                    self.sum[strand].index_add_(
                        0, local[mask], embeddings[mask].to(torch.float32)
                    )
                    self.count[strand].index_add_(0, local[mask], ones[mask])

    def compile(self, block_start):
        rows = None
        if self.keep_rows:
            if self.positions:
                rows = ObservationRows(
                    positions=torch.cat(self.positions),
                    read_positions=torch.cat(self.read_positions),
                    strands=torch.cat(self.strands),
                    readnames=np.concatenate(self.readnames),
                    embeddings=torch.cat(self.embeddings),
                )
            else:
                rows = _empty_rows(self.dim)

        if self.keep_moments:
            mask = self.count > 0
            strands, local = mask.nonzero(as_tuple=True)
            counts = self.count[strands, local]
            means = self.sum[strands, local] / counts.unsqueeze(1).to(torch.float32)
            summary = SiteSummary(
                positions=local.to(torch.long) + int(block_start),
                strands=strands.to(torch.int8),
                counts=counts,
                means=means,
            )
        elif rows is not None:
            summary = summarize_rows(rows)
        else:
            summary = None
        return CompiledPopulationBlock(rows=rows, summary=summary)


class PopulationAccumulator:
    """Accumulate canonical observations into independently releasable blocks."""

    def __init__(self, dim, block_size, keep_rows, keep_moments):
        self.dim = dim
        self.block_size = block_size
        self.keep_rows = keep_rows
        self.keep_moments = keep_moments
        self.blocks = {}
        self.closed_blocks = set()

    def update(self, positions, read_positions, strands, readnames, embeddings):
        if positions.numel() == 0:
            return
        block_ids = torch.div(positions, self.block_size, rounding_mode="floor")
        for block_id in torch.unique(block_ids).tolist():
            block_start = int(block_id) * self.block_size
            if block_start in self.closed_blocks:
                raise RuntimeError(
                    "Received an observation for already-released block "
                    f"{block_start}. The input stream or alignment frontier is "
                    "not monotonic; refusing to overwrite prior output."
                )
            mask = block_ids == block_id
            block = self.blocks.get(block_start)
            if block is None:
                block = _ActivePopulationBlock(
                    self.dim,
                    self.block_size,
                    self.keep_rows,
                    self.keep_moments,
                )
                self.blocks[block_start] = block
            mask_np = mask.numpy()
            block.add(
                block_start,
                positions[mask],
                read_positions[mask],
                strands[mask],
                np.asarray(readnames, dtype=object)[mask_np],
                embeddings[mask],
            )

    def pop(self, block_start):
        block = self.blocks.pop(block_start, None)
        # Release closes the coordinate for this population even when it had
        # no rows. A later first observation is just as unsafe as recreation.
        self.closed_blocks.add(block_start)
        if block is None:
            return None
        return block.compile(block_start)


def _batch_frontier(batch):
    """Return the source-alignment frontier carried by a complete-read batch."""
    value = batch[-3]
    return int(value.item()) if torch.is_tensor(value) else int(value)


class MonotonicCompleteReadIterator:
    """Reject a coordinate stream whose source-alignment frontier regresses.

    A regression means block release is unsafe. Raising before returning the
    regressing batch prevents it from recreating an already-closed partition.
    """

    def __init__(self, iterable: Iterable, label: str):
        self.iterator = iter(iterable)
        self.label = label
        self.previous_frontier = None

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.iterator)
        frontier = _batch_frontier(batch)
        if (
            self.previous_frontier is not None
            and frontier < self.previous_frontier
        ):
            raise ValueError(
                f"Non-monotonic alignment frontier in {self.label}: "
                f"{frontier} follows {self.previous_frontier}. Use "
                "--*-layout independent-streams for unrelated BAMs, or fix "
                "the coordinate-shard order."
            )
        self.previous_frontier = frontier
        return batch


class SortedCompleteReadMerger:
    """K-way merge complete-read batches from independent sorted BAM streams."""

    def __init__(self, iterables: Sequence[Iterable]):
        self.iterators = [iter(item) for item in iterables]
        self.heap = []
        for idx, iterator in enumerate(self.iterators):
            try:
                self._push(next(iterator), idx)
            except StopIteration:
                pass

    @staticmethod
    def _min_start(batch):
        return _batch_frontier(batch)

    def _push(self, batch, index):
        heapq.heappush(self.heap, (self._min_start(batch), index, batch))

    def __iter__(self):
        return self

    def __next__(self):
        if not self.heap:
            raise StopIteration
        _, index, batch = heapq.heappop(self.heap)
        try:
            self._push(next(self.iterators[index]), index)
        except StopIteration:
            pass
        return batch


class EmbeddingModelAdapter:
    """Normalize backbone and classifier checkpoint execution contracts."""

    def __init__(self, model, config, is_classifier, metadata=None):
        self.model = model
        self.config = config
        self.is_classifier = is_classifier
        self.metadata = dict(metadata or {})

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def _empty_rows(dim):
    return ObservationRows(
        positions=torch.empty(0, dtype=torch.long),
        read_positions=torch.empty(0, dtype=torch.long),
        strands=torch.empty(0, dtype=torch.int8),
        readnames=np.empty(0, dtype=object),
        embeddings=torch.empty((0, dim), dtype=torch.bfloat16),
    )


def summarize_rows(rows: ObservationRows) -> SiteSummary:
    """Return full-population counts and means for midpoint-owned rows."""
    if rows.positions.numel() == 0:
        return SiteSummary(
            positions=torch.empty(0, dtype=torch.long),
            strands=torch.empty(0, dtype=torch.int8),
            counts=torch.empty(0, dtype=torch.int32),
            means=torch.empty((0, rows.embeddings.shape[1]), dtype=torch.float32),
        )
    keys = torch.stack((rows.strands.to(torch.long), rows.positions), dim=1)
    unique, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse, minlength=len(unique)).to(torch.int32)
    sums = torch.zeros(
        (len(unique), rows.embeddings.shape[1]), dtype=torch.float32
    )
    sums.index_add_(0, inverse, rows.embeddings.to(torch.float32))
    return SiteSummary(
        positions=unique[:, 1].to(torch.long),
        strands=unique[:, 0].to(torch.int8),
        counts=counts,
        means=sums / counts.unsqueeze(1).to(torch.float32),
    )


def central_owner_mask(readnames, read_positions, candidate_mask, pad_base_seq):
    """Select one most-central eligible chunk for every read position.

    Candidate positions are grouped by read name and read-space coordinate.
    The candidate furthest from either real sequence edge wins; equal margins
    retain the earlier chunk.  The function assumes complete-read chunks are
    present together but does not require read groups to be contiguous.
    """
    if candidate_mask.shape != read_positions.shape:
        raise ValueError("candidate_mask and read_positions must have equal shape")
    owner = torch.zeros_like(candidate_mask, dtype=torch.bool)
    names = np.asarray(readnames, dtype=object)
    real_lengths = pad_base_seq.sum(dim=1).to(torch.long)
    best = {}
    selected = candidate_mask.nonzero(as_tuple=False)
    for row_tensor, col_tensor in selected:
        row = int(row_tensor)
        col = int(col_tensor)
        read_pos = int(read_positions[row, col])
        if read_pos < 0:
            continue
        margin = min(col, int(real_lengths[row]) - col - 1)
        key = (str(names[row]), read_pos)
        previous = best.get(key)
        if previous is None or margin > previous[0]:
            best[key] = (margin, row, col)
    for _, row, col in best.values():
        owner[row, col] = True
    return owner


def _expand_bam_inputs(values):
    paths = []
    seen = set()
    for value in values:
        path = Path(value).expanduser()
        if path.is_dir():
            candidates = natsorted(path.rglob("*.bam"), key=lambda item: str(item))
        elif path.is_file():
            candidates = [path]
        else:
            raise FileNotFoundError(f"BAM input does not exist: {path}")
        for candidate in candidates:
            resolved = str(candidate.resolve())
            if resolved not in seen:
                paths.append(resolved)
                seen.add(resolved)
    if not paths:
        raise FileNotFoundError(f"No BAM inputs found in {values}")
    # Shell globs are expanded lexically before they reach argparse, so sort
    # the combined list too. Numeric suffixes are ordinal for repository-
    # generated coordinate shards (features.2.bam precedes features.10.bam).
    return natsorted(paths)


def _validate_layout(paths, layout, name):
    if len(paths) > 1 and layout is None:
        raise ValueError(
            f"--{name}-layout is required when --{name}-bam expands to multiple files"
        )
    return layout or "ordered-shards"


def _make_samples(paths, chrom_info, args, control):
    polish = (
        args.control_use_polished_sequence
        if control
        else args.test_use_polished_sequence
    )
    return [
        LabelledBamSample(
            bam_path=path,
            chrom_info=chrom_info,
            ref=args.ref,
            num_classes=1,
            pos_file=args.pos_file or "",
            motif=args.motif_string or "",
            motif_label="0",
            flag=0,
            polish=polish,
            polish_indel_limit=args.polish_indel_limit,
            polish_vcf_path=args.polish_vcf_path if polish else None,
        )
        for path in paths
    ]


def _complete_dataset(samples, args, model_config, kmer_table):
    return CompleteLabelledReadIterableDataset(
        bam_list=samples,
        size=args.size,
        step=args.step,
        move_downsample_factor=1,
        seq_type=args.seq_type,
        chunking_mode=resolve_chunking_mode(args), kmer_table=kmer_table,
        reads_per_chunk=args.reads_per_chunk,
        min_read_length=args.min_read_length,
        max_read_length=args.max_read_length,
        min_qscore=args.min_qscore,
        pore_type=args.pore_type,
        base_seed=args.seed,
        rank=0,
        world_size=1,
        discard_empty_labels=bool(args.motif_string or args.pos_file),
        return_pysam=False,
        expected_ts_dim=model_config.ts_dim,
        exclude_bed=args.exclude_bed,
    )


def _worker_count(args, control, bam_count):
    """Resolve a population's process budget without creating empty assignments."""
    override = args.control_num_workers if control else args.test_num_workers
    requested = args.num_workers if override is None else override
    if requested < 0:
        raise ValueError("Loader worker counts must be >= 0")
    if bam_count < 1:
        raise ValueError("Each population must contain at least one BAM")
    return requested, min(requested, bam_count)


class _PopulationWorkerDataset(IterableDataset):
    """Read one explicitly assigned, nonempty group of BAMs on the CPU.

    This wrapper runs in exactly one DataLoader worker (or the main process).
    Consequently the complete-read dataset sees worker ID 0/count 1 and reads
    every assigned BAM exactly once. Ordered shards form one local stream;
    unrelated BAMs must be merged locally before participating in the outer
    worker merge. Complete-read tuples and overlap ownership are unchanged.
    """

    def __init__(self, samples, layout, args, model_config, kmer_table):
        super().__init__()
        if not samples:
            raise ValueError("A population worker must own at least one BAM")
        self.samples = samples
        self.layout = layout
        self.args = args
        self.model_config = model_config
        self.kmer_table = kmer_table

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        if info is not None and (info.id != 0 or info.num_workers != 1):
            raise RuntimeError("Population worker datasets require one worker per loader")
        groups = (
            [self.samples] if self.layout == "ordered-shards"
            else [[sample] for sample in self.samples]
        )
        streams = [
            MonotonicCompleteReadIterator(
                _complete_dataset(group, self.args, self.model_config, self.kmer_table),
                f"{self.layout} starting at {group[0].bam_path}",
            )
            for group in groups
        ]
        yield from SortedCompleteReadMerger(streams)


def _population_iterator(paths, layout, chrom_info, args, model_config, kmer_table, control):
    """Merge a bounded set of CPU loaders by source-alignment frontier.

    Each BAM belongs to exactly one strided assignment. A single-worker loader
    per assignment exposes an independently fetchable head: PyTorch's ordinary
    round-robin multiworker output cannot safely drive coordinate block release.
    Empty/filtered streams disappear on StopIteration, never through guessed
    worker IDs or synthetic frontiers. Spawn avoids inheriting CUDA state.
    """
    requested, effective = _worker_count(args, control, len(paths))
    population = "control" if control else "test"
    print(
        f"{population} loaders: requested={requested}, effective={effective}, "
        f"BAMs={len(paths)}, layout={layout}", flush=True,
    )
    samples = _make_samples(paths, chrom_info, args, control)
    loaders = []
    try:
        assignment_count = max(1, effective)
        for worker_index in range(assignment_count):
            dataset = _PopulationWorkerDataset(
                samples[worker_index::assignment_count], layout, args,
                model_config, kmer_table,
            )
            options = dict(batch_size=None, num_workers=int(effective > 0))
            if effective:
                options.update(
                    multiprocessing_context="spawn", prefetch_factor=args.prefetch_factor,
                )
            # Start all loaders before reading their heads so preprocessing can
            # proceed concurrently even while one assignment has no eligible reads.
            loaders.append(iter(DataLoader(dataset, **options)))
        streams = [
            MonotonicCompleteReadIterator(loader, f"{population} worker {index}")
            for index, loader in enumerate(loaders)
        ]
        yield from SortedCompleteReadMerger(streams)
    finally:
        if effective:
            # DataLoader exposes no public close method. Explicit shutdown also
            # covers model/scoring exceptions and callers closing early.
            for loader in loaders:
                loader._shutdown_workers()


def load_model_adapter(path, device, compile_model=True):
    """Load and by default compile a frozen model plus pretraining metadata.

    Classifier checkpoints use their embedded ``pretrain_meta`` for collection
    parameters such as chunk stride. The supervised ``meta`` mapping describes
    the head-training run and does not reliably contain those values.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    manifest = checkpoint.get("model_manifest") or {}
    is_classifier = bool(
        checkpoint.get("classifier_config")
        or (manifest.get("head_configs") or {}).get("classifier_head")
        or manifest.get("checkpoint_type", "").startswith("classification")
    )
    if is_classifier:
        model = load_checked_model(
            path,
            model_type="classifier",
            device=device,
        )
        config = model.pretrain.config
    else:
        model = load_checked_model(
            path,
            model_type="backbone",
            device=device,
        )
        config = model.config
    if compile_model:
        model = torch.compile(model, dynamic=True)
    metadata = checkpoint.get("pretrain_meta") or checkpoint.get("meta") or {}
    return EmbeddingModelAdapter(model, config, is_classifier, metadata)


def _normalize_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} requested but CUDA is unavailable")
    return device


def _population_class_filter(args, population):
    """Validate paired option lists and return (class index, threshold) pairs.

    A population without either option is unfiltered (``None``). Indices refer
    to zero-based classifier output columns; their upper bound is checked when
    logits are available. Thresholds are inclusive probabilities in [0, 1].
    """
    index_option = f"{population}_filter_class_index"
    probability_option = f"{population}_filter_min_probability"
    indices = getattr(args, index_option, None)
    thresholds = getattr(args, probability_option, None)
    flags = f"--{index_option.replace('_', '-')} and --{probability_option.replace('_', '-')}"
    if (indices is None) != (thresholds is None):
        raise ValueError(f"{flags} must be supplied together")
    if indices is None:
        return None
    if not indices or len(indices) != len(thresholds):
        raise ValueError(f"{flags} must have the same nonzero number of values")
    if any(index < 0 for index in indices):
        raise ValueError(f"{population.capitalize()} filter class indices must be >= 0")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{population.capitalize()} filter class indices must be unique")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in thresholds):
        raise ValueError(f"{population.capitalize()} filter probabilities must be finite and in [0, 1]")
    return tuple(zip(indices, thresholds))


def _class_filter_mask(logits, class_filter):
    """Return a [batch, bases] mask for any passing class/threshold pair.

    ``logits`` has shape [batch, bases, classes]. Softmax uses float32 over all
    classes, preserving the former single-class probability calculation. The
    allowed probabilities are neither summed nor renormalized, and a passing
    class need not be the argmax. Each base is retained at most once.
    """
    for class_index, _ in class_filter:
        if class_index < 0 or class_index >= logits.shape[-1]:
            raise ValueError(
                f"Filter class {class_index} is outside classifier output "
                f"dimension {logits.shape[-1]}"
            )
    probabilities = F.softmax(logits.float(), dim=-1)
    allowed = torch.zeros_like(probabilities[..., 0], dtype=torch.bool)
    for class_index, threshold in class_filter:
        allowed |= probabilities[..., class_index] >= threshold
    return allowed


def process_complete_batch(
    batch,
    model,
    accumulator,
    device,
    embedding_key,
    forward_batch_size,
    num_trim,
    select_labelled,
    class_filter=None,
):
    """Forward a complete-read batch and accumulate retained canonical positions.

    ``class_filter`` is None or a sequence of (zero-based class index, minimum
    probability) pairs. Filtering uses OR across pairs after canonical chunk
    ownership is fixed; a rejected owner never falls back to another chunk.
    Embeddings and classifier logits come from the same existing forward pass.
    """
    (
        dna,
        signal,
        percentiles,
        pad_base,
        pad_signal,
        labels_idx,
        readnames,
        _refnames,
        positions,
        read_positions,
        _min_start,
        strands,
        _pysam_data,
    ) = batch

    eligible = (positions >= 0) & ignore_edges(pad_base, num_trim)
    if select_labelled:
        eligible &= labels_idx.bool()
    owner = central_owner_mask(readnames, read_positions, eligible, pad_base)
    if not owner.any():
        return

    for start in range(0, dna.shape[0], forward_batch_size):
        end = min(start + forward_batch_size, dna.shape[0])
        owner_part = owner[start:end]
        if not owner_part.any():
            continue
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                dna[start:end].to(device, non_blocking=True),
                signal[start:end].to(device, non_blocking=True),
                percentiles_input=percentiles[start:end].to(device, non_blocking=True),
                pad_base_seq=pad_base[start:end].to(device, non_blocking=True),
                pad_ts=pad_signal[start:end].to(device, non_blocking=True),
            )
            embedding = select_base_embedding(output, embedding_key)

        selected = owner_part.to(device)
        if class_filter is not None:
            if "classifier_logits" not in output:
                raise ValueError("Class filtering requires a classifier checkpoint with classifier_logits")
            selected &= _class_filter_mask(output["classifier_logits"], class_filter)

        selected_cpu = selected.cpu()
        if not selected_cpu.any():
            continue
        rows, _ = selected_cpu.nonzero(as_tuple=True)
        global_rows = rows + start
        accumulator.update(
            positions[start:end][selected_cpu].to(torch.long),
            read_positions[start:end][selected_cpu].to(torch.long),
            strands[global_rows].to(torch.int8),
            np.asarray(readnames, dtype=object)[global_rows.numpy()],
            embedding[selected].detach().cpu(),
        )


def _summary_index(summary):
    return {
        (int(strand), int(position)): index
        for index, (strand, position) in enumerate(
            zip(summary.strands.tolist(), summary.positions.tolist())
        )
    }


def score_mean_shift(control, test, min_control, min_test):
    """Return aligned locus-level full-population mean distances."""
    c_index = _summary_index(control.summary)
    t_index = _summary_index(test.summary)
    common = sorted(set(c_index) & set(t_index), key=lambda item: (item[1], item[0]))
    out = collections.defaultdict(list)
    for key in common:
        ci, ti = c_index[key], t_index[key]
        c_count = int(control.summary.counts[ci])
        t_count = int(test.summary.counts[ti])
        if c_count < min_control or t_count < min_test:
            continue
        c_mean = control.summary.means[ci].float()
        t_mean = test.summary.means[ti].float()
        out["position"].append(key[1])
        out["strand"].append(key[0])
        out["control_coverage"].append(c_count)
        out["test_coverage"].append(t_count)
        out["mean_squared_distance"].append(
            float(torch.mean((t_mean - c_mean) ** 2))
        )
        out["cosine_distance"].append(
            float(torch.clamp(1.0 - F.cosine_similarity(t_mean, c_mean, dim=0), 0, 2))
        )
    return dict(out)


def _site_keys(rows):
    """Return CPU strand/position keys without packing genomic coordinates."""
    return np.rec.fromarrays(
        (rows.strands.numpy(), rows.positions.numpy()), names=("strand", "position")
    )


def _match_control_rows(summary, test_rows, min_control, row_indices):
    """Match original test-row indices to eligible full-population means.

    Both returned integer arrays preserve the requested row order. Missing or
    under-covered control sites are removed. Stable sorting and right lookup
    retain the former dictionary's last-entry rule if summary keys repeat.
    """
    if not len(summary.positions) or not len(row_indices):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    control_keys = _site_keys(summary)
    order = np.argsort(control_keys, kind="stable")
    sorted_keys = control_keys[order]
    test_keys = _site_keys(test_rows)[row_indices]
    matches = np.searchsorted(sorted_keys, test_keys, side="right") - 1
    present = matches >= 0
    matches = matches.clip(min=0)
    control_indices = order[matches]
    valid = (
        present
        & (sorted_keys[matches] == test_keys)
        & (summary.counts.numpy()[control_indices] >= min_control)
    )
    return row_indices[valid], control_indices[valid]


def _direct_score_values(summary, test_rows, row_indices, control_indices):
    """Evaluate the original float32 MSE/cosine formulas in bounded CPU batches.

    Each row still reduces over its embedding dimension, uses the stored
    uncapped control mean, and uses PyTorch's default cosine epsilon. Batching
    bounds temporary embedding copies independently of block coverage.
    """
    mse = np.empty(len(row_indices), dtype=np.float32)
    cosine = np.empty(len(row_indices), dtype=np.float32)
    for start in range(0, len(row_indices), 4096):
        end = start + 4096
        embeddings = test_rows.embeddings[row_indices[start:end]].float()
        means = summary.means[control_indices[start:end]].float()
        mse[start:end] = torch.mean((embeddings - means) ** 2, dim=1).numpy()
        cosine[start:end] = torch.clamp(
            1.0 - F.cosine_similarity(embeddings, means, dim=1), 0, 2
        ).numpy()
    return mse.tolist(), cosine.tolist()


def score_rows_to_control_mean(control, test_rows, min_control):
    """Score eligible canonical rows against full, uncapped control means.

    Return the existing list-valued score/metadata columns in input-row order;
    test coverage counts all input observations at the same position/strand.
    Empty or wholly filtered inputs return an empty dictionary.
    """
    row_indices, control_indices = _match_control_rows(
        control.summary, test_rows, min_control,
        np.arange(len(test_rows.positions), dtype=np.int64),
    )
    if not len(row_indices):
        return {}
    _, inverse, counts = np.unique(
        _site_keys(test_rows), return_inverse=True, return_counts=True,
    )
    mse, cosine = _direct_score_values(
        control.summary, test_rows, row_indices, control_indices,
    )
    return {
        "readname": [str(name) for name in test_rows.readnames[row_indices]],
        "position": test_rows.positions.numpy()[row_indices].tolist(),
        "read_position": test_rows.read_positions.numpy()[row_indices].tolist(),
        "strand": test_rows.strands.numpy()[row_indices].tolist(),
        "control_coverage": control.summary.counts.numpy()[control_indices].tolist(),
        "test_coverage": counts[inverse[row_indices]].tolist(),
        "mean_squared_distance": mse,
        "cosine_distance": cosine,
    }


def _last_identity_rows(rows, row_indices):
    """Preserve the existing last-row join for repeated read/site identities.

    The anomaly rows retain their true original indices. The former wrapper's
    read-name dictionaries selected the last matching input row for direct
    scores and read coordinates, even if a repeated identity had a different
    embedding. Resolve that same mapping once using integer group indices.
    """
    if not len(row_indices):
        return row_indices
    keys = np.rec.fromarrays(
        (np.asarray(rows.readnames, dtype=str), rows.positions.numpy(), rows.strands.numpy()),
        names=("readname", "position", "strand"),
    )
    _, reverse_first, inverse = np.unique(
        keys[::-1], return_index=True, return_inverse=True,
    )
    return len(keys) - 1 - reverse_first[inverse[len(keys) - 1 - row_indices]]


def score_sitewise_anomaly(control, test, args):
    """Return per-read scores and compact per-site PCA/null statistics.

    Direct MSE and cosine distances use the original embedding space. Principal
    component coordinates, squared Mahalanobis distances, and control null
    parameters use the site- and strand-specific retained PCA space.
    """
    anomaly = process_block_anomaly_detection_vectorized(
        control.rows.as_anomaly_dict(),
        test.rows.as_anomaly_dict(),
        min_control_cov=args.min_control_coverage,
        min_test_cov=args.min_test_coverage,
        max_fit_cov=args.max_fit_coverage,
        n_comp=args.n_components,
        batch_size=args.anomaly_batch_size,
        score_batch_size=args.score_batch_size,
        max_score_cov=args.max_score_coverage,
        pvalue_method=args.pvalue_method,
        calibration_fraction=args.calibration_fraction,
        device=args.anomaly_device,
        return_projection_details=True,
    )
    row_indices = _last_identity_rows(test.rows, anomaly["row_indices"])
    direct_indices, control_indices = _match_control_rows(
        control.summary, test.rows, args.min_control_coverage, row_indices,
    )
    if len(direct_indices) != len(row_indices):
        raise KeyError("An anomaly row has no eligible full-population control mean")
    result = {}
    if len(row_indices):
        mse, cosine = _direct_score_values(
            control.summary, test.rows, direct_indices, control_indices,
        )
        # Keep the previous column order and Python scalar/list types so the
        # compact Parquet writer receives the same schema and row ordering.
        result = {
            "readname": [str(name) for name in test.rows.readnames[row_indices]],
            "position": anomaly["positions"].tolist(),
            "strand": anomaly["strands"].tolist(),
            "mean_squared_distance": mse,
            "cosine_distance": cosine,
            "read_position": test.rows.read_positions.numpy()[row_indices].tolist(),
            "squared_mahalanobis_distance": anomaly["squared_mahalanobis"].tolist(),
            "p_value": anomaly["p_vals"].tolist(),
        }
        for component_index in range(args.n_components):
            result[f"pc_{component_index + 1}"] = anomaly[
                "principal_components"
            ][:, component_index].tolist()

    site_statistics = anomaly["site_statistics"]
    site_result = {
        "position": site_statistics["positions"],
        "strand": site_statistics["strands"],
        "control_coverage": site_statistics["control_coverage"],
        "test_coverage": site_statistics["test_coverage"],
        "control_fit_coverage": site_statistics["control_fit_coverage"],
        "test_fit_coverage": site_statistics["test_fit_coverage"],
        "scored_test_coverage": site_statistics["scored_test_coverage"],
        "calibration_coverage": site_statistics["calibration_coverage"],
        "calibration_pca_coverage": site_statistics[
            "calibration_pca_coverage"
        ],
        "control_pca_mean": site_statistics["control_pca_mean"],
        "control_pca_covariance": site_statistics[
            "control_pca_covariance"
        ],
    }
    return dict(result), site_result


def _metadata_table(table, metadata):
    encoded = {str(key).encode(): str(value).encode() for key, value in metadata.items()}
    return table.replace_schema_metadata(encoded)


def _atomic_write_parquet(table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="snappy")
    os.replace(temporary, path)


def write_locus_part(rows, output_dir, chrom, block_start):
    """Write one non-empty locus-metric block partition."""
    if not rows or not rows.get("position"):
        return None
    payload = {"chrom": [chrom] * len(rows["position"]), **rows}
    table = _metadata_table(
        pa.table(payload),
        {"schema_version": BLOCK_SCHEMA_VERSION, "ownership": OWNERSHIP_POLICY},
    )
    path = Path(output_dir) / "locus_metrics" / f"{chrom}_{block_start}.parquet"
    _atomic_write_parquet(table, path)
    return path


def _fixed_size_float_list(values, list_size):
    """Encode a two-dimensional float array as an Arrow fixed-size list."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != list_size:
        raise ValueError(
            f"Expected a two-dimensional array with width {list_size}, got "
            f"shape {values.shape}"
        )
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, list_size)


def write_anomaly_site_part(
    rows,
    output_dir,
    chrom,
    block_start,
    pvalue_method=None,
):
    """Write one compact sitewise PCA/null-statistics block partition.

    The covariance is flattened in row-major order into a fixed-size list. It
    includes the diagonal regularization used by the Mahalanobis scorer.
    """
    if not rows or len(rows.get("position", ())) == 0:
        return None
    control_pca_mean = np.asarray(rows["control_pca_mean"], dtype=np.float32)
    control_pca_covariance = np.asarray(
        rows["control_pca_covariance"],
        dtype=np.float32,
    )
    if control_pca_mean.ndim != 2:
        raise ValueError("control_pca_mean must have shape (n_sites, n_components)")
    n_sites, n_components = control_pca_mean.shape
    expected_covariance_shape = (n_sites, n_components, n_components)
    if control_pca_covariance.shape != expected_covariance_shape:
        raise ValueError(
            "control_pca_covariance must have shape "
            f"{expected_covariance_shape}, got {control_pca_covariance.shape}"
        )

    scalar_types = {
        "position": pa.int64(),
        "strand": pa.int8(),
        "control_coverage": pa.int32(),
        "test_coverage": pa.int32(),
        "control_fit_coverage": pa.int32(),
        "test_fit_coverage": pa.int32(),
        "scored_test_coverage": pa.int32(),
        "calibration_coverage": pa.int32(),
        "calibration_pca_coverage": pa.int32(),
    }
    columns = {
        "chrom": pa.array([chrom] * n_sites),
        **{
            name: pa.array(values, type=scalar_types.get(name))
            for name, values in rows.items()
            if name not in {"control_pca_mean", "control_pca_covariance"}
        },
        "control_pca_mean": _fixed_size_float_list(
            control_pca_mean,
            n_components,
        ),
        "control_pca_covariance": _fixed_size_float_list(
            control_pca_covariance.reshape(n_sites, -1),
            n_components * n_components,
        ),
    }
    metadata = {
        "schema_version": BLOCK_SCHEMA_VERSION,
        "ownership": OWNERSHIP_POLICY,
        "n_components": n_components,
        "pca_space": "per-site control-standardized joint PCA",
        "control_pca_covariance_layout": "row-major regularized covariance",
        "covariance_regularization": "1e-6 diagonal ridge",
    }
    if pvalue_method is not None:
        metadata["pvalue_method"] = pvalue_method
    table = _metadata_table(
        pa.table(columns),
        metadata,
    )
    path = (
        Path(output_dir)
        / "anomaly_site_metrics"
        / f"{chrom}_{block_start}.parquet"
    )
    _atomic_write_parquet(table, path)
    return path


def write_read_score_part(rows, output_dir, chrom, block_start):
    """Factor one read-score block into inference-style meta/data tables.

    The metadata table stores one row per read and a ``length`` used to split
    the flattened, read-position-ordered score rows in the data table. This is
    the same compact layout used by classifier inference.
    """
    if not rows or not rows.get("readname"):
        return None
    groups = collections.OrderedDict()
    for idx, (name, strand) in enumerate(zip(rows["readname"], rows["strand"])):
        groups.setdefault((str(name), int(strand)), []).append(idx)

    meta = {"readname": [], "refname": [], "strand": [], "length": []}
    data_fields = [key for key in rows if key not in {"readname", "strand"}]
    data = {key: [] for key in data_fields}
    for (name, strand), indices in groups.items():
        indices.sort(key=lambda idx: (rows["read_position"][idx], rows["position"][idx]))
        meta["readname"].append(name)
        meta["refname"].append(chrom)
        meta["strand"].append(strand)
        meta["length"].append(len(indices))
        for key in data_fields:
            data[key].extend(rows[key][idx] for idx in indices)

    if sum(meta["length"]) != len(next(iter(data.values()), [])):
        raise RuntimeError("Compact read-score meta lengths do not match data rows")
    metadata = {
        "schema_version": BLOCK_SCHEMA_VERSION,
        "ownership": OWNERSHIP_POLICY,
        "pairing": "data rows are contiguous in meta row order",
        "layout": "inference_meta_data_v1",
    }
    root = Path(output_dir) / "read_scores"
    stem = f"{chrom}_{block_start}"
    _atomic_write_parquet(_metadata_table(pa.table(meta), metadata), root / f"{stem}_meta.parquet")
    data_columns = {
        key: (
            pa.array(values, type=pa.float32())
            if key.startswith("pc_")
            else values
        )
        for key, values in data.items()
    }
    _atomic_write_parquet(
        _metadata_table(pa.table(data_columns), metadata),
        root / f"{stem}_data.parquet",
    )
    return root / f"{stem}_meta.parquet", root / f"{stem}_data.parquet"


def _summary_payload(summary):
    if summary is None:
        raise ValueError("Embedding blocks require a site summary for each population")
    return {
        "positions": summary.positions.detach().cpu(),
        "strands": summary.strands.detach().cpu(),
        "counts": summary.counts.detach().cpu(),
        "means": summary.means.detach().cpu().to(torch.float32),
    }


def _rows_payload(rows):
    """Build a CPU block payload with embeddings reduced only for storage."""
    return {
        "positions": rows.positions.detach().cpu(),
        "read_positions": rows.read_positions.detach().cpu(),
        "strands": rows.strands.detach().cpu(),
        "readnames": np.asarray(rows.readnames, dtype=object).copy(),
        "embeddings": rows.embeddings.detach().cpu().to(torch.bfloat16),
    }


def write_embedding_block(
    control,
    test,
    output_dir,
    chrom,
    block_start,
    embedding_key,
    collection_metadata=None,
):
    """Write one versioned midpoint-owned paired PT embedding block.

    Per-observation embeddings are serialized as bfloat16 for compactness.
    Float32 site means and integer counts are stored separately so mean-based
    scientific scores do not depend on whether block storage was requested.
    """
    root = Path(output_dir) / "embeddings"
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": BLOCK_SCHEMA_VERSION,
        "ownership_policy": OWNERSHIP_POLICY,
        "chrom": chrom,
        "block_start": int(block_start),
        "embedding_key": embedding_key,
        "collection_metadata": dict(collection_metadata or {}),
        "control": _rows_payload(control.rows),
        "test": _rows_payload(test.rows),
        "control_summary": _summary_payload(control.summary),
        "test_summary": _summary_payload(test.summary),
    }
    path = root / f"{chrom}_{block_start}.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def _validate_args(args):
    plan = ComparisonPlan(
        **{name: bool(getattr(args, name)) for name in ANALYSIS_FLAGS},
        store_embeddings=bool(args.store_embeddings),
    )
    if not any(dataclasses.asdict(plan).values()):
        raise ValueError("Select at least one scoring mode or --store-embeddings")
    if plan.sitewise_anomaly_scoring and plan.readwise_anomaly_scoring:
        raise ValueError(
            "--sitewise-anomaly-scoring already includes control-mean read scores; "
            "do not combine it with --readwise-anomaly-scoring"
        )
    if args.min_control_coverage < 1 or args.min_test_coverage < 1:
        raise ValueError("Minimum coverage values must be >= 1")
    if args.block_size < 1:
        raise ValueError("--block-size must be >= 1")
    if args.max_fit_coverage < 2:
        raise ValueError("--max-fit-coverage must be >= 2")
    if args.n_components < 1:
        raise ValueError("--n-components must be >= 1")
    if args.anomaly_batch_size < 1:
        raise ValueError("--anomaly-batch-size must be >= 1")
    if args.pvalue_method == "empirical" and not 0 < args.calibration_fraction < 1:
        raise ValueError(
            "--calibration-fraction must be strictly between 0 and 1"
        )
    for name in ("num_workers", "control_num_workers", "test_num_workers"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1")
    for population in ("control", "test"):
        _population_class_filter(args, population)

    if args.annotated_bam_dir:
        if not plan.writes_read_scores:
            raise ValueError(
                "BAM annotation requires --sitewise-anomaly-scoring or "
                "--readwise-anomaly-scoring"
            )
        if args.bam_score is None:
            raise ValueError("--bam-score is required with --annotated-bam-dir")
        if args.bam_score == "anomaly-phred" and not plan.sitewise_anomaly_scoring:
            raise ValueError(
                "--bam-score anomaly-phred requires --sitewise-anomaly-scoring"
            )
        if not args.bam_base or not args.bam_mod_code:
            raise ValueError(
                "--bam-base and --bam-mod-code are required for BAM annotation"
            )
        if not math.isfinite(args.bam_phred_max) or args.bam_phred_max <= 0:
            raise ValueError("--bam-phred-max must be finite and greater than zero")
        for tag_name in (args.bam_score_tag, args.bam_coverage_tag):
            if len(tag_name) != 2:
                raise ValueError("BAM auxiliary tag names must contain two characters")
        if args.bam_score_tag in {"MM", "ML", "MN", args.bam_coverage_tag}:
            raise ValueError(
                "--bam-score-tag must differ from MM, ML, MN, and the coverage tag"
            )
        if args.bam_coverage_tag in {"MM", "ML", "MN"}:
            raise ValueError("--bam-coverage-tag must differ from MM, ML, and MN")
    return plan


def _args_manifest(args, plan, extra=None):
    excluded = {"resume", "model_cache_dir", "offline"}
    if not getattr(args, "model", None):
        excluded.add("model")
    if not getattr(args, "exclude_bed", None):
        excluded.add("exclude_bed")
    arguments = {
        key: value
        for key, value in vars(args).items()
        if key not in excluded and isinstance(value, (str, int, float, bool, list, type(None)))
    }
    # Retain the collection-source field for existing manifests and resume checks.
    # BAM input is fixed; it is no longer a selectable CLI mode.
    arguments["input_mode"] = "from-bam"
    payload = {
        "schema_version": BLOCK_SCHEMA_VERSION,
        "ownership_policy": OWNERSHIP_POLICY,
        "arguments": arguments,
        "plan": dataclasses.asdict(plan),
        "repository_provenance": build_repository_provenance(),
    }
    if getattr(args, "model_source", None) is not None:
        payload["model_source"] = args.model_source
    if extra:
        payload.update(extra)
    return payload


def _manifest_comparison_payload(manifest):
    """Compare effective settings without rewriting older run manifests.

    Older control filters stored scalars, and test filtering did not exist.
    Normalize only these representations so equivalent requests can resume;
    changing any class or threshold still rejects the request.
    """
    payload = dict(manifest)
    payload.pop("repository_provenance", None)
    arguments = dict(payload["arguments"])
    for population in ("control", "test"):
        for suffix in ("class_index", "min_probability"):
            key = f"{population}_filter_{suffix}"
            value = arguments.get(key)
            if value is not None and not isinstance(value, list):
                value = [value]
            arguments[key] = value
    payload["arguments"] = arguments
    return payload


def _prepare_output(args, manifest):
    root = Path(args.output)
    manifest_path = root / "run_manifest.json"
    if root.exists() and any(root.iterdir()):
        if not args.resume:
            raise FileExistsError(
                f"Output directory {root} is non-empty; choose a new path or use --resume"
            )
        if not manifest_path.is_file():
            raise ValueError(f"Cannot resume {root}: run_manifest.json is missing")
        existing = json.loads(manifest_path.read_text())
        if _manifest_comparison_payload(existing) != _manifest_comparison_payload(manifest):
            raise ValueError("Cannot resume because the requested run manifest differs")
        return root
    root.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    return root


class _BlockOutputWriter:
    """Write completed CPU block results on one bounded background thread.

    ``max_pending`` includes the running task. Submission waits for the oldest
    task when that bound is reached, preserving block order and limiting tensor
    retention. Context exit joins all successful writes before BAM annotation;
    write failures are raised in the submitting thread. Scoring and random
    sampling never execute in this worker.
    """

    def __init__(self, max_pending=2):
        if max_pending < 1:
            raise ValueError("max_pending must be >= 1")
        self.max_pending = max_pending
        self.pending = collections.deque()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="comparison-output")

    def __enter__(self):
        return self

    def submit(self, function, *args, **kwargs):
        """Queue an immutable completed payload, surfacing prior write failures."""
        while self.pending and self.pending[0].done():
            self.pending.popleft().result()
        if len(self.pending) >= self.max_pending:
            self.pending.popleft().result()
        self.pending.append(self.executor.submit(function, *args, **kwargs))

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc is None:
                while self.pending:
                    self.pending.popleft().result()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
            # Preserve an existing scoring/loader exception, but report any
            # concurrent output failure as additional diagnostic context.
            if exc is not None:
                for future in self.pending:
                    if not future.cancelled() and future.exception() is not None:
                        logging.getLogger(__name__).error(
                            "Background comparison output also failed: %s", future.exception(),
                        )
            self.pending.clear()


def _write_block_outputs(
    locus_rows, read_rows, site_rows, embedding_blocks, output, chrom,
    block_start, embedding_key, collection_metadata, pvalue_method,
):
    """Serialize completed block results with the existing schemas and writers."""
    try:
        if locus_rows is not None:
            write_locus_part(locus_rows, output, chrom, block_start)
        if read_rows is not None:
            write_read_score_part(read_rows, output, chrom, block_start)
        if site_rows is not None:
            write_anomaly_site_part(
                site_rows, output, chrom, block_start, pvalue_method=pvalue_method,
            )
        if embedding_blocks is not None:
            write_embedding_block(
                *embedding_blocks, output, chrom, block_start, embedding_key,
                collection_metadata,
            )
    except Exception as error:
        raise RuntimeError(f"Failed writing comparison block {chrom}:{block_start}") from error


def _score_and_write_block(control, test, chrom, block_start, args, plan, output_writer=None):
    """Score in release order, then submit CPU-only output or write synchronously."""
    if control is None or test is None:
        return
    locus_rows = read_rows = site_rows = None
    if plan.sitewise_embedding_shift:
        locus_rows = score_mean_shift(
            control,
            test,
            args.min_control_coverage,
            args.min_test_coverage,
        )

    if plan.sitewise_anomaly_scoring:
        read_rows, site_rows = score_sitewise_anomaly(control, test, args)
    elif plan.readwise_anomaly_scoring:
        read_rows = score_rows_to_control_mean(
            control, test.rows, args.min_control_coverage
        )
    payload = (
        locus_rows, read_rows, site_rows,
        (control, test) if plan.store_embeddings else None,
        args.output, chrom, block_start, args.embedding_key,
        getattr(args, "collection_metadata", None), args.pvalue_method,
    )
    if output_writer is None:
        _write_block_outputs(*payload)
    else:
        output_writer.submit(_write_block_outputs, *payload)


def _release_safe_blocks(control_acc, test_acc, safe_frontier, chrom, args, plan, output_writer=None):
    all_starts = sorted(set(control_acc.blocks) | set(test_acc.blocks))
    for block_start in all_starts:
        if block_start + args.block_size > safe_frontier:
            continue
        control = control_acc.pop(block_start)
        test = test_acc.pop(block_start)
        _score_and_write_block(control, test, chrom, block_start, args, plan, output_writer)


def _run_chromosome(
    chrom,
    chrom_length,
    control_paths,
    test_paths,
    control_layout,
    test_layout,
    args,
    plan,
    model,
    device,
    kmer_table,
    output_writer=None,
):
    chrom_info = [(chrom, 0, chrom_length)]
    control_iter = _population_iterator(
        control_paths, control_layout, chrom_info, args, model.config, kmer_table, True
    )
    test_iter = _population_iterator(
        test_paths, test_layout, chrom_info, args, model.config, kmer_table, False
    )
    try:
        control_batch = next(control_iter, None)
        test_batch = next(test_iter, None)
        dim = get_embedding_dimension(model.config, args.embedding_key)
        control_acc = PopulationAccumulator(
            dim, args.block_size, plan.control_rows, plan.control_moments
        )
        test_acc = PopulationAccumulator(
            dim, args.block_size, plan.test_rows, plan.test_moments
        )
        control_filter = _population_class_filter(args, "control")
        test_filter = _population_class_filter(args, "test")
        select_labelled = bool(args.motif_string or args.pos_file)

        while control_batch is not None or test_batch is not None:
            control_key = _batch_frontier(control_batch) if control_batch is not None else math.inf
            test_key = _batch_frontier(test_batch) if test_batch is not None else math.inf
            if control_key <= test_key:
                process_complete_batch(
                    control_batch,
                    model,
                    control_acc,
                    device,
                    args.embedding_key,
                    args.forward_batch_size,
                    args.num_trim,
                    select_labelled,
                    control_filter,
                )
                control_batch = next(control_iter, None)
            else:
                process_complete_batch(
                    test_batch,
                    model,
                    test_acc,
                    device,
                    args.embedding_key,
                    args.forward_batch_size,
                    args.num_trim,
                    select_labelled,
                    test_filter,
                )
                test_batch = next(test_iter, None)

            next_control = _batch_frontier(control_batch) if control_batch is not None else math.inf
            next_test = _batch_frontier(test_batch) if test_batch is not None else math.inf
            _release_safe_blocks(
                control_acc,
                test_acc,
                min(next_control, next_test),
                chrom,
                args,
                plan,
                output_writer,
            )

        _release_safe_blocks(control_acc, test_acc, math.inf, chrom, args, plan, output_writer)
    finally:
        for iterator in (control_iter, test_iter):
            close = getattr(iterator, "close", None)
            if close is not None:
                close()


def run_from_bam(args, plan):
    """Collect midpoint-owned embeddings from BAMs and execute the plan."""
    resolve_cli_preprocessing(args)
    require_single_process_execution(
        "pairwise-comparison",
        "Embedding comparison coordinates two finite population streams.",
    )
    control_paths = _expand_bam_inputs(args.control_bam)
    test_paths = _expand_bam_inputs(args.test_bam)
    control_layout = _validate_layout(control_paths, args.control_layout, "control")
    test_layout = _validate_layout(test_paths, args.test_layout, "test")
    args.effective_control_num_workers = _worker_count(args, True, len(control_paths))[1]
    args.effective_test_num_workers = _worker_count(args, False, len(test_paths))[1]
    # Validate before model loading, and detect BED content changes on resume.
    if args.exclude_bed:
        args.exclude_bed_sha256 = get_excluded_regions(args.exclude_bed).sha256

    device = _normalize_device(args.device)
    model = load_model_adapter(args.model_path, device, args.compile_model)
    for population in ("control", "test"):
        if _population_class_filter(args, population) is not None and not model.is_classifier:
            raise ValueError(f"{population.capitalize()} class filtering requires a classifier checkpoint")
    args.size = args.size or int(model.config.max_ts_len)
    args.step = resolve_downstream_step(
        args.size,
        args.seq_type,
        args.step,
        context="pairwise-comparison",
    )
    args.anomaly_device = str(
        _normalize_device(args.anomaly_device)
        if args.anomaly_device is not None
        else device
    )
    args.collection_metadata = {
        "seq_type": args.seq_type,
        "reference": str(Path(args.ref).resolve()),
        "test_use_polished_sequence": bool(args.test_use_polished_sequence),
        "polish_indel_limit": int(args.polish_indel_limit),
        "polish_vcf_path": (
            str(Path(args.polish_vcf_path).resolve())
            if args.polish_vcf_path
            else None
        ),
    }
    if args.exclude_bed:
        args.collection_metadata.update(
            exclude_bed=str(Path(args.exclude_bed).resolve()),
            exclude_bed_sha256=args.exclude_bed_sha256,
        )
    kmer_table = resolve_cli_preprocessing(args).kmer_table

    with pysam.FastaFile(args.ref) as fasta:
        chroms = list(args.chrom_list or fasta.references)
        chrom_lengths = {chrom: fasta.get_reference_length(chrom) for chrom in chroms}

    manifest = _args_manifest(
        args,
        plan,
        {
            "control_bams": control_paths,
            "test_bams": test_paths,
            "control_layout": control_layout,
            "test_layout": test_layout,
            "embedding_dimension": get_embedding_dimension(model.config, args.embedding_key),
            "collection_metadata": args.collection_metadata,
        },
    )
    _prepare_output(args, manifest)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    start = time.time()
    with _BlockOutputWriter() as output_writer:
        for chrom in chroms:
            print(f"Processing {chrom}:0-{chrom_lengths[chrom]}", flush=True)
            _run_chromosome(
                chrom,
                chrom_lengths[chrom],
                control_paths,
                test_paths,
                control_layout,
                test_layout,
                args,
                plan,
                model,
                device,
                kmer_table,
                output_writer,
            )
    if args.annotated_bam_dir:
        annotate_bams_from_compact_scores(
            test_paths,
            Path(args.output) / "read_scores",
            args.annotated_bam_dir,
            args,
            drop_tags=args.drop_tags,
        )
    print(f"Completed pairwise comparison in {time.time() - start:.2f}s", flush=True)


def _load_anomaly_site_coverages(read_scores_dir):
    """Load coverage keyed by reference, strand, and position for BAM tags."""
    coverage_by_site = {}
    site_dir = Path(read_scores_dir).parent / "anomaly_site_metrics"
    for path in sorted(site_dir.glob("*.parquet")):
        table = pq.read_table(
            path,
            columns=[
                "chrom",
                "position",
                "strand",
                "control_coverage",
                "test_coverage",
            ],
        ).to_pydict()
        for chrom, position, strand, control_coverage, test_coverage in zip(
            table["chrom"],
            table["position"],
            table["strand"],
            table["control_coverage"],
            table["test_coverage"],
        ):
            key = (str(chrom), int(strand), int(position))
            value = (int(control_coverage), int(test_coverage))
            previous = coverage_by_site.setdefault(key, value)
            if previous != value:
                raise ValueError(f"Conflicting anomaly-site coverage for {key}")
    return coverage_by_site


def _anomaly_phred_byte(pvalue, phred_max):
    """Map an anomaly p-value onto the full BAM byte visualization range.

    ``phred_max`` defines the Phred value represented by 255. Larger Phred
    values saturate, while p-values greater than one map to zero. For the
    default maximum of 40, p=0.01 maps to byte 128.
    """
    pvalue = max(float(pvalue), np.finfo(np.float64).tiny)
    phred = -10.0 * math.log10(pvalue)
    return int(round(255.0 * np.clip(phred / phred_max, 0.0, 1.0)))


def _load_compact_calls(read_scores_dir, args):
    site_coverages = _load_anomaly_site_coverages(read_scores_dir)
    calls = collections.defaultdict(list)
    for meta_path in sorted(Path(read_scores_dir).glob("*_meta.parquet")):
        data_path = meta_path.with_name(meta_path.name.replace("_meta.parquet", "_data.parquet"))
        meta = pq.read_table(meta_path).to_pydict()
        data = pq.read_table(data_path).to_pydict()
        offset = 0
        for name, refname, strand, length in zip(
            meta["readname"], meta["refname"], meta["strand"], meta["length"]
        ):
            end = offset + int(length)
            for idx in range(offset, end):
                if args.bam_score == "anomaly-phred":
                    score = _anomaly_phred_byte(
                        data["p_value"][idx], args.bam_phred_max
                    )
                else:
                    distance = float(data["cosine_distance"][idx])
                    score = int(
                        round(255.0 * np.clip(args.bam_cosine_scale * distance, 0.0, 1.0))
                    )
                inline_coverage = {
                    "control_coverage",
                    "test_coverage",
                }.issubset(data)
                if inline_coverage:
                    control_coverage = data["control_coverage"][idx]
                    test_coverage = data["test_coverage"][idx]
                else:
                    site_key = (
                        str(refname),
                        int(strand),
                        int(data["position"][idx]),
                    )
                    try:
                        control_coverage, test_coverage = site_coverages[site_key]
                    except KeyError as exc:
                        raise ValueError(
                            f"Missing anomaly-site coverage for {site_key}"
                        ) from exc
                coverage = min(255, int(min(control_coverage, test_coverage)))
                calls[str(name)].append(
                    (
                        str(refname),
                        int(strand),
                        int(data["position"][idx]),
                        score,
                        coverage,
                    )
                )
            offset = end
        if offset != len(data.get("position", [])):
            raise ValueError(f"Meta/data row count mismatch for {meta_path}")
    return calls


def _reference_to_oriented_read_positions(read):
    read_length = read.query_length or 0
    mapping = {}
    for query_pos, ref_pos in read.get_aligned_pairs(matches_only=False):
        if query_pos is None or ref_pos is None:
            continue
        oriented = read_length - int(query_pos) - 1 if read.is_reverse else int(query_pos)
        mapping.setdefault(int(ref_pos), oriented)
    return mapping


def _base_match_settings(seq, base, seq_type):
    """Resolve sequence matching while preserving the requested MM base."""
    base = base.upper()
    if base not in {"A", "C", "G", "T", "U", "N"}:
        raise ValueError("--bam-base must be one of A,C,G,T,U,N")
    if base == "U":
        return ("U" if "U" in seq else "T"), "U"
    return base, base


def _score_mm_tag(seq, coords, score_bytes, coverage_bytes, base, mod_code, seq_type):
    """Encode aligned score bytes as MM/ML, retaining matched coverage bytes."""
    seq = seq.upper()
    coords = np.asarray(coords, dtype=np.int64)
    scores = np.asarray(score_bytes, dtype=np.uint8)
    coverage = np.asarray(coverage_bytes, dtype=np.uint8)
    valid_range = (coords >= 0) & (coords < len(seq))
    coords, scores, coverage = (
        coords[valid_range],
        scores[valid_range],
        coverage[valid_range],
    )
    match_base, mm_base = _base_match_settings(seq, base, seq_type)
    if match_base == "N":
        candidates = np.arange(len(seq), dtype=np.int64)
    else:
        candidates = np.flatnonzero(
            np.frombuffer(seq.encode("ascii"), dtype=np.uint8) == ord(match_base)
        )
    ranks = np.searchsorted(candidates, coords)
    valid = ranks < len(candidates)
    if valid.any():
        valid_indices = np.flatnonzero(valid)
        valid[valid_indices] &= candidates[ranks[valid_indices]] == coords[valid_indices]
    coords, scores, coverage, ranks = (
        coords[valid],
        scores[valid],
        coverage[valid],
        ranks[valid],
    )
    if not len(coords):
        return None, [], []
    order = np.argsort(coords, kind="stable")
    coords, scores, coverage, ranks = (
        coords[order],
        scores[order],
        coverage[order],
        ranks[order],
    )

    # One MM delta can represent only one call per read coordinate. Retain the
    # greatest score and its aligned coverage if malformed inputs duplicate it.
    keep = []
    for index, coord in enumerate(coords):
        if keep and coords[keep[-1]] == coord:
            if scores[index] > scores[keep[-1]]:
                keep[-1] = index
        else:
            keep.append(index)
    scores, coverage, ranks = scores[keep], coverage[keep], ranks[keep]
    deltas = np.diff(np.concatenate(([-1], ranks))) - 1
    return (
        f"{mm_base}+{mod_code}?," + ",".join(map(str, deltas.tolist())) + ";",
        scores.tolist(),
        coverage.tolist(),
    )


def _clear_score_tags(read, args):
    """Remove stale modification and comparison tags from an output record."""
    for tag in {"MM", "ML", "MN", args.bam_score_tag, args.bam_coverage_tag}:
        if read.has_tag(tag):
            read.set_tag(tag, None)


def _annotation_reads(source, chrom_list):
    """Yield records in the requested reference scope, or all when unspecified.

    Indexed inputs fetch selected references in BAM header order so reversed
    or repeated CLI names do not reorder or duplicate records. Unindexed inputs
    are filtered during a sequential scan. A chromosome restriction excludes
    unmapped records, including those carrying a mate's reference coordinate;
    selected mapped records are retained even when they have no score.
    References absent from an individual shard contribute no records.
    """
    if not chrom_list:
        yield from source.fetch(until_eof=True)
        return

    selected = set(chrom_list)
    if source.has_index():
        for chrom in source.references:
            if chrom in selected:
                for read in source.fetch(chrom):
                    if not read.is_unmapped:
                        yield read
    else:
        for read in source.fetch(until_eof=True):
            if not read.is_unmapped and read.reference_name in selected:
                yield read


def annotate_bams_from_compact_scores(test_bams, read_scores_dir, output_dir, args, drop_tags=()):
    """Attach compact scores to source records within ``args.chrom_list``.

    With no chromosome restriction, copy all source records. Otherwise, copy
    only mapped records on selected references, using indexed region access
    when available. Filtering precedes polishing and tag rewriting. Records
    without scores on selected references are still copied; existing BAM
    headers are preserved and augmented with run provenance.
    ``drop_tags`` optionally lists auxiliary tag names to remove immediately
    before writing, after any polishing and scoring (for example SG and mv).
    """
    calls = _load_compact_calls(read_scores_dir, args)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    basenames = [Path(path).name for path in test_bams]
    duplicates = sorted(
        name for name, count in collections.Counter(basenames).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            "Annotated BAM output names would collide: " + ", ".join(duplicates)
        )
    for bam_path in test_bams:
        output_path = output_dir / Path(bam_path).name
        if output_path.resolve() == Path(bam_path).resolve():
            raise ValueError(
                f"Annotated BAM output would overwrite its input: {output_path}"
            )
        with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as source:
            header = add_pg_to_header(source.header)
            polish_sample = None
            polish_variant_cache = {}
            if getattr(args, "test_use_polished_sequence", False):
                polish_sample = LabelledBamSample(
                    bam_path=bam_path,
                    chrom_info=(),
                    ref=args.ref,
                    num_classes=1,
                    polish=True,
                    polish_indel_limit=args.polish_indel_limit,
                    polish_vcf_path=args.polish_vcf_path,
                )
            with pysam.AlignmentFile(
                output_path, "wb", header=header, threads=args.bam_threads
            ) as destination:
                for read in _annotation_reads(source, getattr(args, "chrom_list", None)):
                    is_primary_mapped = not read.is_unmapped and not (read.flag & 0x900)
                    if polish_sample is not None and is_primary_mapped:
                        polished = get_polished_read_for_sample(
                            read,
                            polish_sample,
                            polish_variant_cache,
                        )
                        read_dict = make_polished_read_dict(
                            read,
                            polished,
                            {args.bam_score_tag, args.bam_coverage_tag},
                        )
                        read = pysam.AlignedSegment.from_dict(read_dict, header)
                    _clear_score_tags(read, args)
                    read_calls = calls.get(read.query_name, ())
                    matching = [
                        item
                        for item in read_calls
                        if item[0] == read.reference_name
                        and item[1] == (0 if read.is_forward else 1)
                    ]
                    if matching and is_primary_mapped:
                        ref_to_read = _reference_to_oriented_read_positions(read)
                        oriented = []
                        for _, _, refpos, score, coverage in matching:
                            read_pos = ref_to_read.get(refpos)
                            if read_pos is not None:
                                oriented.append((read_pos, score, coverage))
                        if oriented:
                            oriented.sort()
                            seq = read.query_sequence or ""
                            if read.is_reverse:
                                seq = revcomp(seq)
                            coords = [item[0] for item in oriented]
                            scores = [item[1] for item in oriented]
                            coverage = [item[2] for item in oriented]
                            mm, kept_scores, kept_coverage = _score_mm_tag(
                                seq,
                                coords,
                                scores,
                                coverage,
                                args.bam_base,
                                args.bam_mod_code,
                                args.seq_type,
                            )
                            if kept_scores:
                                read.set_tag("MM", mm, value_type="Z")
                                read.set_tag("ML", array.array("B", kept_scores))
                                read.set_tag(args.bam_score_tag, array.array("B", kept_scores))
                                read.set_tag(
                                    args.bam_coverage_tag,
                                    array.array("B", kept_coverage),
                                )
                                read.set_tag("MN", len(seq))
                    for tag in drop_tags:
                        if read.has_tag(tag):
                            read.set_tag(tag, None)
                    destination.write(read)
        try:
            pysam.index(str(output_path))
        except pysam.utils.SamtoolsError:
            # Preserve valid annotated output even when an input was not in an
            # indexable coordinate order.
            pass


def run(args):
    """Validate arguments and compare the control and test BAM populations."""
    plan = _validate_args(args)
    return run_from_bam(args, plan)


def parse_args(argv=None):
    """Resolve a published backbone/classifier before pairwise workflow startup."""
    from porefm.cli.model_args import resolve_model_arguments
    parser = build_parser()
    args = parser.parse_args(argv)
    resolve_model_arguments(args, roles=('backbone', 'supervised_classifier'), parser=parser)
    return args


def main():
    """CLI entry point for pairwise BAM comparisons."""
    print(
        f"\n{datetime.datetime.now()}:\nCommand: python {' '.join(sys.argv)}\n",
        flush=True,
    )
    return run(parse_args())


if __name__ == "__main__":
    main()
