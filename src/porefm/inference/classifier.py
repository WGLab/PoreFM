"""Classifier inference, read-wise stitching, and Parquet/BAM output."""

from porefm.preprocessing import preprocessing_from_checkpoint
from porefm.cli.preprocessing_args import add_preprocessing_arguments, resolve_cli_preprocessing
from porefm.cli.chunking_args import add_chunking_arguments, resolve_chunking_mode
import argparse
import sys

from porefm.cli._parser import PoreFMArgumentParser
from porefm.cli.model_args import add_model_arguments, resolve_model_arguments
from porefm.cli.classifier_input_args import add_classifier_input_arguments, resolve_classifier_input_config
from porefm.cli.dual_stream_args import add_exclude_bed_argument
from porefm.common.modification_metadata import resolve_modification_metadata
from porefm.common.run_outputs import add_output_safety_argument, prepare_run_outputs


def build_arg_parser():
    """Build the classifier-inference command-line parser."""
    parser = PoreFMArgumentParser(description="Detect DNA and RNA modifications with a fine-tuned model.")

    add_classifier_input_arguments(parser)
    parser.add_argument('--size', type=int, help='Size of the signal chunk window.')
    parser.add_argument('--step', type=int, help='Sliding step size for the signal chunk window.')
    add_model_arguments(parser, dest='model_load_path', required=False,
                        local_flags=('--model_load_path', '--model-path', '--model_path'),
                        local_help='Path to the trained modification-detection checkpoint.')
    parser.add_argument('--pretrain_load_path', type=str, help='Path to pretrained model.')
    add_chunking_arguments(parser)
    parser.add_argument('--kmer_dict_path', type=str, help='Path to kmer expected signal file.')

    parser.add_argument('--output', type=str, help='Directory to save inference results.')
    parser.add_argument('--output_format', type=str, default='parquet', choices=['parquet', 'bam', 'both'], help='Output format.')

    # --- QC Arguments ---
    parser.add_argument('--min_read_length', type=int, default=100, help='Minimum read length.')
    parser.add_argument('--min_qscore', type=int, default=10, help='Minimum read qscore.')
    add_exclude_bed_argument(parser)

    parser.add_argument('--batch_size', type=int, default=32, help='Number of signal chunks scored per model batch.')
    parser.add_argument('--disable_compile', help='Disable model compile.',default=False, action='store_true')

    # --- Data Processing Arguments ---
    parser.add_argument('--checkpoint_batch_step', type=int, default=100, help='Number of read batches between writing prediction outputs.')

    parser.add_argument('--bam_threads', type=int, default=4, help='Number of threads to use for BAM output writing.')
    parser.add_argument('--base', type=str, help='Override checkpoint canonical base; for legacy checkpoints defaults to the first class label.')
    parser.add_argument('--mod_code', type=str, nargs='+', help='Override checkpoint modification codes in class 1 onward logit order. Required for legacy checkpoints when writing BAM output.')

    parser.add_argument('--seed', type=int, default=0, help='Seed to use for data batching.')
    parser.add_argument('--reads_per_chunk', type=int, default=50, help='Number of reads to bundle into a single data chunk.')
    parser.add_argument('--dataloader_workers', type=int, default=0, help='DataLoader worker processes for prefetching/pinning.')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Batches prefetched per worker (only if workers > 0).')

    # --- Model Architecture Arguments ---
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--class_labels', nargs='+', type=str, default=[],
                            help='A list of modification class names in the same order as labels')
    model_group.add_argument('--num_classes', type=int, help='Number of classes')
    model_group.add_argument('--temperature', type=float, default=1, help='Logit temperature scaling')

    model_group.add_argument('--num_trim', type=int, default=10, help='Number of bases to trim from either end for loss.')

    add_output_safety_argument(parser)
    add_preprocessing_arguments(parser)
    return parser


def parse_args(argv=None):
    """Resolve versioned classifier releases before preprocessing or inference."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        resolve_classifier_input_config(args, class_labels=args.class_labels or None, required=True)
    except ValueError as exc:
        parser.error(str(exc))
    resolve_model_arguments(args, roles=('supervised_classifier',),
                            dest='model_load_path', parser=parser)
    return args


if __name__ == '__main__' and any(arg in ('-h', '--help') for arg in sys.argv[1:]):
    build_arg_parser().parse_args()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, IterableDataset
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import random, time, os, datetime, csv
from collections import defaultdict
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

import pysam, array
import queue
import threading
from pathlib import Path
from natsort import natsorted
from typing import List, Optional

from typing import Optional, Dict, Any, Union
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from dataclasses import dataclass, asdict

from contextlib import nullcontext

import torch.distributed as dist

from porefm.common.chunking import resolve_downstream_step
from porefm.common.utils import align_to_power_of_two, rewarmup_then_constant, get_dummy_loss
from porefm.common.tensor_masks import ignore_edges
from porefm.config.validation import load_supervised_config
from porefm.data.bam_headers import add_pg_to_header

torch.multiprocessing.set_start_method('spawn', force=True)
torch.set_num_threads(1)

from porefm.data.bam import build_lookup_table, revcomp
from porefm.data.reader import LabelledBamSample, TaggedBamSample
from porefm.data.excluded_regions import get_excluded_regions
from porefm.data.pod5 import find_pod5_files, validate_pod5_samples, pod5_worker_count
from porefm.data.bam_index import BamReadIndex
from porefm.models.backbone.model import ONTBackboneConfig
from porefm.models.backbone.utils import strip_module_prefix
from porefm.models.classifier import ONTModelForClassification
from porefm.models.checkpointing import get_pretraining_state_dict_from_checkpoint
from porefm.models.configuration import get_backbone_config_from_checkpoint
from porefm.models.loading import load_model as load_checked_model
import porefm.training.pretrain as pretrain_train
from porefm.data.complete_reads import CompleteLabelledReadIterableDataset
from porefm.inference.predictions import (
    collect_checkpoint_predictions,
    make_mm_ml_tags as encode_mm_ml_tags,
)

from concurrent.futures import ThreadPoolExecutor

torch.set_float32_matmul_precision('high')
torch._dynamo.config.capture_scalar_outputs = True

def get_global_counts(local_instances, local_base_tokens, local_signal_tokens, device):
    """Sum instance and token counters across initialized distributed ranks."""
    if not dist.is_initialized():
        return local_instances, local_base_tokens, local_signal_tokens
        
    # Pack into a tensor for efficient transfer
    tensor = torch.tensor([local_instances, local_base_tokens, local_signal_tokens], device=device, dtype=torch.long)
    
    # Sums the values from all ranks in-place
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    return tensor[0].item(), tensor[1].item(), tensor[2].item()

def setup_distributed(store=None):
    """Initialize NCCL, optionally reusing the completed POD5 startup rendezvous."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        if store is None:
            dist.init_process_group(backend="nccl")
        else:
            dist.init_process_group(backend="nccl", store=store, rank=rank, world_size=world_size)
        return rank, world_size, local_rank

    return 0, 1, 0


class NoOpWriter:
    """TensorBoard-compatible sink used on nonzero distributed ranks."""
    def add_scalar(self, *args, **kwargs):
        """Discard a scalar log event."""
        return None

    def flush(self):
        """Provide the writer flush interface without side effects."""
        return None

def reduce_loss_stats(total_loss: float, count: int, device: torch.device, world_size: int):
    """Sum loss and observation accumulators across distributed ranks."""
    if world_size == 1:
        return total_loss, count

    tensor = torch.tensor([total_loss, count], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor[0].item(), int(tensor[1].item())


def get_unwrapped_model(model):
    """Remove DDP and ``torch.compile`` wrappers when present."""
    model = model.module if isinstance(model, DDP) else model
    return getattr(model, "_orig_mod", model)


def get_pretrain_module(model):
    """Return the pretrained backbone from a classifier model or wrapper."""
    return get_unwrapped_model(model).pretrain
   
def _get_chrom_info(contigs: List[str], ref_path: str) -> List[tuple]:
    """Return full zero-based half-open FASTA spans for selected contigs."""
    fref = pysam.FastaFile(ref_path)
    
    # If the contigs list is empty, use all references in the FASTA
    if not contigs:
        return [(c, 0, fref.get_reference_length(c)) for c in fref.references]
    
    # Otherwise, use only the specified contigs
    return [(c, 0, fref.get_reference_length(c)) for c in contigs]

def parse_bam_config(
    json_config_path: str,
    num_classes: int,
    read_based: bool = False,
    class_labels=None,
    pod5_bam_index=None,
):
    """Expand a JSON config into complete-read inference sample objects.

    SG directory BAM paths are expanded recursively in natural order. Per-sample
    read-based mode searches motifs in read coordinates; a supplied reference
    still permits aligned reference positions, while absent reference produces
    ``-1`` positions. Unscoped SG read-based sources stream until EOF and
    explicit contigs use indexed regions. ``read_based`` is a legacy override.

    Returns labelled samples followed by tagged samples, with ``num_classes``
    injected and polishing/filter fields forwarded unchanged. An optional
    labelled-sample ``pod5`` file or directory supplies raw signal in POD5
    traversal order; it requires one BAM file, which need not be sorted or
    coordinate-indexed. ``bam_index`` names an optional saved read-ID index.
    ``pod5_bam_index`` supplies the shared startup index when the JSON omits it.
    """
    validation_labels = class_labels or [str(index) for index in range(num_classes)]
    config = load_supervised_config(
        json_config_path,
        class_labels=validation_labels,
        allow_pod5=True,
    )
    return samples_from_config(config, num_classes, read_based=read_based, pod5_bam_index=pod5_bam_index)


def classifier_input_configs(args, *, class_labels=None):
    """Return validated supervised configs from direct CLI inputs or JSON files.

    Both sources pass through the same schema/default normalization. Before
    loading a checkpoint, omit ``class_labels`` to defer class-index bounds
    checks; afterwards supply the effective class order or count placeholders.
    """
    direct = resolve_classifier_input_config(args, class_labels=class_labels)
    if direct is not None:
        return [direct]
    return [
        load_supervised_config(path, class_labels=class_labels, allow_pod5=True)
        for path in args.bam_config or []
    ]


def samples_from_config(config, num_classes, *, read_based=False, pod5_bam_index=None):
    """Expand a validated config into labelled/tagged inference samples.

    Uses the same expansion for JSON and CLI sources: recursively discover SG
    BAMs in natural order, resolve contig spans, and forward target and polishing
    settings. ``read_based`` retains the global JSON override. A shared POD5
    index fills omitted per-sample indexes without mutating the input config.
    """
    labelled_data = []
    tagged_data = []

    # 1. Parse Labelled Samples
    for sample_kwargs in config.get("labelled_samples", []):
        kwargs = dict(sample_kwargs)
        if kwargs.get("pod5"):
            if pod5_bam_index is not None:
                kwargs.setdefault("bam_index", pod5_bam_index)
            find_pod5_files(kwargs["pod5"])
            if not Path(kwargs["bam_path"]).is_file():
                raise ValueError("POD5 inference requires a single BAM file, not a directory")
        contigs = kwargs.pop("contigs", [])
        configured_read_based = kwargs.get("read_based", False)
        if not isinstance(configured_read_based, bool):
            raise TypeError(
                f"read_based must be true or false for {kwargs.get('bam_path')!r}, "
                f"got {configured_read_based!r}"
            )
        sample_read_based = configured_read_based
        if read_based:
            sample_read_based = True
            kwargs["read_based"] = True

        if sample_read_based:
            kwargs["chrom_info"] = [(c, 0, None) for c in contigs]
            kwargs.setdefault("ref", None)
        else:
            kwargs["chrom_info"] = _get_chrom_info(contigs, kwargs["ref"])
        kwargs['num_classes']=num_classes
        
        # Extract the bam_path to check if it's a file or directory
        bam_path_str = kwargs.pop("bam_path")
        path_obj = Path(bam_path_str)
        
        if path_obj.is_dir():
            # Find all bam files, convert to string, and naturally sort them
            bam_files = natsorted([str(p) for p in path_obj.rglob('*.bam')])
            for bf in bam_files:
                # Re-inject the individual bam_path along with the rest of the kwargs
                labelled_data.append(LabelledBamSample(bam_path=bf, **kwargs))
        else:
            labelled_data.append(LabelledBamSample(bam_path=bam_path_str, **kwargs))

    # 2. Parse Tagged Samples
    for sample_kwargs in config.get("tagged_samples", []):
        kwargs = dict(sample_kwargs)
        contigs = kwargs.pop("contigs", [])
        kwargs["chrom_info"] = (
            [(contig, 0, None) for contig in contigs]
            if kwargs["label_coordinates"] == "read"
            else _get_chrom_info(contigs, kwargs["ref"])
        )
        kwargs['num_classes']=num_classes
        # Extract the bam_path to check if it's a file or directory
        bam_path_str = kwargs.pop("bam_path")
        path_obj = Path(bam_path_str)
        
        if path_obj.is_dir():
            # Find all bam files, convert to string, and naturally sort them
            bam_files = natsorted([str(p) for p in path_obj.rglob('*.bam')])
            for bf in bam_files:
                # Re-inject the individual bam_path along with the rest of the kwargs
                tagged_data.append(TaggedBamSample(bam_path=bf, **kwargs))
        else:
            tagged_data.append(TaggedBamSample(bam_path=bam_path_str, **kwargs))

    return labelled_data+tagged_data

def validate_pod5_inference_options(samples):
    """Validate the BAM/POD5 input and optional read index.

    Require one BAM/POD5 sample and validate any supplied read index. Return
    whether the run uses POD5, which also determines the output sort-order tag.
    """
    has_pod5 = any(getattr(sample, 'pod5', None) for sample in samples)
    if has_pod5:
        validate_pod5_samples(samples)
        if samples[0].bam_index:
            with BamReadIndex(samples[0].bam_index, samples[0].bam_path):
                pass
    return has_pod5


def inference_bam_header(bam_path, *, pod5_input=False):
    """Copy an input header, marking POD5 traversal output explicitly unsorted."""
    with pysam.AlignmentFile(bam_path, 'rb', check_sq=False) as bam:
        if not pod5_input:
            return pysam.AlignmentHeader.from_text(str(bam.header))
        header = bam.header.to_dict()
    hd = header.setdefault('HD', {})
    hd['SO'] = 'unsorted'
    hd.pop('SS', None)
    hd.pop('GO', None)
    return pysam.AlignmentHeader.from_dict(header)


def prepare_classifier_pod5_input(args):
    """Prepare an optional shared index and outputs before distributed setup.

    Read config inputs without loading a model. POD5 startup reserves outputs
    and builds/validates the index on rank zero, then shares its path via the
    torchrun rendezvous store. SG startup retains its existing output lifecycle.
    Return ``(index_path, store)``; both are ``None`` for SG inputs.
    """
    from types import SimpleNamespace
    from porefm.inference.pod5_input import prepare_before_process_group, prepare_pod5_samples

    # The CLI's empty default means unspecified labels. Class-count checks
    # run again when constructing samples once the checkpoint's output size is known.
    validation_labels = args.class_labels or None
    samples = []
    for config in classifier_input_configs(args, class_labels=validation_labels):
        for entry in config.get('labelled_samples', []) + config.get('tagged_samples', []):
            samples.append(SimpleNamespace(
                bam_path=entry['bam_path'], pod5=entry.get('pod5'), bam_index=entry.get('bam_index'),
            ))
    if not any(sample.pod5 for sample in samples):
        return None, None
    validate_pod5_samples(samples)

    def prepare():
        prepare_run_outputs(args, command='infer', directories={'output': args.output})
        return prepare_pod5_samples(samples, args.output)

    return prepare_before_process_group(prepare)


def create_model_from_pretrain(args, add_signal_cross_attn=None):
    """Construct a classifier head and load pretrained backbone weights.

    Architecture settings are derived from checkpoint configuration and CLI
    options. Returns the initialized model plus pretrained ``meta`` mapping.
    """
    pretrain_ckpt = torch.load(args.pretrain_load_path, map_location="cpu")
    pretrain_config_dict = get_backbone_config_from_checkpoint(
        pretrain_ckpt,
        source=args.pretrain_load_path,
    )
    
    pretrain_config=ONTBackboneConfig(**pretrain_config_dict)
    
    if args.classifier_type=='cnn':
        classifier_config_dict=dict(hidden_dim=pretrain_config.hidden_dim,
                                                  out_dim=args.num_classes,
                                                  fc_dim_ff=args.fc_dim_ff,
                                                  base_emb_only=args.base_emb_only,
                                                  classifier_type='cnn')

    else:
        if args.base_seq_window:
            base_seq_window=(args.base_seq_window, args.base_seq_window)
        else:
            base_seq_window=pretrain_config.base_seq_window

        classifier_config_dict = dict(hidden_dim=pretrain_config.hidden_dim,
                                    dim_feedforward=pretrain_config.dim_feedforward,
                                    out_dim=args.num_classes, fc_dim_ff=args.fc_dim_ff,
                                    num_heads=pretrain_config.num_heads,
                                    kv_num_heads=pretrain_config.kv_num_heads,
                                    base_seq_window=base_seq_window,
                                    dropout=0.0,
                                    classifier_type=args.classifier_type,
                                    predict_center=True, residual_control=None,
                                    cross_attn_temp=pretrain_config.cross_attn_temp,
                                    expected_ratio=pretrain_config.expected_ratio,
                                    pre_attention_mixer_type=pretrain_config.pre_attention_mixer_type)
        
    model = ONTModelForClassification(
        classifier_config_dict=classifier_config_dict,
        pretrain_config=pretrain_config, add_signal_cross_attn=add_signal_cross_attn
        )

    state_dict=get_pretraining_state_dict_from_checkpoint(pretrain_ckpt)
    state_dict=strip_module_prefix(state_dict)
    model.pretrain.load_state_dict(state_dict, strict=False)
    model.preprocessing = preprocessing_from_checkpoint(pretrain_ckpt, required=False)
    model.pretrain.preprocessing = model.preprocessing
    pretrain_meta=pretrain_ckpt.get('meta', {})
    
    return model, pretrain_meta


def _get_unique_sorted_mask(arr):
    if arr.size == 0:
        return arr
    
    # Create a mask: True if the element is different from the previous one
    mask = np.empty(arr.shape, dtype=bool)
    mask[0] = True
    mask[1:] = arr[1:] != arr[:-1]
    
    return mask

def _stitch_1d_distances_ultra_fast(metric_array, read_positions, ref_positions, pad_base_seq, readnames):
    """
    Stitches chunked 1-D L2 metrics with maximum throughput.
    """
    read_wise_metric = []
    read_pos = []
    read_wise_refpos = []
    
    read_mask = _get_unique_sorted_mask(readnames)
    boundaries = np.where(read_mask[1:])[0] + 1
    chunks_per_read = np.split(np.arange(len(readnames)), boundaries)
    
    for read_indices in chunks_per_read:
        read_length = int(read_positions[read_indices].max().item()) + 1
        
        # Pre-allocate 1-D tensors (Notice no 'dim' argument here)
        # Initialize with NaNs so unwritten padded edges don't default to 0.0
        accum_metric = torch.full((read_length, metric_array.size(2)), float('nan'), dtype=metric_array.dtype)
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
            
            # Calculate overlap to determine split_offset
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
                
                # In-place 1-D assignment
                accum_metric[write_start:write_end] = write_metric
                accum_ref[write_start:write_end] = write_ref
                
                last_written_pos = write_end - 1
        
        valid_label_rows=torch.where(~accum_metric.isnan().any(dim=1))[0]
        
        read_wise_metric.append(accum_metric[valid_label_rows])
        read_pos.append(valid_label_rows)
        read_wise_refpos.append(accum_ref[valid_label_rows])
        
    return read_mask, read_wise_metric, read_pos, read_wise_refpos

def bam_writer_worker(write_queue, bam_output, bam_threads, out_header, base, mod_codes, seq_type, temperature):
    """Consume prediction batches and write multiclass MM/ML BAM records.

    Queue items are lists of inference-step outputs; ``None`` terminates the
    worker. Each group is stitched and temperature-scaled before one MM
    group per ``mod_codes`` entry is written. Every queue item is
    acknowledged with ``task_done`` and the output header gains a ``@PG`` row.
    """
    out_header = add_pg_to_header(out_header)
    with pysam.AlignmentFile(bam_output, "wb", threads=bam_threads, header=out_header) as outf:
        while True:
            item = write_queue.get()
            
            if item is None:
                write_queue.task_done()
                break
                
            predictions = collect_checkpoint_predictions(
                item,
                temperature=temperature,
            )

            for i, rname in enumerate(predictions.readnames):
                rname = str(rname)
                r_metric = predictions.probs_raw[i]
                r_pos_indices = predictions.read_pos[i]

                read_dict = predictions.pysam_data[rname]
                seq = read_dict['seq']
                if int(read_dict['flag']) & 16:
                    seq = revcomp(seq)

                mm_string, ml_array = encode_mm_ml_tags(
                    seq, base, r_pos_indices, r_metric, seq_type,
                    mod_codes=mod_codes, strand="+", skipped_flag="?"
                )

                new_read = pysam.AlignedSegment.from_dict(read_dict, out_header)

                if len(ml_array)>0:
                    new_read.set_tag("MM", mm_string, value_type="Z")
                    new_read.set_tag("ML", array.array('B', ml_array))

                outf.write(new_read)
                    
            write_queue.task_done()

def make_mm_ml_tags(seq, base, coords, probs, seq_type, mod_codes, strand="+", skipped_flag="?"):
    """
    Build SAM MM/ML tags for multiple modification classes on a single base type.

    Parameters
    ----------
    seq : str
        Read sequence in original/as-sequenced 5'->3' orientation.
    base : str
        Fundamental base to use for candidate-base counting.
    coords : array-like of int
        0-based read coordinates of called sites.
    probs : array-like of float, shape (N, K)
        Probabilities in [0, 1]. Class 0 is canonical (ignored in tag).
        Classes 1 to K-1 correspond to mod_codes.
    seq_type : str
        'dna' or 'rna'.
    mod_codes : list[str]
        SAM modification codes, e.g. ['m', 'h'] for 5mC and 5hmC.
    strand : str
        '+' or '-'.
    skipped_flag : str or None
        '?', '.', or None.

    Returns
    -------
    tuple
        Concatenated MM string and group-major ML byte list. Invalid or
        nonmatching coordinates are skipped; coordinates are stably sorted and
        probabilities are clipped before quantization.
    """
    seq = seq.upper()
    base = base.upper()
    seq_type = seq_type.lower()
    
    seq_match_base = base
    mm_base = base

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
        raise ValueError(f"probs must be 2D array of shape (N, {len(mod_codes) + 1})")
    if len(coords) != len(probs):
        raise ValueError("coords length must match probs length")

    flag = "" if skipped_flag in (None, "") else skipped_flag
    empty_mm = "".join([f"{mm_base}{strand}{mc}{flag};" for mc in mod_codes])

    if len(coords) == 0:
        return empty_mm, []

    # Keep only in-range coords
    in_range = (coords >= 0) & (coords < len(seq))
    coords = coords[in_range]
    probs = probs[in_range]

    if len(coords) == 0:
        return empty_mm, []

    # Sort by coordinate in case input is not sorted
    order = np.argsort(coords, kind="stable")
    coords = coords[order]
    probs = probs[order]

    # Candidate positions counted by MM deltas
    if base == "N":
        base_pos = np.arange(len(seq), dtype=np.int64)
        valid = np.ones(len(coords), dtype=bool)
        ranks = coords.copy()
    else:
        seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        base_pos = np.flatnonzero(seq_bytes == ord(seq_match_base))

        if len(base_pos) == 0:
            return empty_mm, []

        ranks_all = np.searchsorted(base_pos, coords)
        in_bounds = ranks_all < len(base_pos)
        valid = np.zeros(len(coords), dtype=bool)

        if np.any(in_bounds):
            valid[in_bounds] = (base_pos[ranks_all[in_bounds]] == coords[in_bounds])

        coords = coords[valid]
        probs = probs[valid]
        ranks = ranks_all[valid]

        if len(coords) == 0:
            return empty_mm, []

    # Delta encode ranks (same for all modifications on this base)
    deltas = np.diff(np.concatenate(([-1], ranks))) - 1
    deltas_str = ",".join(map(str, deltas.tolist()))

    mm_parts = []
    ml_all = []

    # Build tags by iterating over the provided modification codes
    for i, mod_code in enumerate(mod_codes):
        # 1. MM Tag Segment
        mm_parts.append(f"{mm_base}{strand}{mod_code}{flag},{deltas_str};")
        
        # 2. ML Tag Segment (i + 1 because class 0 is canonical)
        mod_probs = probs[:, i + 1]
        ml = np.minimum(
            (np.clip(mod_probs, 0.0, 1.0) * 256.0).astype(np.int64),
            255
        ).astype(np.uint8)
        
        ml_all.extend(ml.tolist())

    return "".join(mm_parts), ml_all


def process_and_write_checkpoint(eval_data_chunk, checkpoint, output_dir, class_labels, temperature, rank=None):
    """Stitch one output checkpoint and write aligned metadata/data Parquet.

    The metadata file has one row per read and records the length used to split
    the flat data table. The data file stores read/reference positions, raw
    probabilities and logits for each class. ``rank`` prefixes distributed
    filenames.
    """
    predictions = collect_checkpoint_predictions(
        eval_data_chunk,
        temperature=temperature,
    )

    if len(predictions.logits) == 0:
        return

    flat_logits = np.concatenate(predictions.logits, axis=0)
    flat_probs_raw = np.concatenate(predictions.probs_raw, axis=0)

    flat_read_pos = np.concatenate(predictions.read_pos, axis=0)
    flat_refpos = np.concatenate(predictions.refpos, axis=0)

    # 3. Create Meta Table
    meta_table = pa.table({
        'readname': predictions.readnames,
        'refname': predictions.refnames,
        'strand': predictions.strands,
        'length': predictions.lengths
    })
    
    # 4. Create Data Table with Dynamic Class Labels
    data_dict = {
        'read_pos': flat_read_pos,
        'refpos': flat_refpos
    }
    
    # Dynamically assign column names based on class_labels
    for i, class_name in enumerate(class_labels):
        data_dict[f'prob_{class_name}'] = flat_probs_raw[:, i]
        data_dict[f'logit_{class_name}'] = flat_logits[:, i]
        
    data_table = pa.table(data_dict)
    
    # 5. Write to Disk
    os.makedirs(output_dir, exist_ok=True)
    rank_prefix = f"rank_{rank}_" if rank is not None else ""
    pq.write_table(meta_table, os.path.join(output_dir, f"{rank_prefix}checkpoint_{checkpoint}_meta.parquet"), compression='snappy')
    pq.write_table(data_table, os.path.join(output_dir, f"{rank_prefix}checkpoint_{checkpoint}_data.parquet"), compression='snappy')

def read_inference_step(model, batch, device, batch_size, num_trim=16):
    """Run classifier inference over sub-batches of a complete-read batch.

    Logits outside labelled positions or within ``num_trim`` tokens of either
    valid sequence edge are filled with NaN before CPU return. Returns
    logits, read positions, reference positions, base padding mask,
    read names, reference names, strands, and pysam dictionaries.
    """
    base_seq_input, ts_input, percentiles_input, pad_base_seq, pad_ts, labels_idx, readnames, refnames, ref_positions, read_positions, min_start, strands, pysam_data = batch

    trimmed_pad_base_seq = ignore_edges(pad_base_seq, num_trim)
    valid_pos = trimmed_pad_base_seq & labels_idx

    total_logits=[]
    for sub_batch in range(0, base_seq_input.size(0), batch_size):
        batch_base_seq_input = base_seq_input[sub_batch : sub_batch + batch_size].to(device, non_blocking=True)
        batch_ts_input = ts_input[sub_batch : sub_batch + batch_size].to(device, non_blocking=True)
        batch_percentiles_input = percentiles_input[sub_batch : sub_batch + batch_size].to(device, non_blocking=True)
        batch_pad_base_seq = pad_base_seq[sub_batch : sub_batch + batch_size].to(device, non_blocking=True)
        batch_pad_ts = pad_ts[sub_batch : sub_batch + batch_size].to(device, non_blocking=True)

        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = model(batch_base_seq_input, batch_ts_input, percentiles_input= batch_percentiles_input,
                        pad_base_seq=batch_pad_base_seq, pad_ts=batch_pad_ts)
            logits = out['classifier_logits']
            total_logits.append(logits.cpu().float())

    total_logits=torch.concatenate(total_logits)
    total_logits.masked_fill_(~valid_pos.unsqueeze(-1), float("nan"))
    
    out_tuple = (
        total_logits,
        read_positions, 
        ref_positions, 
        pad_base_seq, 
        readnames, 
        refnames, 
        strands,
        pysam_data
    )

    return out_tuple


def load_classifier_model(model_path, device):
    """Load a checked classifier and its non-parameter checkpoint metadata.

    Model construction and parameter validation are delegated to the public
    loader. The raw payload is read separately because inference also needs
    pretraining collection metadata and class labels.
    """
    model = load_checked_model(
        model_path,
        model_type="classifier",
        device=device,
    )
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    return model, checkpoint


def inference(args):
    """Run checkpointed classifier inference from a CLI namespace.

    The routine loads a classifier, creates complete-read datasets from JSON
    configs or direct sample options, optionally compiles the model, and writes
    stitched Parquet or BAM output. Optional ``exclude_bed`` removes whole reads
    whose reference span overlaps a matching region before signal decoding.
    The region file is validated before checkpoint loading or output creation,
    and its SHA-256 digest is recorded in the startup manifest.
    """
    exclude_bed = getattr(args, 'exclude_bed', None)
    if exclude_bed:
        args.exclude_bed_sha256 = get_excluded_regions(exclude_bed).sha256
    resolve_cli_preprocessing(args)
    checkpoint_batch_step = args.checkpoint_batch_step
    output=args.output
    pod5_bam_index, startup_store = prepare_classifier_pod5_input(args)
    rank, world_size, local_rank = (
        setup_distributed(store=startup_store) if startup_store is not None else setup_distributed()
    )

    if pod5_bam_index is None:
        prepare_run_outputs(args, command="infer", directories={
            "output": output,
        }, distributed=dist)

    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    model, model_ckpt = load_classifier_model(args.model_load_path, device)
    pretrain_meta = model_ckpt['pretrain_meta']
        
    for p in model.parameters():
        p.requires_grad = False
    
    pretrain_config = model.pretrain.config
    class_labels=model_ckpt['meta']['class_labels']
    
    num_classes = model.classifier_config.out_dim
    args.base, args.mod_code = resolve_modification_metadata(
        args.base, args.mod_code,
        metadata=model_ckpt['meta'],
        num_classes=num_classes,
        legacy_base=class_labels[0],
        require_bam_tags=args.output_format in {'bam', 'both'},
    )
    base = args.base
    
    seq_type, pore_type = args.seq_type, args.pore_type
    max_base_seq_len, max_ts_len = pretrain_config.max_base_seq_len, pretrain_config.max_ts_len
    ts_dim = pretrain_config.ts_dim
    
    model=model.to(device)
    model.eval()
    
    t=time.time()
    
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16), torch.no_grad():
        model(torch.ones(1,max_ts_len).long().to(device),
              torch.randn(1,max_ts_len,ts_dim).to(device),
              percentiles_input=torch.randn(1,max_ts_len, ts_dim).to(device),
              pad_base_seq=torch.ones(1,max_ts_len).to(torch.bool).to(device),
              pad_ts=torch.ones(1,max_ts_len).to(torch.bool).to(device),
             );

    
    if args.disable_compile==False:
        model = torch.compile(model, dynamic=True)
                   
    size = min(max_ts_len, args.size or max_ts_len)
    step = resolve_downstream_step(
        size,
        seq_type,
        args.step,
        context="classifier inference",
        announce=rank == 0,
    )
    
    kmer_table=resolve_cli_preprocessing(args).kmer_table

    output_format = args.output_format
    
    bam_config_lists=[]
    validation_labels = args.class_labels or [str(index) for index in range(num_classes)]
    for config in classifier_input_configs(args, class_labels=validation_labels):
        bam_config_lists.extend(
            samples_from_config(
                config,
                num_classes=num_classes,
                read_based=args.read_based,
                pod5_bam_index=pod5_bam_index,
            )
        )

    if not bam_config_lists:
        raise ValueError('No BAM samples found; supply --bam_path or --bam_config')

    if args.read_based:
        missing_motif = [sample.bam_path for sample in bam_config_lists if not getattr(sample, "motif", None)]
        if missing_motif:
            raise ValueError("--read_based requires motif info in every BAM config sample")

    has_pod5 = validate_pod5_inference_options(bam_config_lists)
    ds = CompleteLabelledReadIterableDataset(
        bam_list=bam_config_lists,
        size=size,
        step=step,
        move_downsample_factor=1,
        seq_type=seq_type,
        chunking_mode=resolve_chunking_mode(args), kmer_table=kmer_table,
        reads_per_chunk=args.reads_per_chunk,
        min_read_length=args.min_read_length,
        min_qscore=args.min_qscore,
        pore_type=pore_type,
        base_seed=args.seed,
        rank=rank,             
        world_size=world_size,
        discard_empty_labels=True,
        tags_to_drop=['ML', 'MM', 'SG', 'mv'],
        return_pysam=output_format in {"bam", "both"},
        read_based=args.read_based,
        expected_ts_dim=pretrain_config.ts_dim,
        exclude_bed=exclude_bed,
        allow_pod5=True,
        
    )

    dataloader_workers = args.dataloader_workers
    if has_pod5:
        dataloader_workers = pod5_worker_count(ds.pod5_files, dataloader_workers, rank, world_size)
        print(f"POD5 inference: rank {rank} uses {dataloader_workers} DataLoader workers "
              f"for {len(ds.pod5_files[rank::world_size])} POD5 files.", flush=True)
    dataloader_kwargs = dict(
        batch_size=None,
        num_workers=dataloader_workers,
        pin_memory=True,
    )

    if dataloader_workers > 0:
        dataloader_kwargs.update(
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
        )

    dataloader = DataLoader(ds, **dataloader_kwargs)

    temperature=args.temperature
    batch_num, checkpoint = 0, 0
    eval_data = []
    futures = []
    
    batch_size=args.batch_size   
    
    out_header = inference_bam_header(bam_config_lists[0].bam_path, pod5_input=has_pod5)
    
    # --- Format-Specific IO Setup ---
    if output_format in {"parquet", "both"}:
        io_executor = ThreadPoolExecutor(max_workers=2)

    if output_format in {"bam", "both"}:
        bam_queue = queue.Queue(maxsize=10) # Limit size to prevent RAM overflow
        # Start the dedicated BAM thread
        output_bam_name = 'output.bam' if world_size == 1 else f'output.rank{rank}.bam'
        output_bam_path=os.path.join(output, output_bam_name)
        
        bam_thread = threading.Thread(
            target=bam_writer_worker, 
            args=(
                bam_queue, output_bam_path, args.bam_threads, 
                out_header, base, args.mod_code, 
                seq_type, temperature
            ),
            daemon=True
        )
        bam_thread.start()
        
    
    for batch in dataloader:
        batch_data = read_inference_step(
            model,
            batch,
            device,
            batch_size,
            num_trim=args.num_trim,
        )
        
        eval_data.append(batch_data)
        
        batch_num+=1

        # --- Checkpoint and Logging ---
        if (batch_num)%checkpoint_batch_step==0:
            checkpoint+=1
            
            if output_format in {"parquet", "both"}:
                futures. append(
                    io_executor.submit(                        
                        process_and_write_checkpoint, 
                        eval_data,
                        checkpoint,
                        args.output,
                        class_labels,
                        temperature,
                        rank if world_size > 1 else None,
                    )
                )
            if output_format in {"bam", "both"}:
                
                bam_queue.put(eval_data)
                
            eval_data = []
            
            print(f"\n{str(datetime.datetime.now())}: Checkpoint:{checkpoint}  Steps:{batch_num}  Time={time.time()-t:.2f}", flush=True)
            
    if len(eval_data)>0:
            checkpoint += 1
            
            if output_format in {"parquet", "both"}:
                futures. append(
                    io_executor.submit(                        
                        process_and_write_checkpoint, 
                        eval_data,
                        checkpoint,
                        args.output,
                        class_labels,
                        temperature,
                        rank if world_size > 1 else None,
                    )
                )
            if output_format in {"bam", "both"}:
                bam_queue.put(eval_data)

            print(f"\n{str(datetime.datetime.now())}: Checkpoint:{checkpoint}  Steps:{batch_num}  Time={time.time()-t:.2f}", flush=True)
            
    if output_format in {"parquet", "both"}:
        for fut in futures:
            fut.result()
        io_executor.shutdown(wait=True)
    if output_format in {"bam", "both"}:
        bam_queue.put(None) # Send termination sentinel
        bam_queue.join()    # Wait for queue to empty
        bam_thread.join()   # Wait for thread to exit safely
    return
    
if __name__=='__main__':
    
    print('\n%s:\nCommand: python %s\n' %(str(datetime.datetime.now()), ' '.join(sys.argv)), flush=True)
    
    args = parse_args()

    print(f"{str(datetime.datetime.now())}: Starting modification detection.", flush=True)

    
    inference(args)
        
    print(f"{str(datetime.datetime.now())}: Modification detection finished.", flush=True)
