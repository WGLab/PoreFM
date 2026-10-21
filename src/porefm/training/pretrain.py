"""Command-line configuration and training loop for multimodal pretraining.

The module supports masked-base language modeling, random-substitution
detection, and masked signal-representation prediction, including distributed
training and log-guided recovery of iterable-dataset worker positions.
"""

from porefm.preprocessing import attach_preprocessing
from porefm.cli.preprocessing_args import add_preprocessing_arguments, resolve_cli_preprocessing
from porefm.cli.chunking_args import add_chunking_arguments, resolve_chunking_mode
import argparse
import sys

from porefm.cli._parser import PoreFMArgumentParser, PoreFMHelpFormatter
from porefm.common.run_outputs import add_output_safety_argument, prepare_run_outputs
from porefm.config.validation import load_pretraining_config
from porefm.models.configuration import (
    EMA_SIGNAL_DROPOUT_MODES,
    get_backbone_config_from_checkpoint,
)

class DefaultsHelpFormatter(PoreFMHelpFormatter):
    """Append argparse defaults to optional arguments whose help omits them."""


class DefaultsArgumentParser(PoreFMArgumentParser):
    """Argument parser that reserves help text for otherwise undocumented options."""

    def add_argument(self, *args, **kwargs):
        """Add an argument with a temporary blank help value when none is supplied."""
        kwargs.setdefault('help', ' ')
        return super().add_argument(*args, **kwargs)


def finalize_default_help(parser):
    """Replace temporary blank help with a default annotation or no text."""
    for action in parser._actions:
        if action.help is argparse.SUPPRESS or (action.help or '').strip():
            continue
        if (
            action.option_strings
            and action.default is not argparse.SUPPRESS
            and action.default is not None
            and not action.required
        ):
            action.help = '(default: %(default)s)'
        else:
            action.help = None


def build_arg_parser():
    """Build the full multimodal-pretraining command-line parser."""
    parser = DefaultsArgumentParser(
        formatter_class=DefaultsHelpFormatter,
        description="Train a multi-modal model on Nanopore data.",
    )

    # --- Required Arguments ---
    parser.add_argument(
        '--bam_config',
        type=str,
        help='Path to a versioned pretraining JSON config containing signal-BAM samples.',
    )
    parser.add_argument('--model_save_path', type=str, help='Directory to save trained models.')
    parser.add_argument('--size', type=int, help='Size of the signal chunk window.')
    parser.add_argument('--step', type=int, help='Sliding step size for the signal chunk window.')
    parser.add_argument("--seq_type", help='Specify DNA or direct RNA sequencing.', choices=['dna', 'rna'], type=str, required=True)
    parser.add_argument('--model_load_path', type=str, help='Path to existing trained model.')
    add_chunking_arguments(parser)
    parser.add_argument('--kmer_dict_path', type=str, help='Path to kmer expected signal file.')
    parser.add_argument(
        '--eval_only',
        help='Evaluate one finite pass over each assigned BAM without updating or saving the model; --max_steps optionally caps local batches.',
        default=False,
        action='store_true',
    )

    # --- QC Arguments ---
    parser.add_argument('--min_read_length', type=int, default=100, help='Minimum read length.')
    parser.add_argument('--min_qscore', type=int, default=10, help='Minimum read qscore.')

    # --- Training Arguments ---
    parser.add_argument('--max_steps', type=int, default=1e10, help='Number of batch update steps.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.set_defaults(auto_batch_size=True)
    parser.add_argument('--no_auto_batch_size', dest='auto_batch_size', action='store_false',
                        help='Disable automatic batch size tuning.')
    parser.add_argument('--auto_batch_size_safety', type=float, default=0.9,
                        help='Fraction of reported free memory to target when auto-tuning batch size.')
    parser.add_argument('--max_auto_batch_size', type=int, default=0,
                        help='Optional upper limit for auto-tuned batch size (0 disables the limit).')

    # --- Data Processing Arguments ---
    parser.add_argument('--checkpoint_batch_step', type=int, default=10000, help='Number of batches before saving a model checkpoint and logging.')
    parser.add_argument('--num_warmp_cp', type=int, default=1, help='Number of checkpoints for warmup to peak LR.')
    parser.add_argument('--num_peak_cp', type=int, default=50, help='Number of checkpoints to keep peak LR.')
    parser.add_argument('--seed', type=int, default=0, help='Seed to use for data batching.')
    parser.add_argument('--reads_per_chunk', type=int, default=500, help='Number of reads to bundle into a single data chunk.')
    parser.add_argument('--pore_type', type=str, default='R10/RNA004', help='Pore type.')
    parser.add_argument('--dataloader_workers', type=int, default=0, help='DataLoader worker processes for prefetching/pinning.')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Batches prefetched per worker (only if workers > 0).')
    parser.add_argument('--use_polished_sequence', default=False, action='store_true',
                        help='Use reference-guided polished base sequences. Requires mapped BAM records with CIGAR and MD information.')
    parser.add_argument('--polish_indel_limit', type=int, default=5,
                        help='Maximum unmasked insertion/deletion length corrected during polishing.')
    parser.add_argument('--polish_vcf_path', default=None,
                        help='Optional VCF/BCF whose ALT SNPs and indel loci are preserved during polishing.')

    # --- Model Architecture Arguments ---
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--use_loaded_config', help='config loaded from model path', default=False, action='store_true')
    model_group.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size of the model.')
    model_group.add_argument('--num_heads', type=int, default=8, help='Number of attention heads.')
    model_group.add_argument('--num_layers_base_seq_self', type=int, default=2, help='Number of base sequence self-attn layers.')
    model_group.add_argument('--num_layers_ts_self', type=int, default=2, help='Number of signal self-attn layers.')
    model_group.add_argument('--num_layers_cross', type=int, default=2, help='Number of cross-attn layers.')
    model_group.add_argument('--dim_feedforward', type=int, default=1024, help='Dimension of the feedforward network.')
    model_group.add_argument('--base_seq_window', type=int, default=16, help='Window size for base sequence self-attn.')
    model_group.add_argument('--ts_window', type=int, default=64, help='Window size for signal self-attn.')
    model_group.add_argument('--kv_num_heads', type=int, default=0, help='How many KV heads to use for GQA. GQA will be applied to corss-attention layers only. Default is value is 0, in which case kv_num_heads is set to num_heads and full MHA is done.')
    model_group.add_argument('--pre_attention_mixer_type', type=str, choices=['cnn', 'self_attn'], help='Local content mixer for queries in cross attention layer. Choices are "cnn" and "self_attn" if the argument is specified otherwise it is identity by default.', default=None)
    model_group.add_argument('--residual_control', type=str, choices=['skip', 'gate'], help='Residual control for last layer in base to signal cross attention. Choices are "skip" and "gate" if the argument is specified otherwise it is usual residual connection by default.', default=None)
    model_group.add_argument('--add_percentile', help='Use signal percentiles as a feature and target.', default=False, action='store_true')
    model_group.add_argument('--disable_signal_cross_attn', dest="add_signal_cross_attn", help='Use signal percentiles as a feature and target.', action='store_false')
    model_group.add_argument('--weight_decay', type=float, default=0, help='AdamW weight decay.')
    model_group.add_argument('--self_attn_temp', type=float, default=1.0, help='self_attn_temp.')
    model_group.add_argument('--cross_attn_temp', type=float, default=1.0, help='cross_attn_temp.')
    model_group.add_argument('--signal_kernels', type=int, nargs=4, default=[7, 7, 7, 7], help='Kernel width per conv layer for signal projection CNN.')
    model_group.add_argument('--signal_strides', type=int, nargs=4, default=[2, 3, 1, 1], help='Stride per conv layer for signal projection CNN. ')
    model_group.add_argument('--ts_dim', type=int, default=6, help='Move-table resolution from the BAM mv tag; must equal mv[0] and divide prod(signal_strides).')

    # --- Training Task Arguments ---
    task_group = parser.add_argument_group('Training Tasks')
    task_group.add_argument('--disable_compile', help='Disable model compile.', default=False, action='store_true')
    task_group.add_argument('--resume', help='True resume from --model_load_path: load optimizer/scheduler state, restore global_step, append logs, and continue checkpoint numbering.', default=False, action='store_true')
    task_group.add_argument('--resume_interrupted_log', '--interrupted_resume_log', dest='resume_interrupted_log',
                            type=str, default=None,
                            help='Path to the stdout/stderr log from an interrupted run. Implies --resume and resumes each data-loader worker from the BAM it had open when the loaded checkpoint was saved. If --model_load_path is omitted, infer it as <logged model_save_path>/model.<last logged checkpoint>.')
    task_group.add_argument('--load_optimizer', help='Load the optimizer state_dict.', default=False, action='store_true')
    task_group.add_argument('--mask_prob_range', type=float, nargs=2, default=[0.15, 0.4], help='Probability of masking tokens for pre-training tasks, start and end value after warmup.')
    task_group.add_argument('--mask_prob_warmup', type=int, default=10, help='Number of checkpoint_batch_step to use for warmup.')
    task_group.add_argument('--mask_base_span_range', type=int, nargs=2, default=[5, 9], help='How many base tokens in a span to mask.')
    task_group.add_argument('--mask_signal_span', type=int, default=1, help='How many signal tokens in a span to mask.')
    task_group.add_argument('--lm_label_smoothing', type=float, default=0.05,
                            help='Label smoothing amount for sublm and masklm cross-entropy losses.')
    task_group.add_argument('--task_list', nargs='+', default=['sublm', 'masklm', 'mtr'],
                            choices=['sublm', 'masklm', 'mtr'],
                            help='A list of training tasks to use.')
    task_group.add_argument('--weights', nargs='+', type=int, default=[1, 1, 1],
                            help='A list of sampling weights corresponding to the tasks in --task_list.')
    task_group.add_argument('--num_loss_trim', nargs=2, type=int, default=[16, 32],
                            help='[BASE_TRIM, SIGNAL_TRIM]. How many tokens to trim towards each boundary edge for loss computation.')
    task_group.add_argument(
        '--signal_dropout_prob',
        type=float,
        default=None,
        help='Per-forward probability of dropping raw signal or percentiles with equal probability. Omit to inherit with --use_loaded_config/--resume, otherwise 0.',
    )
    task_group.add_argument(
        '--ema_signal_dropout_mode',
        choices=sorted(EMA_SIGNAL_DROPOUT_MODES),
        default=None,
        help='EMA modality-dropout coupling. Omit to inherit with --use_loaded_config/--resume; new runs default to disabled.',
    )

    finalize_default_help(parser)
    add_output_safety_argument(parser)
    add_preprocessing_arguments(parser)
    return parser


if __name__ == '__main__' and any(arg in ('-h', '--help') for arg in sys.argv[1:]):
    build_arg_parser().parse_args()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, IterableDataset
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import random, time, os, datetime, csv, math, re, shlex

from collections import defaultdict

from dataclasses import dataclass, asdict

from contextlib import nullcontext

import torch.distributed as dist

torch.multiprocessing.set_start_method('spawn', force=True)
torch.set_num_threads(1)

from porefm.models.backbone.model import ONTBackboneConfig, ONTModelForPreTraining
from porefm.models.backbone.utils import strip_module_prefix
from porefm.models.checkpointing import get_pretraining_state_dict_from_checkpoint, modular_pretraining_payload
from porefm.data.bam import build_lookup_table
from porefm.data.reader import (
    BamIterableDataset,
    collate_modal_batch,
    validate_ddp_bam_sharding,
)
from porefm.data.polishing_options import validate_polishing_options
from porefm.common.utils import align_to_power_of_two, estimate_per_sample_memory, get_dummy_loss, get_warmup_cosine_schedule, TorchSeqToSignal, CurriculumManager
from porefm.common.tensor_masks import ignore_edges

torch.set_float32_matmul_precision('high')
torch._dynamo.config.capture_scalar_outputs = True

def make_param_groups(model, weight_decay=1e-4):
    """Split trainable parameters into AdamW decay and no-decay groups.

    Biases, one-dimensional parameters, and embedding tables are excluded from
    weight decay. All remaining trainable tensors receive ``weight_decay``.
    """
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_bias = name.endswith(".bias")
        is_low_dim = (p.ndim <= 1)
        is_embedding = name.endswith("embed.weight") or ("_embed." in name)

        if is_bias or is_low_dim or is_embedding:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def get_mask_span(input_size, device, mask_prob: float = 0.15, span: int = 1) -> torch.Tensor:
    """Sample fixed-length masking spans independently for each batch row.

    Parameters
    ----------
    input_size : tuple of int
        ``(batch_size, sequence_length)``.
    device : torch.device or str
        Device on which to generate the mask.
    mask_prob : float
        Approximate fraction of positions to cover. The implementation samples
        ``int(mask_prob / span * sequence_length)`` distinct span starts, so
        overlap and right-edge clipping may lower actual coverage.
    span : int
        Number of consecutive positions beginning at each sampled start.

    Returns
    -------
    torch.Tensor
        Boolean mask with shape ``input_size``.
    """
    B, L = input_size
    seed_mask_prob=mask_prob/span
    idx = torch.rand(B, L, device=device).topk(int(seed_mask_prob*L), dim=1).indices

    # A range-add/prefix-sum construction expands every sampled start without
    # materializing a batch-by-seed-by-span tensor.
    starts = idx.clamp(min=0, max=L-1)
    ends   = (starts + span).clamp(max=L)  # exclusive

    delta = torch.zeros((B, L + 1), dtype=torch.int32, device=idx.device)

    ones  = torch.ones_like(starts, dtype=delta.dtype)
    neg1s = -torch.ones_like(ends,   dtype=delta.dtype)

    delta.scatter_add_(dim=1, index=starts, src=ones)
    delta.scatter_add_(dim=1, index=ends,   src=neg1s)

    covered = delta[:, :L].cumsum(dim=1) > 0
    return covered

def mask_base_seq_for_mlm(base_seq_input, pad_base_seq, mask_prob=0.15, span=1):
    """Replace sampled valid base tokens with the repository mask token ``5``.

    Returns a cloned sequence tensor and the final boolean mask, both with shape
    ``(batch, bases)``. Padding positions are never masked.
    """
    mask = get_mask_span(base_seq_input.shape[:2], base_seq_input.device, mask_prob, span)
    mask=mask&pad_base_seq
    corrupted = base_seq_input.clone()
    corrupted[mask] = 5
    return corrupted, mask

def sub_base_seq_for_mlm(base_seq_input, pad_base_seq, mask_prob=0.15):
    """Create random-base substitutions for the binary substitution task.

    Candidate positions exclude four valid tokens at each sequence boundary.
    Selected tokens are replaced by random canonical IDs 1--4. The returned
    integer target is ``1`` wherever the final token equals the input and ``0``
    wherever it differs; a random replacement equal to the input is therefore
    labeled unchanged.
    """
    boundary_kmers=ignore_edges(pad_base_seq, 4)
    mask = get_mask_span(base_seq_input.shape[:2], base_seq_input.device, mask_prob, 1)
    
    mask=mask&pad_base_seq&boundary_kmers
    
    
    rand_replace = torch.randint_like(base_seq_input, low=1, high=5)
   
    corrupted = base_seq_input.clone()
    corrupted[mask] = rand_replace[mask]
    
    mask=(corrupted==base_seq_input)
    return corrupted, mask.long()

def mask_ts_for_mtr(ts_input, percentiles_input, pad_ts, mask_prob=0.15, span=1):
    """Sample valid signal-token spans for masked target representation.

    ``percentiles_input`` is accepted to mirror the multimodal task interface;
    mask construction depends only on signal shape and ``pad_ts``.
    """
    mask = get_mask_span(ts_input.shape[:2], ts_input.device, mask_prob, span)
    mask=mask&pad_ts
    return mask


def get_unwrapped_model(model):
    """Remove DDP and ``torch.compile`` wrappers from a model when present."""
    model = model.module if isinstance(model, DDP) else model
    return getattr(model, "_orig_mod", model)


def get_pretraining_component(model):
    """Return the pretraining backbone from a classifier wrapper or the model itself."""
    module = get_unwrapped_model(model)
    return module.pretrain if hasattr(module, "pretrain") and hasattr(module, "classifier_head") else module


def forward_pretraining_model(
    model,
    base_seq_input,
    ts_input,
    percentiles_input,
    pad_base_seq,
    pad_ts,
    mask_mlm=None,
    mask_ts=None,
    signal_dropout_prob=0.0,
):
    """Forward pretraining tensors through a backbone or classifier-wrapped backbone.

    Classifier wrappers receive ``return_pretrain=True`` so callers get the
    task-head output dictionary rather than classifier logits.
    """
    kwargs = dict(
        base_seq_input=base_seq_input,
        ts_input=ts_input,
        percentiles_input=percentiles_input,
        pad_base_seq=pad_base_seq,
        pad_ts=pad_ts,
        mask_mlm=mask_mlm,
        mask_ts=mask_ts,
        signal_dropout_prob=signal_dropout_prob,
    )

    module = get_unwrapped_model(model)
    if hasattr(module, "pretrain") and hasattr(module, "classifier_head"):
        kwargs["return_pretrain"] = True

    return model(**kwargs)


def choose_pretraining_step_controls(task_list, weights, mask_span_start, mask_span_end, device, rank, world_size):
    """Sample a task and base-mask span consistently across distributed ranks.

    Rank zero samples weighted task choice and an inclusive integer span, then
    broadcasts their indices. Single-process execution samples locally.
    """
    if world_size > 1:
        if rank == 0:
            task_idx = random.choices(range(len(task_list)), weights=weights, k=1)[0]
            mask_span = random.randint(mask_span_start, mask_span_end)
            controls = torch.tensor([task_idx, mask_span], dtype=torch.long, device=device)
        else:
            controls = torch.zeros(2, dtype=torch.long, device=device)
        dist.broadcast(controls, src=0)
        task_idx, mask_span = controls.tolist()
        return task_list[task_idx], int(mask_span)

    return random.choices(task_list, weights=weights, k=1)[0], random.randint(mask_span_start, mask_span_end)


def training_step(model, batch, optimizer, scheduler, device, task, mask_prob, mask_base_span, mask_signal_span, add_percentile, eval_only, add_signal_cross_attn, num_base_trim=16, num_signal_trim=32, lm_label_smoothing=0.05, include_dummy_loss=False, signal_dropout_prob=0.0):
    """Run one pretraining or evaluation batch for the selected objective.

    ``batch`` contains base IDs, signal tokens, signal percentiles, and boolean
    validity masks for bases and signal. ``task`` is ``"sublm"`` (binary random
    substitution detection), ``"masklm"`` (four-class masked-base recovery), or
    ``"mtr"`` (normalized latent prediction at masked signal positions). Losses
    exclude ``num_base_trim`` or ``num_signal_trim`` valid tokens at each edge.

    Training mode clears gradients, backpropagates, clips the global gradient
    norm to one, advances optimizer and scheduler, and updates the EMA target
    encoder when configured. Evaluation runs under ``no_grad`` and does not
    mutate optimizer/model state.

    Returns
    -------
    tuple
        ``(loss, predictions, targets)``. Predictions and targets are CPU
        tensors for classification tasks and ``None`` for MTR.
    """
    pretraining_component = get_pretraining_component(model)
    base_seq_input, ts_input, percentiles_input, pad_base_seq, pad_ts = batch
    
    base_seq_input = base_seq_input.to(device, non_blocking=True)
    ts_input = ts_input.to(device, non_blocking=True)
    percentiles_input = percentiles_input.to(device, non_blocking=True)
    pad_base_seq = pad_base_seq.to(device, non_blocking=True)
    pad_ts = pad_ts.to(device, non_blocking=True)
    
    trimmed_pad_base_seq=ignore_edges(pad_base_seq, num_base_trim)
    trimmed_pad_ts=ignore_edges(pad_ts, num_signal_trim)
    
    preds, targets = None, None
    
    if not eval_only:
        optimizer.zero_grad()
    
    if task == 'sublm':
        task_base_seq, targets_full = sub_base_seq_for_mlm(base_seq_input, pad_base_seq, mask_prob)
        task_ts = ts_input

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            with torch.no_grad() if eval_only else nullcontext():
                out = forward_pretraining_model(
                    model, task_base_seq, task_ts, percentiles_input,
                    pad_base_seq, pad_ts,
                    signal_dropout_prob=signal_dropout_prob,
                )
        
        logits = out['sublm_logits'].float()[trimmed_pad_base_seq]
        targets = targets_full[trimmed_pad_base_seq]
        loss = F.cross_entropy(logits, targets, label_smoothing=lm_label_smoothing)
        preds = logits.argmax(dim=-1)

    elif task == 'masklm':
        mask_data = mask_base_seq_for_mlm(base_seq_input, pad_base_seq, mask_prob, mask_base_span)
        task_base_seq, mask_mlm = mask_data
        task_ts = ts_input

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            with torch.no_grad() if eval_only else nullcontext():
                out = forward_pretraining_model(
                    model, task_base_seq, task_ts, percentiles_input,
                    pad_base_seq, pad_ts, mask_mlm=mask_mlm,
                    signal_dropout_prob=signal_dropout_prob,
                )
        
        mask_mlm=mask_mlm&trimmed_pad_base_seq
        
        logits = out['masklm_logits'].float()[mask_mlm]
        targets = base_seq_input[mask_mlm]-1
        loss = F.cross_entropy(logits, targets, label_smoothing=lm_label_smoothing)
        preds = logits.argmax(dim=-1)
        
       
    elif task == 'mtr':
        if not getattr(pretraining_component.config, 'add_signal_cross_attn', False) or not getattr(pretraining_component.config, 'add_ema', False):
            raise ValueError("MTR requires signal-to-base cross-attention and the JEPA/MTR EMA encoder to be enabled.")

        mask_mtr = mask_ts_for_mtr(ts_input, percentiles_input, pad_ts, mask_prob, mask_signal_span)
        task_base_seq = base_seq_input
        task_ts = ts_input
        
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            with torch.no_grad() if eval_only else nullcontext():
                out = forward_pretraining_model(
                    model, task_base_seq, task_ts, percentiles_input,
                    pad_base_seq, pad_ts, mask_ts=mask_mtr,
                    signal_dropout_prob=signal_dropout_prob,
                )
        
        latent_pred=out['latent_pred']
        latent_target=out['latent_target']
        
        latent_target_norm = F.normalize(latent_target, p=2, dim=-1)
        latent_pred_norm = F.normalize(latent_pred, p=2, dim=-1)
        
        mask_mtr=mask_mtr&trimmed_pad_ts
        
        raw_loss = F.mse_loss(latent_pred_norm.float()[mask_mtr], latent_target_norm[mask_mtr], reduction='none')
        loss = raw_loss.sum(dim=-1).mean()        
     
    if not eval_only:
        if include_dummy_loss:
            loss += get_dummy_loss(model)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        
        if add_signal_cross_attn and getattr(pretraining_component.config, 'add_ema', False):
            pretraining_component.update_ema(momentum=0.996)
        

    return loss.item(), \
           preds.cpu() if preds is not None else None, \
           targets.cpu() if targets is not None else None


def custom_collate_fn(batch):
    """Collate raw modal samples into tensors and validity masks for pretraining.

    Base IDs use 0 for padding and 1--4 for A/C/G/T; mask token 5 is introduced
    later by task corruption. The trailing metadata returned by the shared
    collator is intentionally omitted.
    """
    seqs_batch, signals_batch, percentiles_batch = [], [], []
    for item_seqs, item_signals, item_percentiles in batch:
        seqs_batch.append(item_seqs)
        signals_batch.append(item_signals)
        percentiles_batch.append(item_percentiles)

    return collate_modal_batch(seqs_batch, signals_batch, percentiles_batch)[:-1]
            
def load_pretraining_bam_list(bam_config):
    """Load a pretraining JSON config as ``(BAM, probability)`` tuples."""
    return load_pretraining_config(bam_config).bam_list


BAM_OPEN_LOG_RE = re.compile(
    r"global_rank=(\d+)\s+(?:cycle=\d+\s+)?worker_id=(\d+)\.\s+BAM=(.+?)\s*$"
)
CHECKPOINT_LOG_RE = re.compile(r"Checkpoint:(\d+)\s+Steps:(\d+)")


@dataclass
class InterruptedResumePlan:
    """Recovered worker-to-BAM positions and topology for an interrupted checkpoint."""

    log_path: str
    target_checkpoint: int
    checkpoint_step: int
    resume_start_bam_by_worker: dict
    open_events_before_checkpoint: int
    completed_file_events_before_checkpoint: int
    workers_seen: int
    observed_world_size: int
    observed_workers_per_rank: int


@dataclass
class InterruptedLogCheckpointSummary:
    """Last checkpoint identity and model directory recovered from a run log."""

    log_path: str
    model_save_path: str
    last_checkpoint: int
    last_checkpoint_step: int


def bam_log_name(bam_path):
    """Return the basename used to identify a BAM in worker progress logs."""
    return os.path.basename(str(bam_path).strip())


def extract_model_save_path_from_command(line):
    """Extract ``--model_save_path`` from a logged shell command, if present."""
    if "Command:" not in line or "--model_save_path" not in line:
        return None

    command = line.split("Command:", 1)[1].strip()
    try:
        command_parts = shlex.split(command)
    except ValueError:
        return None

    for idx, part in enumerate(command_parts):
        if part == "--model_save_path" and idx + 1 < len(command_parts):
            return command_parts[idx + 1]
        if part.startswith("--model_save_path="):
            return part.split("=", 1)[1]

    return None


def parse_interrupted_log_checkpoint_summary(log_path):
    """Recover the model directory and last logged checkpoint from a run log.

    The model directory may come from an argument dump or the logged command.
    A ``ValueError`` is raised when either required field cannot be recovered.
    """
    model_save_path = None
    last_checkpoint = None
    last_checkpoint_step = None

    with open(log_path, 'r', errors='replace') as log_file:
        for line in log_file:
            stripped = line.strip()
            if stripped.startswith("model_save_path:"):
                value = stripped.split(":", 1)[1].strip()
                if value and value != "None":
                    model_save_path = value
            else:
                command_model_save_path = extract_model_save_path_from_command(line)
                if command_model_save_path:
                    model_save_path = command_model_save_path

            checkpoint_match = CHECKPOINT_LOG_RE.search(line)
            if checkpoint_match:
                last_checkpoint = int(checkpoint_match.group(1))
                last_checkpoint_step = int(checkpoint_match.group(2))

    if model_save_path is None:
        raise ValueError(
            f"Could not infer model_save_path from --resume_interrupted_log={log_path}. "
            "Pass --model_load_path explicitly or use a log that includes the train.py arguments."
        )
    if last_checkpoint is None:
        raise ValueError(
            f"Could not infer the last checkpoint from --resume_interrupted_log={log_path}."
        )

    return InterruptedLogCheckpointSummary(
        log_path=log_path,
        model_save_path=model_save_path,
        last_checkpoint=last_checkpoint,
        last_checkpoint_step=last_checkpoint_step,
    )


def infer_model_load_path_from_interrupted_log(log_path):
    """Resolve and verify the checkpoint path implied by an interrupted run log."""
    summary = parse_interrupted_log_checkpoint_summary(log_path)
    model_save_path = os.path.realpath(os.path.expanduser(summary.model_save_path))
    model_load_path = os.path.join(model_save_path, f"model.{summary.last_checkpoint}")
    if not os.path.exists(model_load_path):
        raise FileNotFoundError(
            f"Inferred --model_load_path={model_load_path} from --resume_interrupted_log={log_path}, "
            "but that checkpoint file does not exist. Pass --model_load_path explicitly if "
            "the checkpoint lives elsewhere."
        )
    return model_load_path


def maybe_infer_model_load_path_from_interrupted_log(args):
    """Enable resume and fill a missing checkpoint path from ``args`` in place."""
    if args.resume_interrupted_log:
        args.resume = True
        if not args.model_load_path:
            args.model_load_path = infer_model_load_path_from_interrupted_log(args.resume_interrupted_log)


def parse_interrupted_resume_log(log_path, target_checkpoint, workers_per_rank):
    """Recover each global loader worker's open BAM at a target checkpoint.

    Only BAM-open events preceding the requested checkpoint line are considered.
    Global worker IDs are reconstructed as ``rank * workers_per_rank + worker``.
    """
    worker_current_bam = {}
    rank_ids = set()
    worker_ids = set()
    open_events = 0
    completed_events = 0
    checkpoint_step = None

    with open(log_path, 'r', errors='replace') as log_file:
        for line in log_file:
            checkpoint_match = CHECKPOINT_LOG_RE.search(line)
            if checkpoint_match:
                checkpoint = int(checkpoint_match.group(1))
                if checkpoint == target_checkpoint:
                    checkpoint_step = int(checkpoint_match.group(2))
                    break

            bam_match = BAM_OPEN_LOG_RE.search(line)
            if not bam_match:
                continue

            global_rank = int(bam_match.group(1))
            worker_id = int(bam_match.group(2))
            bam_name = bam_log_name(bam_match.group(3))
            global_worker_id = global_rank * workers_per_rank + worker_id

            rank_ids.add(global_rank)
            worker_ids.add(worker_id)
            if global_worker_id in worker_current_bam:
                completed_events += 1
            worker_current_bam[global_worker_id] = bam_name
            open_events += 1

    if checkpoint_step is None:
        raise ValueError(
            f"Could not find Checkpoint:{target_checkpoint} in --resume_interrupted_log={log_path}."
        )
    if not worker_current_bam:
        raise ValueError(
            f"No data-loader BAM open lines were found before Checkpoint:{target_checkpoint} "
            f"in --resume_interrupted_log={log_path}."
        )

    return InterruptedResumePlan(
        log_path=log_path,
        target_checkpoint=target_checkpoint,
        checkpoint_step=checkpoint_step,
        resume_start_bam_by_worker=worker_current_bam,
        open_events_before_checkpoint=open_events,
        completed_file_events_before_checkpoint=completed_events,
        workers_seen=len(worker_current_bam),
        observed_world_size=max(rank_ids) + 1,
        observed_workers_per_rank=max(worker_ids) + 1,
    )


def validate_unique_bam_log_names(bam_list):
    """Reject BAM configurations whose basenames are ambiguous in progress logs."""
    paths_by_name = defaultdict(list)
    for bam_path, _ in bam_list:
        paths_by_name[bam_log_name(bam_path)].append(bam_path)

    duplicates = {name: paths for name, paths in paths_by_name.items() if len(paths) > 1}
    if not duplicates:
        return

    examples = []
    for name, paths in list(duplicates.items())[:5]:
        examples.append(f"{name}: {paths[:2]}")
    raise ValueError(
        "--resume_interrupted_log cannot disambiguate duplicate BAM basenames in --bam_config. "
        "The training log records only os.path.basename(bam_path). Examples: "
        + "; ".join(examples)
    )


def build_interrupted_resume_plan(log_path, target_checkpoint, bam_list, world_size, dataloader_workers):
    """Validate logged worker positions against the current distributed BAM assignment.

    Resuming requires identical world size, worker count, unique BAM basenames,
    and round-robin BAM ownership.
    """
    workers_per_rank = max(int(dataloader_workers), 1)
    plan = parse_interrupted_resume_log(log_path, target_checkpoint, workers_per_rank)

    if plan.observed_world_size != world_size:
        raise ValueError(
            "--resume_interrupted_log requires the same distributed world size as the interrupted run. "
            f"Log observed world_size={plan.observed_world_size}, current world_size={world_size}."
        )
    if plan.observed_workers_per_rank != workers_per_rank:
        raise ValueError(
            "--resume_interrupted_log requires the same --dataloader_workers setting as the interrupted run. "
            f"Log observed dataloader_workers={plan.observed_workers_per_rank}, "
            f"current worker slots per rank={workers_per_rank}."
        )

    validate_unique_bam_log_names(bam_list)

    global_worker_count = max(world_size * workers_per_rank, 1)
    assigned_bam_names_by_worker = {worker_id: set() for worker_id in range(global_worker_count)}
    for bam_idx, (bam_path, _) in enumerate(bam_list):
        assigned_bam_names_by_worker[bam_idx % global_worker_count].add(bam_log_name(bam_path))

    missing = []
    for global_worker_id, bam_name in plan.resume_start_bam_by_worker.items():
        assigned_bam_names = assigned_bam_names_by_worker.get(global_worker_id, set())
        if bam_name not in assigned_bam_names:
            missing.append((global_worker_id, bam_name))

    if missing:
        examples = ", ".join(
            f"worker {worker_id}: {bam_name}"
            for worker_id, bam_name in missing[:8]
        )
        raise ValueError(
            "The interrupted-resume log does not match the current --bam_config worker assignment. "
            f"Could not find checkpoint-time BAMs in their current worker lists: {examples}"
        )

    return plan


def describe_interrupted_resume_plan(plan):
    """Format a compact human-readable summary of recovered worker positions."""
    starts = sorted(plan.resume_start_bam_by_worker.items())
    examples = ", ".join(f"{worker_id}:{bam_name}" for worker_id, bam_name in starts[:8])
    if len(starts) > 8:
        examples += ", ..."
    return (
        f"Interrupted data resume from log={plan.log_path}; checkpoint={plan.target_checkpoint} "
        f"steps={plan.checkpoint_step}; observed BAM opens={plan.open_events_before_checkpoint}; "
        f"completed transitions before checkpoint={plan.completed_file_events_before_checkpoint}; "
        f"workers resumed={plan.workers_seen}; start BAMs={examples}"
    )


def setup_distributed():
    """Initialize NCCL from torchrun variables, or return single-process ranks."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank

    return 0, 1, 0

def distributed_mean(value: float, device: torch.device, world_size: int) -> float:
    """Average one scalar across ranks, avoiding collectives for one process."""
    if world_size == 1:
        return value

    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor.item()


class NoOpWriter:
    """TensorBoard-compatible sink used by non-main processes."""

    def add_scalar(self, *args, **kwargs):
        """Discard a scalar log event."""
        return None

    def flush(self):
        """Provide the writer flush interface without side effects."""
        return None

def reduce_loss_stats(total_loss: float, count: int, device: torch.device, world_size: int):
    """Sum a loss accumulator and observation count across ranks."""
    if world_size == 1:
        return total_loss, count

    tensor = torch.tensor([total_loss, count], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor[0].item(), int(tensor[1].item())


CLASSIFICATION_TASK_NUM_CLASSES = {
    'sublm': 2,
    'masklm': 4,
}


def new_sub_epoch_metrics():
    """Create an empty loss/confusion accumulator for one task window."""
    return {'loss_sum': 0.0, 'loss_count': 0, 'confusion': None}


def confusion_matrix_from_predictions(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Build a CPU ``(target, prediction)`` count matrix from valid class IDs."""
    preds = preds.reshape(-1).to(device='cpu', dtype=torch.long)
    targets = targets.reshape(-1).to(device='cpu', dtype=torch.long)

    valid = (targets >= 0) & (targets < num_classes) & (preds >= 0) & (preds < num_classes)
    if valid.any():
        encoded = targets[valid] * num_classes + preds[valid]
        counts = torch.bincount(encoded, minlength=num_classes * num_classes)
    else:
        counts = torch.zeros(num_classes * num_classes, dtype=torch.long)

    return counts.reshape(num_classes, num_classes)


def update_classification_stats(task_metrics, task: str, preds: torch.Tensor | None, targets: torch.Tensor | None):
    """Accumulate a task's batch confusion matrix when it is a classifier task."""
    num_classes = CLASSIFICATION_TASK_NUM_CLASSES.get(task)
    if num_classes is None or preds is None or targets is None:
        return

    batch_confusion = confusion_matrix_from_predictions(preds, targets, num_classes)
    if task_metrics['confusion'] is None:
        task_metrics['confusion'] = torch.zeros(num_classes, num_classes, dtype=torch.long)
    task_metrics['confusion'] += batch_confusion


def reduce_confusion_stats(confusion: torch.Tensor, device: torch.device, world_size: int) -> torch.Tensor:
    """Sum a confusion matrix across ranks and return it on CPU."""
    if world_size == 1:
        return confusion

    reduced = confusion.to(device=device, dtype=torch.long)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced.cpu()


def macro_metrics_from_confusion(confusion: torch.Tensor, labels: list[int] | None = None):
    """Compute accuracy and macro precision/recall/F1 from a confusion matrix.

    Rows are target classes and columns are predictions. When ``labels`` is
    omitted, macro averaging includes every class with actual or predicted
    support; otherwise it includes exactly the supplied class indices. The
    return tuple also contains per-class precision, recall, and F1 lists.
    """
    confusion = confusion.to(dtype=torch.float64)
    total = confusion.sum().item()
    accuracy = confusion.diag().sum().item() / total if total > 0 else 0.0

    tp = confusion.diag()
    predicted = confusion.sum(dim=0)
    actual = confusion.sum(dim=1)

    precision_by_class = torch.where(predicted > 0, tp / predicted.clamp_min(1), torch.zeros_like(tp))
    recall_by_class = torch.where(actual > 0, tp / actual.clamp_min(1), torch.zeros_like(tp))
    denom = precision_by_class + recall_by_class
    f1_by_class = torch.where(denom > 0, 2 * precision_by_class * recall_by_class / denom, torch.zeros_like(denom))

    if labels is None:
        included = (actual > 0) | (predicted > 0)
    else:
        included = torch.zeros(confusion.shape[0], dtype=torch.bool)
        included[labels] = True

    if included.any():
        precision = precision_by_class[included].mean().item()
        recall = recall_by_class[included].mean().item()
        f1 = f1_by_class[included].mean().item()
    else:
        precision = recall = f1 = 0.0

    return accuracy, precision, recall, f1, precision_by_class.tolist(), recall_by_class.tolist(), f1_by_class.tolist()


def log_pretraining_metrics(
    sub_epoch_metrics,
    print_task_list,
    checkpoint,
    writer,
    device,
    world_size,
    is_main_process,
    log_writer=None,
):
    """Reduce one metric window across ranks and log it from rank zero."""
    dna_bases = ['A', 'C', 'G', 'T']
    for task in print_task_list:
        task_metrics = sub_epoch_metrics[task]
        total_loss, total_count = reduce_loss_stats(
            task_metrics['loss_sum'],
            task_metrics['loss_count'],
            device,
            world_size,
        )
        if total_count == 0:
            continue
        avg_loss = total_loss / total_count

        if is_main_process:
            writer.add_scalar(f'Loss/{task}', avg_loss, checkpoint)

        num_classes = CLASSIFICATION_TASK_NUM_CLASSES.get(task)
        confusion = None
        if num_classes is not None:
            confusion = task_metrics['confusion']
            if confusion is None:
                confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
            confusion = reduce_confusion_stats(confusion, device, world_size)

        if confusion is not None and confusion.sum().item() > 0:
            accuracy, precision, recall, f1, p_class, r_class, f1_class = macro_metrics_from_confusion(confusion)
            if not is_main_process:
                continue

            writer.add_scalar(f'Accuracy/{task}', accuracy, checkpoint)
            print_str = f"  {task.upper():<7} Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}"

            if task in ['sublm', 'match']:
                print_str += f" | Macro P: {precision:.4f} | Macro R: {recall:.4f} | Macro F1: {f1:.4f}"
                writer.add_scalar(f'Precision/{task}_macro', precision, checkpoint)
                writer.add_scalar(f'Recall/{task}_macro', recall, checkpoint)
                writer.add_scalar(f'F1-Score/{task}_macro', f1, checkpoint)
                if log_writer:
                    log_writer.writerow(
                        [checkpoint, task, avg_loss, accuracy, precision, recall, f1]
                        + ['N/A'] * 12
                    )

            elif task == 'masklm':
                labels_of_interest = [0, 1, 2, 3]
                _, macro_p, macro_r, macro_f1, p_class, r_class, f1_class = macro_metrics_from_confusion(
                    confusion,
                    labels=labels_of_interest,
                )
                print_str += (
                    f" | Macro P: {macro_p:.4f} | Macro R: {macro_r:.4f} "
                    f"| Macro F1: {macro_f1:.4f}"
                )
                writer.add_scalar('Precision/masklm_macro', macro_p, checkpoint)
                writer.add_scalar('Recall/masklm_macro', macro_r, checkpoint)
                writer.add_scalar('F1-Score/masklm_macro', macro_f1, checkpoint)

                log_row_extra = []
                for i, base in enumerate(dna_bases):
                    print_str += f" | {base} F1: {f1_class[i]:.4f}"
                    writer.add_scalar(f'F1-Score/masklm_{base}', f1_class[i], checkpoint)
                    writer.add_scalar(f'Precision/masklm_{base}', p_class[i], checkpoint)
                    writer.add_scalar(f'Recall/masklm_{base}', r_class[i], checkpoint)
                    log_row_extra.extend([p_class[i], r_class[i], f1_class[i]])

                if log_writer:
                    log_writer.writerow(
                        [checkpoint, task, avg_loss, accuracy, macro_p, macro_r, macro_f1]
                        + log_row_extra
                    )

            print(print_str, flush=True)
        elif is_main_process:
            print(f"  {task.upper():<7} Loss: {avg_loss:.4f}", flush=True)
            if log_writer:
                log_writer.writerow([checkpoint, task, avg_loss] + ['N/A'] * 16)


def get_global_counts(local_instances, local_base_tokens, local_signal_tokens, device):
    """Sum instance, base-token, and signal-token counts across all ranks."""
    if not dist.is_initialized():
        return local_instances, local_base_tokens, local_signal_tokens
        
    tensor = torch.tensor([local_instances, local_base_tokens, local_signal_tokens], device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    return tensor[0].item(), tensor[1].item(), tensor[2].item()

def training_log_header():
    """Return CSV columns for aggregate and per-base pretraining metrics."""
    header = ['checkpoint', 'task', 'avg_loss', 'accuracy', 'precision', 'recall', 'f1']
    for base in ['A', 'C', 'G', 'T']:
        header.extend([f'precision_{base}', f'recall_{base}', f'f1_{base}'])
    return header

def apply_scheduler_lrs(scheduler):
    """Copy a scheduler's cached learning rates into its optimizer groups."""
    if not hasattr(scheduler, "_last_lr"):
        return
    for param_group, lr in zip(scheduler.optimizer.param_groups, scheduler._last_lr):
        param_group["lr"] = lr

def fast_forward_scheduler(scheduler, step):
    """Set a LambdaLR-style scheduler to ``step`` without replaying prior steps."""
    scheduler.last_epoch = int(step)
    scheduler._step_count = max(getattr(scheduler, "_step_count", 0), int(step) + 1)
    new_lrs = [
        base_lr * lr_lambda(scheduler.last_epoch)
        for lr_lambda, base_lr in zip(scheduler.lr_lambdas, scheduler.base_lrs)
    ]
    for param_group, lr in zip(scheduler.optimizer.param_groups, new_lrs):
        param_group["lr"] = lr
    scheduler._last_lr = new_lrs


def resolve_signal_dropout_policy(args, loaded_config=None):
    """Resolve configured and active modality-dropout policy for one run.

    Loaded policy is inherited only for ``--use_loaded_config`` or true resume.
    Resume rejects policy changes, while a non-resume warm start may override
    the loaded values. Evaluation retains recorded configuration but always
    executes with zero signal-modality dropout.
    """
    inherit_loaded = loaded_config is not None and (args.use_loaded_config or args.resume)

    if inherit_loaded:
        saved_probability = float(loaded_config["signal_dropout_prob"])
        saved_mode = loaded_config["ema_signal_dropout_mode"]
        if args.resume and args.signal_dropout_prob is not None and args.signal_dropout_prob != saved_probability:
            raise ValueError(
                "--resume cannot change signal_dropout_prob; omit the option or use "
                f"the saved value {saved_probability}."
            )
        if args.resume and args.ema_signal_dropout_mode is not None and args.ema_signal_dropout_mode != saved_mode:
            raise ValueError(
                "--resume cannot change ema_signal_dropout_mode; omit the option or use "
                f"the saved value {saved_mode!r}."
            )

        configured_probability = (
            saved_probability
            if args.signal_dropout_prob is None
            else float(args.signal_dropout_prob)
        )
        configured_mode = args.ema_signal_dropout_mode or saved_mode
        probability_source = "checkpoint" if args.signal_dropout_prob is None else "CLI override"
        mode_source = "checkpoint" if args.ema_signal_dropout_mode is None else "CLI override"
    else:
        configured_probability = 0.0 if args.signal_dropout_prob is None else float(args.signal_dropout_prob)
        configured_mode = args.ema_signal_dropout_mode or "disabled"
        probability_source = "new-run default" if args.signal_dropout_prob is None else "CLI"
        mode_source = "new-run default" if args.ema_signal_dropout_mode is None else "CLI"

    active_probability = 0.0 if args.eval_only else configured_probability
    return {
        "configured_probability": configured_probability,
        "active_probability": active_probability,
        "ema_mode": configured_mode,
        "probability_source": probability_source,
        "mode_source": mode_source,
    }

def train_model(args, rank, world_size, local_rank, is_main_process):
    """Run pretraining/evaluation for one distributed rank from parsed arguments.

    The routine constructs BAM iterable datasets, model, optimizer, curriculum,
    scheduler, checkpoint/log writers, and optional interrupted-data resume
    state. Rank zero owns durable output and metric reporting. Training proceeds
    until ``max_steps``; evaluation performs a finite pass without updates or
    checkpoint writes.
    """
    maybe_infer_model_load_path_from_interrupted_log(args)
    resolve_cli_preprocessing(args)

    # Evaluation also writes TensorBoard/CSV logs, so it needs a fresh run.
    prepare_run_outputs(args, command="pretrain", directories={
        "model_save_path": args.model_save_path,
    }, distributed=dist)

    eval_only=args.eval_only
    model_save_path, batch_size = args.model_save_path, args.batch_size
    
    max_steps=args.max_steps
    weight_decay=args.weight_decay
    checkpoint_batch_step, lr, max_ts_len = args.checkpoint_batch_step, args.lr, args.size
    
    hidden_dim, num_heads, dim_feedforward = args.hidden_dim, args.num_heads, args.dim_feedforward
    
    num_layers_base_seq_self, num_layers_ts_self, num_layers_cross= args.num_layers_base_seq_self, args.num_layers_ts_self, args.num_layers_cross
    
    base_seq_window, ts_window = args.base_seq_window, args.ts_window
    base_seq_window, ts_window = (base_seq_window, base_seq_window), (ts_window, ts_window)
    add_percentile = args.add_percentile
    
    num_base_trim, num_signal_trim = args.num_loss_trim
    
    kv_num_heads = args.kv_num_heads
    
    if kv_num_heads==0:
        kv_num_heads=num_heads
        
    task_list, weights = args.task_list, args.weights
    
    mask_prob_range, mask_prob_warmup = args.mask_prob_range, args.mask_prob_warmup
    mask_base_span_range, mask_signal_span= args.mask_base_span_range, args.mask_signal_span

    mask_prob_start, mask_prob_end = mask_prob_range
    mask_span_start, mask_span_end = mask_base_span_range
    
    assert mask_prob_start<=mask_prob_end, 'start should <= end'
    assert mask_span_start<=mask_span_end, 'start should <= end'
    
    print_task_list = [x for x in task_list]
    
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    
    # --- Setup (before the training loop) ---
    # 1. TensorBoard Writer
    if is_main_process:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(model_save_path, 'runs'))
    else:
        writer = NoOpWriter()
    loaded_checkpoint = None
    loaded_config_dict = None
    if args.model_load_path:
        loaded_checkpoint = torch.load(args.model_load_path, map_location="cpu")
        loaded_config_dict = get_backbone_config_from_checkpoint(
            loaded_checkpoint,
            source=args.model_load_path,
        )

    signal_dropout_policy = resolve_signal_dropout_policy(args, loaded_config_dict)
    configured_signal_dropout_prob = signal_dropout_policy["configured_probability"]
    active_signal_dropout_prob = signal_dropout_policy["active_probability"]
    ema_signal_dropout_mode = signal_dropout_policy["ema_mode"]
    if is_main_process:
        print(
            "Signal modality dropout: "
            f"configured={configured_signal_dropout_prob:g} "
            f"active={active_signal_dropout_prob:g} "
            f"probability_source={signal_dropout_policy['probability_source']} "
            f"ema_mode={ema_signal_dropout_mode} "
            f"mode_source={signal_dropout_policy['mode_source']}",
            flush=True,
        )

    resume_batch_num = 0
    resume_checkpoint = 0
    if args.resume:
        if loaded_checkpoint is None:
            raise ValueError("--resume requires --model_load_path.")
        if eval_only:
            raise ValueError("--resume is only valid for training, not --eval_only.")
        if "global_step" not in loaded_checkpoint:
            raise KeyError("--resume requires checkpoint['global_step'].")
        resume_batch_num = int(loaded_checkpoint["global_step"])
        resume_checkpoint = resume_batch_num // checkpoint_batch_step
        if resume_batch_num % checkpoint_batch_step != 0 and is_main_process:
            print(
                f"Warning: checkpoint global_step={resume_batch_num} is not aligned to "
                f"checkpoint_batch_step={checkpoint_batch_step}; next checkpoint will be "
                f"{resume_batch_num // checkpoint_batch_step + 1}.",
                flush=True,
            )

    # 2. CSV Logger (only on rank 0)
    if is_main_process:
        log_file_path = os.path.join(model_save_path, 'training_log.csv')
        log_file_exists = os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0
        log_file = open(log_file_path, 'a' if args.resume else 'w', newline='')
        log_writer = csv.writer(log_file)
        dna_bases = ['A', 'C', 'G', 'T']
        if not args.resume or not log_file_exists:
            log_writer.writerow(training_log_header())
    else:
        log_file = None
        log_writer = None
        dna_bases = ['A', 'C', 'G', 'T']
    if args.model_load_path and (args.use_loaded_config or args.resume):
        print(f'Loading config from:{args.model_load_path}', flush=True)
        config_dict = dict(loaded_config_dict)
        config_dict['signal_dropout_prob'] = configured_signal_dropout_prob
        config_dict['ema_signal_dropout_mode'] = ema_signal_dropout_mode
        config=ONTBackboneConfig(**config_dict)
        ts_dim=config.ts_dim
        
        signal_kernels=config.signal_kernels
        signal_strides = config.signal_strides
        signal_stride_product = math.prod(signal_strides)
        if signal_stride_product % ts_dim != 0:
            raise ValueError(
                f"Loaded config has signal_strides product {signal_stride_product}, "
                f"which is not divisible by ts_dim={ts_dim}."
            )
        expected_move_downsample_factor = signal_stride_product // ts_dim
        if config.move_downsample_factor != expected_move_downsample_factor:
            raise ValueError(
                f"Loaded config move_downsample_factor={config.move_downsample_factor} does not match "
                f"prod(signal_strides)//ts_dim={expected_move_downsample_factor}."
            )
        move_downsample_factor=config.move_downsample_factor
        max_base_seq_len=max_ts_len*move_downsample_factor
        add_signal_cross_attn=config.add_signal_cross_attn
        
    else:
        ts_dim=args.ts_dim
        signal_kernels=args.signal_kernels
        signal_strides = args.signal_strides
        signal_stride_product = math.prod(signal_strides)
        if signal_stride_product % ts_dim != 0:
            raise ValueError(
                f"signal_strides product {signal_stride_product} must be divisible by ts_dim={ts_dim}."
            )
        move_downsample_factor=signal_stride_product//ts_dim
        max_base_seq_len=max_ts_len*move_downsample_factor
        add_signal_cross_attn=args.add_signal_cross_attn
        
        config = ONTBackboneConfig(
            base_seq_vocab=6, 
            ts_dim=ts_dim, 
            signal_kernels=signal_kernels,
            signal_strides=signal_strides,
            move_downsample_factor=move_downsample_factor,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers_base_seq_self=num_layers_base_seq_self,
            num_layers_ts_self=num_layers_ts_self, 
            num_layers_base_cross=num_layers_cross, 
            num_layers_signal_cross = max(1, num_layers_cross//2) if add_signal_cross_attn else 0, 
            max_base_seq_len = max_base_seq_len,
            add_signal_cross_attn=add_signal_cross_attn,
            kv_num_heads=kv_num_heads, 
            add_percentile=add_percentile,
            max_ts_len = max_ts_len,
            dim_feedforward=dim_feedforward,
            base_seq_window=base_seq_window,
            ts_window=ts_window,
            dropout=0.0,
            signal_dropout_prob=configured_signal_dropout_prob,
            ema_signal_dropout_mode=ema_signal_dropout_mode,
            self_attn_temp=args.self_attn_temp,
            cross_attn_temp=args.cross_attn_temp,
            expected_ratio=2 if args.seq_type=='dna' else 5,
            residual_control=args.residual_control,
            pre_attention_mixer_type=args.pre_attention_mixer_type
        )

    model = ONTModelForPreTraining(config)
    
    if args.model_load_path:
        print(f'Loading model weights from:{args.model_load_path}', flush=True)
        state_dict=get_pretraining_state_dict_from_checkpoint(loaded_checkpoint)
        state_dict = strip_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=True)

    model.preprocessing = resolve_cli_preprocessing(args)
    kmer_table = model.preprocessing.kmer_table
    
    meta_dict=dict(task_list = task_list, weights = weights, lr = lr,
                   size = args.size, step = args.step,
                   min_read_length = args.min_read_length, min_qscore = args.min_qscore, 
                   lm_label_smoothing=args.lm_label_smoothing,
                   signal_dropout_prob=configured_signal_dropout_prob,
                   ema_signal_dropout_mode=ema_signal_dropout_mode)
    if args.use_polished_sequence:
        meta_dict.update(
            use_polished_sequence=True,
            polish_indel_limit=args.polish_indel_limit,
            polish_vcf_path=args.polish_vcf_path,
        )
    
    model=model.to(device)
    model.eval()
    
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16), torch.no_grad():
        model(torch.ones(1, max_base_seq_len).long().to(device),
              torch.randn(1, max_ts_len, ts_dim*move_downsample_factor).to(device),
              percentiles_input=torch.randn(1, max_ts_len, ts_dim*move_downsample_factor).to(device),
              pad_base_seq=torch.ones(1, max_base_seq_len).to(torch.bool).to(device),
              pad_ts=torch.ones(1, max_ts_len).to(torch.bool).to(device),
             );

    if args.disable_compile==False:
        model = torch.compile(model, dynamic=True)
        
    # Evaluation ranks may exhaust finite BAM shards at different times. They
    # do not need gradient synchronization, so keep independent model replicas
    # and reduce their bounded metric accumulators once after all ranks finish.
    if world_size > 1 and not eval_only:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    model_to_print = model.module if isinstance(model, DDP) else model
    model_details=str(model_to_print)
    if is_main_process:
        print(model_details, flush=True)
        num_params=sum(p.numel() for p in model_to_print.parameters())
        print('# Parameters=', num_params, flush=True)

    
    if args.auto_batch_size and not eval_only and device.type == "cuda":
        per_sample_mem = estimate_per_sample_memory(
            model,
            device,
            max_base_seq_len,
            max_ts_len,
            ts_dim,
            add_percentile,
            mask_base_seq_for_mlm,
        )

        if per_sample_mem > 0:
            free_mem, _ = torch.cuda.mem_get_info(device)
            max_local_batch = max(int((free_mem * args.auto_batch_size_safety) // per_sample_mem), 1)
            if args.max_auto_batch_size > 0:
                max_local_batch = min(max_local_batch, args.max_auto_batch_size)

            max_local_batch = align_to_power_of_two(max_local_batch)

            max_batch_tensor = torch.tensor([max_local_batch], device=device, dtype=torch.int64)
            if world_size > 1:
                dist.all_reduce(max_batch_tensor, op=dist.ReduceOp.MIN)
            adjusted_batch_size = int(max_batch_tensor.item())

            if is_main_process:
                print(
                    f"Auto-tuned batch size from {args.batch_size} to {adjusted_batch_size} using max lengths (power-of-two aligned)",
                    flush=True
                )
            args.batch_size = max(1, adjusted_batch_size)
        elif is_main_process:
            print("Auto batch size estimation skipped (no CUDA memory info).", flush=True)
    
    t=time.time()
    
    if not eval_only:
        optimizer = torch.optim.AdamW(make_param_groups(model, weight_decay=weight_decay), lr=lr)
        if args.model_load_path and (args.load_optimizer or args.resume):
            if "optimizer_state_dict" not in loaded_checkpoint:
                raise KeyError("Loading optimizer state requires checkpoint['optimizer_state_dict'].")
            optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
            if not args.resume:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                
        scheduler = get_warmup_cosine_schedule(optimizer, args.num_warmp_cp*checkpoint_batch_step, args.num_peak_cp*checkpoint_batch_step, max_steps, start_factor=0.01, min_factor=0.05)
        if args.resume:
            if "scheduler_state_dict" in loaded_checkpoint:
                scheduler.load_state_dict(loaded_checkpoint["scheduler_state_dict"])
                apply_scheduler_lrs(scheduler)
            else:
                if is_main_process:
                    print(
                        "Warning: checkpoint has no scheduler_state_dict; "
                        f"fast-forwarding scheduler to global_step={resume_batch_num}.",
                        flush=True,
                    )
                fast_forward_scheduler(scheduler, resume_batch_num)
        
        model.train()
    else:
        optimizer=None
        scheduler=None
    
    sub_epoch_metrics = defaultdict(new_sub_epoch_metrics)

    batch_num, checkpoint = resume_batch_num, resume_checkpoint
    if args.resume and is_main_process:
        current_lrs = scheduler.get_last_lr() if scheduler is not None else []
        print(
            f"Resuming from global_step={batch_num}, checkpoint={checkpoint}; "
            f"next checkpoint={checkpoint + 1}; lr={current_lrs}",
            flush=True,
        )
            
    bam_list = load_pretraining_bam_list(args.bam_config)
    validate_ddp_bam_sharding(
        len(bam_list),
        world_size,
        args.dataloader_workers,
        "pretraining BAM config",
    )
    resume_start_bam_by_worker = None
    if args.resume_interrupted_log:
        interrupted_resume_plan = build_interrupted_resume_plan(
            args.resume_interrupted_log,
            resume_checkpoint,
            bam_list,
            world_size,
            args.dataloader_workers,
        )
        resume_start_bam_by_worker = interrupted_resume_plan.resume_start_bam_by_worker
        if is_main_process:
            print(describe_interrupted_resume_plan(interrupted_resume_plan), flush=True)

    ds = BamIterableDataset(
        bam_list=bam_list,
        size=args.size,
        move_downsample_factor=move_downsample_factor,
        step=args.step,
        seq_type=args.seq_type,
        chunking_mode=resolve_chunking_mode(args), kmer_table=kmer_table,
        reads_per_chunk=args.reads_per_chunk,
        min_read_length=args.min_read_length,
        min_qscore=args.min_qscore,
        pore_type=args.pore_type,
        base_seed=args.seed,
        rank=rank,             
        world_size=world_size, 
        expected_ts_dim=ts_dim,
        resume_start_bam_by_worker=resume_start_bam_by_worker,
        use_polished_sequence=args.use_polished_sequence,
        polish_indel_limit=args.polish_indel_limit,
        polish_vcf_path=args.polish_vcf_path,
        cycle=not eval_only,
    )

    dataloader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        num_workers=args.dataloader_workers,
        pin_memory=True,
        drop_last=True,
    )

    if args.dataloader_workers > 0:
        dataloader_kwargs.update(
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
        )

    dataloader = DataLoader(ds, **dataloader_kwargs)
    
    total_instances_seen = 0
    total_base_tokens_seen = 0
    total_signal_tokens_seen = 0
    eval_stopped_at_max_steps = False
    
    for batch in dataloader:    
        #base_seq_input, ts_input, percentiles_input, pad_base_seq, pad_ts = batch
        
        current_mask_prob=mask_prob_start+(mask_prob_end-mask_prob_start)*min(checkpoint, mask_prob_warmup)/mask_prob_warmup
        task, current_mask_span = choose_pretraining_step_controls(
            task_list,
            weights,
            mask_span_start,
            mask_span_end,
            device,
            rank,
            1 if eval_only else world_size,
        )
        loss, preds, targets = training_step(model=model, 
                                             batch=batch,
                                             optimizer=optimizer,
                                             scheduler=scheduler,
                                             device=device,
                                             task=task,
                                             mask_prob=current_mask_prob,
                                             mask_base_span=current_mask_span,
                                             mask_signal_span=mask_signal_span,
                                             add_percentile=add_percentile,
                                             eval_only=eval_only,
                                             add_signal_cross_attn=add_signal_cross_attn,
                                             num_base_trim=num_base_trim,
                                             num_signal_trim=num_signal_trim,
                                             lm_label_smoothing=args.lm_label_smoothing,
                                             include_dummy_loss=False,
                                             signal_dropout_prob=active_signal_dropout_prob)
        
        
        # Store results for this step
        
        task_metrics = sub_epoch_metrics[task]
        task_metrics['loss_sum'] += loss
        task_metrics['loss_count'] += 1
        update_classification_stats(task_metrics, task, preds, targets)

        
        local_batch_size = len(batch[0]) # counting number of sequences
        local_base_token_count = sum(len(seq) for seq in batch[0]) # counting number of bases in all sequences
        local_signal_token_count = torch.sum(batch[4]).item()
        
        total_instances_seen += local_batch_size
        total_base_tokens_seen += local_base_token_count
        total_signal_tokens_seen += local_signal_token_count
        
        batch_num += 1

        if eval_only:
            if is_main_process and batch_num % checkpoint_batch_step == 0:
                print(
                    f"{str(datetime.datetime.now())}: Evaluation progress: "
                    f"local rank-zero batches={batch_num} Time={time.time()-t:.2f}",
                    flush=True,
                )
            if batch_num >= max_steps:
                eval_stopped_at_max_steps = True
                break
            continue

        # Training ranks remain in lockstep, so checkpoint statistics can be
        # reduced at fixed update intervals.
        if batch_num % checkpoint_batch_step == 0:
            checkpoint = batch_num // checkpoint_batch_step
            global_instances, global_base_tokens, global_signal_tokens = get_global_counts(
                total_instances_seen,
                total_base_tokens_seen,
                total_signal_tokens_seen,
                device,
            )

            if is_main_process:
                print(f"{str(datetime.datetime.now())}: Global Instances: {global_instances:,} | Global Base Tokens: {global_base_tokens:,} | Global Signal Tokens: {global_signal_tokens:,}", flush=True)
                print(f"{str(datetime.datetime.now())}: Checkpoint:{checkpoint}  Steps:{batch_num}  Time={time.time()-t:.2f}   Mask Prob={current_mask_prob:.4f}", flush=True)

            log_pretraining_metrics(
                sub_epoch_metrics,
                print_task_list,
                checkpoint,
                writer,
                device,
                world_size,
                is_main_process,
                log_writer,
            )

            if is_main_process:
                print('______________________________________________________________', flush=True)

            sub_epoch_metrics = defaultdict(new_sub_epoch_metrics)

            if is_main_process:
                model_to_save = get_unwrapped_model(model)
                model_path = os.path.join(model_save_path, 'model.%d' % (checkpoint))
                ckpt = {
                        "checkpoint_type": "ont_pretraining_v2",
                        "model_class": "ONTModelForPreTraining",
                        'optimizer_state_dict': optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": asdict(model_to_save.config),
                        "meta": meta_dict,
                        "global_step": batch_num,
                    }
                ckpt.update(modular_pretraining_payload(model_to_save))
                attach_preprocessing(ckpt, args.preprocessing)
                torch.save(ckpt, model_path)

                # Flush CSV writer to save to disk immediately
                if log_file:
                    log_file.flush()

            if batch_num >= max_steps:
                break

    if eval_only:
        global_instances, global_base_tokens, global_signal_tokens = get_global_counts(
            total_instances_seen,
            total_base_tokens_seen,
            total_signal_tokens_seen,
            device,
        )
        max_batches = torch.tensor([batch_num], dtype=torch.long, device=device)
        if world_size > 1:
            dist.all_reduce(max_batches, op=dist.ReduceOp.MAX)
        checkpoint = max(1, math.ceil(int(max_batches.item()) / checkpoint_batch_step))

        if is_main_process:
            completion_reason = (
                "the max_steps limit"
                if eval_stopped_at_max_steps
                else "one finite BAM pass"
            )
            print(
                f"{str(datetime.datetime.now())}: Evaluation complete after {completion_reason}. "
                f"Global Instances: {global_instances:,} | Global Base Tokens: {global_base_tokens:,} "
                f"| Global Signal Tokens: {global_signal_tokens:,}",
                flush=True,
            )

        log_pretraining_metrics(
            sub_epoch_metrics,
            print_task_list,
            checkpoint,
            writer,
            device,
            world_size,
            is_main_process,
            log_writer,
        )
        if is_main_process:
            print('______________________________________________________________', flush=True)
            writer.flush()
            if log_file:
                log_file.flush()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    return


if __name__=='__main__':
    
    print('\n%s:\nCommand: python %s\n' %(str(datetime.datetime.now()), ' '.join(sys.argv)), flush=True)
    
    parser = build_arg_parser()
    args = parser.parse_args()
    maybe_infer_model_load_path_from_interrupted_log(args)
    
    if len(args.task_list) != len(args.weights):
        raise ValueError("The number of tasks in --task_list must match the number of weights in --weights.")
    if not 0 <= args.lm_label_smoothing < 1:
        raise ValueError("--lm_label_smoothing must be in the range [0, 1).")
    if args.resume and not args.model_load_path:
        raise ValueError("--resume requires --model_load_path.")
    if args.resume and args.eval_only:
        raise ValueError("--resume is only valid for training, not --eval_only.")
    (
        args.use_polished_sequence,
        args.polish_indel_limit,
        args.polish_vcf_path,
    ) = validate_polishing_options(
        args.use_polished_sequence,
        args.polish_indel_limit,
        args.polish_vcf_path,
    )
    
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0
    
    if is_main_process:
        print(f"{str(datetime.datetime.now())}: Starting Training", flush=True)

        print('Arguments::')
        for k in vars(args):
            print('{}: {}'.format(k,vars(args)[k]))

    
    train_model(args, rank, world_size, local_rank, is_main_process)
        
    print(f"{str(datetime.datetime.now())}: Training Finished.", flush=True)
