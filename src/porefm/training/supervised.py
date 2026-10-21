"""Supervised per-base fine-tuning with classification or exclusion labels."""

from porefm.preprocessing import attach_preprocessing, preprocessing_from_checkpoint
from porefm.cli.preprocessing_args import resolve_cli_preprocessing
from porefm.cli.chunking_args import resolve_chunking_mode
import sys

from porefm.cli.supervised_args import build_supervised_finetune_parser


if __name__ == '__main__' and any(arg in ('-h', '--help') for arg in sys.argv[1:]):
    build_supervised_finetune_parser().parse_args()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, IterableDataset
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import random, time, os, datetime, csv, math
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

from dataclasses import dataclass, asdict

from contextlib import nullcontext

import torch.distributed as dist

from porefm.common.chunking import resolve_downstream_step
from porefm.common.modification_metadata import resolve_modification_metadata
from porefm.common.utils import align_to_power_of_two, rewarmup_then_constant, get_dummy_loss, focal_loss_fn
from porefm.common.tensor_masks import ignore_edges
from porefm.config.validation import load_supervised_config

torch.multiprocessing.set_start_method('spawn', force=True)
torch.set_num_threads(1)

from porefm.data.bam import build_lookup_table
from porefm.data.reader import (
    BamIterableDataset,
    LabelledBamSample,
    SupervisedWeightedLabelledBamIterableDataset,
    TaggedBamSample,
    labelled_collate_fn,
    validate_ddp_bam_sharding,
)
from porefm.models.backbone.model import ONTBackboneConfig
from porefm.models.backbone.utils import strip_module_prefix
from porefm.models.classifier import (
    ONTModelForClassification,
    checkpoint_has_signal_cross_attn_weights,
)
from porefm.models.checkpointing import (
    get_pretraining_state_dict_from_checkpoint,
    get_classification_state_dict_from_checkpoint,
    modular_classification_payload,
)
from porefm.models.configuration import get_backbone_config_from_checkpoint
from porefm.models.lora import (
    LORA_TARGET_HELP,
    add_lora_to_pretrain_backbone,
    get_lora_config,
    maybe_restore_lora_args_from_checkpoint,
    set_pretrain_lora_trainability,
)
import porefm.training.pretrain as pretrain_train
from porefm.data.complete_reads import CompleteLabelledReadIterableDataset

from concurrent.futures import ThreadPoolExecutor

torch.set_float32_matmul_precision('high')
torch._dynamo.config.capture_scalar_outputs = True

@dataclass(frozen=True)
class SampleLabelSmoothing:
    """Resolved loss epsilon for one JSON sample, before BAM expansion."""

    sample_index: int
    bam_path: str
    value: float
    source: str


def resolve_sample_label_smoothing(config, label_smoothing=0.0):
    """Resolve validated JSON sample > config > CLI smoothing policies.

    ``config`` is normalized by ``load_supervised_config``. The returned tuple
    follows its sample-list order; directory expansions inherit the same
    policy. Exclusion always resolves to zero, ignoring the CLI fallback.
    ``source`` records ``sample``, ``config``, ``cli``, or ``exclusion``.
    """
    if (isinstance(label_smoothing, bool) or not isinstance(label_smoothing, (int, float))
            or not math.isfinite(label_smoothing) or not 0 <= label_smoothing < 1):
        raise ValueError("--label_smoothing must be a finite number in [0, 1).")
    samples = config.get('labelled_samples', config.get('tagged_samples', []))
    is_exclusion = config['supervision']['type'] == 'exclusion'
    resolved = []
    for index, sample in enumerate(samples):
        if is_exclusion:
            value, source = 0.0, 'exclusion'
        elif 'label_smoothing' in sample:
            value, source = sample['label_smoothing'], 'sample'
        elif 'label_smoothing' in config:
            value, source = config['label_smoothing'], 'config'
        else:
            value, source = label_smoothing, 'cli'
        resolved.append(SampleLabelSmoothing(index, sample['bam_path'], float(value), source))
    return tuple(resolved)


@dataclass(frozen=True)
class SupervisionSpec:
    """Validated JSON-wide supervision policy for one source-config group.

    Classification specs identify the config and resolved sample smoothing.
    Exclusion specs additionally identify the excluded class, its complementary
    allowed indices, and optional expected allowed-class proportions with a
    tolerance. Proportions follow ``allowed_class_indices`` order.
    """
    config_path: str
    supervision_type: str
    excluded_class_index: Optional[int] = None
    excluded_class_name: Optional[str] = None
    allowed_class_indices: tuple[int, ...] = ()
    expected_allowed_proportions: Optional[tuple[float, ...]] = None
    proportion_tolerance: Optional[float] = None
    sample_label_smoothing: tuple[SampleLabelSmoothing, ...] = ()


def load_and_validate_supervision_spec(
    json_config_path: str,
    class_labels: List[str],
    label_smoothing: float = 0.0,
) -> SupervisionSpec:
    """Load and validate the supervision policy for one BAM JSON config.

    The config must contain exactly one non-empty ``labelled_samples`` or
    ``tagged_samples`` list. Missing supervision defaults to classification.
    Exclusion is supported only for labelled samples, requires a named class,
    prohibits nonzero JSON smoothing, and requires supplied motif labels to equal the
    excluded class index. Optional expected proportions must cover every
    allowed class exactly, sum to one, and include a tolerance in ``[0, 1)``.

    Parameters
    ----------
    json_config_path : str
        JSON path recorded in the returned immutable specification.
    class_labels : list of str
        Unique class names in classifier-index order.
    label_smoothing : float, default=0
        CLI fallback for samples without sample/config overrides.

    Returns
    -------
    SupervisionSpec
        Normalized group-wide policy used to route retained bases during loss
        computation.
    """
    config = load_supervised_config(json_config_path, class_labels=class_labels)
    sample_smoothing = resolve_sample_label_smoothing(config, label_smoothing)

    if len(class_labels) == 0 or len(set(class_labels)) != len(class_labels):
        raise ValueError("--class_labels must contain unique class names before parsing supervision configs.")

    has_labelled = 'labelled_samples' in config
    has_tagged = 'tagged_samples' in config
    if has_labelled == has_tagged:
        raise ValueError(
            f"{json_config_path}: define exactly one of 'labelled_samples' or 'tagged_samples'."
        )

    sample_key = 'labelled_samples' if has_labelled else 'tagged_samples'
    samples = config[sample_key]
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError(f"{json_config_path}: '{sample_key}' must be a nonempty list.")
    for sample_idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise TypeError(f"{json_config_path}: {sample_key}[{sample_idx}] must be a JSON object.")
        if 'supervision' in sample:
            raise ValueError(
                f"{json_config_path}: supervision is JSON-wide and cannot appear in {sample_key}[{sample_idx}]."
            )

    raw_spec = config.get('supervision', {'type': 'classification'})
    if not isinstance(raw_spec, dict):
        raise TypeError(f"{json_config_path}: 'supervision' must be a JSON object.")

    supervision_type = raw_spec.get('type', 'classification')
    if supervision_type == 'classification':
        unknown = set(raw_spec) - {'type'}
        if unknown:
            raise ValueError(
                f"{json_config_path}: classification supervision has unsupported fields: {sorted(unknown)}"
            )
        return SupervisionSpec(
            config_path=str(Path(json_config_path)),
            supervision_type='classification',
            sample_label_smoothing=sample_smoothing,
        )

    if supervision_type != 'exclusion':
        raise ValueError(
            f"{json_config_path}: supervision.type must be 'classification' or 'exclusion', "
            f"got {supervision_type!r}."
        )
    if not has_labelled:
        raise ValueError(f"{json_config_path}: exclusion supervision currently requires 'labelled_samples'.")

    allowed_fields = {
        'type',
        'excluded_class',
        'expected_allowed_proportions',
        'proportion_tolerance',
    }
    unknown = set(raw_spec) - allowed_fields
    if unknown:
        raise ValueError(f"{json_config_path}: exclusion supervision has unsupported fields: {sorted(unknown)}")

    excluded_class_name = raw_spec.get('excluded_class')
    if excluded_class_name not in class_labels:
        raise ValueError(
            f"{json_config_path}: excluded_class must be one of {class_labels}, got {excluded_class_name!r}."
        )
    excluded_class_index = class_labels.index(excluded_class_name)
    allowed_class_indices = tuple(i for i in range(len(class_labels)) if i != excluded_class_index)

    for sample_idx, sample in enumerate(samples):
        if sample.get('motif') is not None:
            try:
                motif_labels = [int(value.strip()) for value in str(sample['motif_label']).split(';')]
            except ValueError as exc:
                raise ValueError(
                    f"{json_config_path}: exclusion sample {sample_idx} has invalid motif_label "
                    f"{sample['motif_label']!r}."
                ) from exc
            if any(value != excluded_class_index for value in motif_labels):
                raise ValueError(
                    f"{json_config_path}: exclusion sample {sample_idx} motif_label values must all match "
                    f"excluded class index {excluded_class_index}; got {motif_labels}."
                )

    raw_expected = raw_spec.get('expected_allowed_proportions')
    raw_tolerance = raw_spec.get('proportion_tolerance')
    expected = None
    tolerance = None
    if raw_expected is None:
        if raw_tolerance is not None:
            raise ValueError(
                f"{json_config_path}: proportion_tolerance requires expected_allowed_proportions."
            )
    else:
        if not isinstance(raw_expected, dict):
            raise TypeError(
                f"{json_config_path}: expected_allowed_proportions must be a class-to-proportion object."
            )
        allowed_names = [class_labels[i] for i in allowed_class_indices]
        if set(raw_expected) != set(allowed_names):
            raise ValueError(
                f"{json_config_path}: expected_allowed_proportions must contain exactly {allowed_names}; "
                f"got {sorted(raw_expected)}."
            )

        expected_values = []
        for name in allowed_names:
            value = raw_expected[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{json_config_path}: proportion for {name} must be a finite number.")
            if value < 0:
                raise ValueError(f"{json_config_path}: proportion for {name} must be nonnegative.")
            expected_values.append(float(value))
        if not math.isclose(sum(expected_values), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"{json_config_path}: expected allowed proportions must sum to 1; "
                f"got {sum(expected_values):.8g}."
            )

        if isinstance(raw_tolerance, bool) or not isinstance(raw_tolerance, (int, float)):
            raise ValueError(
                f"{json_config_path}: proportion_tolerance is required and must be a finite number."
            )
        tolerance = float(raw_tolerance)
        if not math.isfinite(tolerance) or not 0 <= tolerance < 1:
            raise ValueError(
                f"{json_config_path}: proportion_tolerance must be in [0, 1); got {raw_tolerance!r}."
            )
        expected = tuple(expected_values)

    return SupervisionSpec(
        config_path=str(Path(json_config_path)),
        supervision_type='exclusion',
        excluded_class_index=excluded_class_index,
        excluded_class_name=excluded_class_name,
        allowed_class_indices=allowed_class_indices,
        expected_allowed_proportions=expected,
        proportion_tolerance=tolerance,
        sample_label_smoothing=sample_smoothing,
    )


def supervised_labelled_collate_fn(batch):
    """Collate records with one config ID and smoothing epsilon per chunk.

    Returns ``(labelled_batch, group_ids, label_smoothing)``. The payload follows
    :func:`porefm.data.reader.labelled_collate_fn`; metadata tensors have shape
    ``(batch_size,)`` with long IDs and float32 epsilons. Targets stay unchanged.
    """
    items, group_ids, smoothing_values = zip(*batch)
    return (
        labelled_collate_fn(list(items)),
        torch.tensor(group_ids, dtype=torch.long),
        torch.tensor(smoothing_values, dtype=torch.float32),
    )
    
def extend_classifier_head(model, new_out_dim: int, copy_weights: bool = True):
    """Replace a transformer classifier output layer with a wider layer.

    ``model`` is mutated at ``classifier_head.classifier_head.fc2``.
    ``new_out_dim`` must exceed the current class count. With
    ``copy_weights=True``, existing class rows are copied and only new rows use
    random initialization; otherwise all rows are reinitialized. A summary is
    printed after replacement.
    """
    head = model.classifier_head.classifier_head
    old_fc2 = head.fc2
    
    old_out_dim = old_fc2.out_features  # N
    in_features = old_fc2.in_features   # dim_ff
    
    if new_out_dim <= old_out_dim:
        raise ValueError(f"New dimension ({new_out_dim}) must be greater than old dimension ({old_out_dim})")

    new_fc2 = nn.Linear(in_features, new_out_dim)
    
    if copy_weights:
        with torch.no_grad():
            new_fc2.weight[:old_out_dim, :] = old_fc2.weight.data

            if old_fc2.bias is not None:
                new_fc2.bias[:old_out_dim] = old_fc2.bias.data
                
        print(f"Success: Copied weights for {old_out_dim} classes. Classes {old_out_dim} to {new_out_dim-1} are random.")
    else:
        print(f"Success: Initialized all {new_out_dim} class logits randomly.")

    head.fc2 = new_fc2
    
def get_global_counts(local_instances, local_base_tokens, local_signal_tokens, device):
    """Sum instance and token counters across distributed ranks.

    Local values are returned unchanged when no process group is initialized.
    Otherwise an all-reduce on ``device`` returns three Python integers.
    """
    if not dist.is_initialized():
        return local_instances, local_base_tokens, local_signal_tokens
        
    tensor = torch.tensor([local_instances, local_base_tokens, local_signal_tokens], device=device, dtype=torch.long)

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    return tensor[0].item(), tensor[1].item(), tensor[2].item()

def setup_distributed():
    """Initialize NCCL from torchrun environment variables when present.

    Returns ``(rank, world_size, local_rank)`` and selects the local CUDA device
    in distributed mode. A normal invocation returns ``(0, 1, 0)`` without
    creating a process group.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
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

def new_classifier_metrics(num_classes: int):
    """Create zeroed loss, confusion, and exclusion metric accumulators."""
    return {
        'loss_sum': 0.0,
        'loss_count': 0,
        'classification_loss_sum': 0.0,
        'classification_loss_count': 0,
        'exclusion_loss_sum': 0.0,
        'exclusion_loss_count': 0,
        'proportion_loss_sum': 0.0,
        'proportion_loss_count': 0,
        'confusion': torch.zeros(num_classes, num_classes, dtype=torch.long),
        'excl_success': torch.zeros(num_classes, dtype=torch.long),
        'excl_count': torch.zeros(num_classes, dtype=torch.long),
    }

def add_loss_metric(task_metrics, loss: float):
    """Add one scalar batch loss to a mutable metric accumulator."""
    task_metrics['loss_sum'] += loss
    task_metrics['loss_count'] += 1

def add_supervised_loss_component_metrics(task_metrics, component_stats):
    """Accumulate classification, exclusion, and proportion loss components."""
    for name in ('classification', 'exclusion', 'proportion'):
        task_metrics[f'{name}_loss_sum'] += component_stats[f'{name}_loss_sum']
        task_metrics[f'{name}_loss_count'] += component_stats[f'{name}_loss_count']

def add_confusion_metric(task_metrics, preds: torch.Tensor | None, targets: torch.Tensor | None, num_classes: int):
    """Add a batch confusion matrix when predictions and targets are available."""
    if preds is None or targets is None:
        return

    task_metrics['confusion'] += pretrain_train.confusion_matrix_from_predictions(preds, targets, num_classes)

def add_exclusion_metric(task_metrics, excluded_classes: torch.Tensor | None, exclusion_success: torch.Tensor | None, num_classes: int):
    """Accumulate per-class exclusion cases and successful avoidances."""
    if excluded_classes is None or exclusion_success is None:
        return

    excluded_classes = excluded_classes.reshape(-1).to(device='cpu', dtype=torch.long)
    exclusion_success = exclusion_success.reshape(-1).to(device='cpu', dtype=torch.bool)
    valid = (excluded_classes >= 0) & (excluded_classes < num_classes)
    if not valid.any():
        return

    valid_classes = excluded_classes[valid]
    valid_success = exclusion_success[valid]
    task_metrics['excl_count'] += torch.bincount(valid_classes, minlength=num_classes)

    if valid_success.any():
        task_metrics['excl_success'] += torch.bincount(valid_classes[valid_success], minlength=num_classes)

def reduce_long_stats(stats: torch.Tensor, device: torch.device, world_size: int) -> torch.Tensor:
    """Sum an integer statistics tensor across ranks and return it on CPU."""
    if world_size == 1:
        return stats

    reduced = stats.to(device=device, dtype=torch.long)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced.cpu()

def get_unwrapped_model(model):
    """Remove DDP and ``torch.compile`` wrappers when present."""
    model = model.module if isinstance(model, DDP) else model
    return getattr(model, "_orig_mod", model)

def get_pretrain_module(model):
    """Return the pretrained backbone from a classifier model or wrapper."""
    return get_unwrapped_model(model).pretrain

def normalize_classifier_residual_control(value):
    """Normalize CLI ``None``/``"none"`` residual control to Python ``None``."""
    if value in (None, 'none'):
        return None
    return value

def uses_weight_decay(name, param):
    """Return whether a classifier parameter belongs in the decay group.

    Scalars/vectors, biases, normalization parameters, and gate parameters are
    excluded. Backbone parameters are separated before this predicate is used.
    """
    if param.ndim < 2:
        return False

    name_lower = name.lower()
    if name_lower.endswith('.bias'):
        return False
    if 'norm' in name_lower:
        return False
    if name_lower.endswith('.gate') or '.gate.' in name_lower:
        return False

    return True

def reset_optimizer_hparams(optimizer, lr, pretrain_lr, classifier_weight_decay):
    """Reset learning rates and decay after loading optimizer state.

    Named modern groups are updated by role. A two-group positional fallback
    supports checkpoints saved before group names were introduced.
    """
    for idx, param_group in enumerate(optimizer.param_groups):
        group_name = param_group.get("name")
        if group_name == "pretrain":
            param_group["lr"] = pretrain_lr
            param_group["weight_decay"] = 0.0
        elif group_name == "classifier_decay":
            param_group["lr"] = lr
            param_group["weight_decay"] = classifier_weight_decay
        elif group_name == "classifier_no_decay":
            param_group["lr"] = lr
            param_group["weight_decay"] = 0.0
        else:
            # Backward-compatible fallback for optimizer states saved before
            # named parameter groups were added.
            is_pretrain_group = len(optimizer.param_groups) == 2 and idx == 1
            param_group["lr"] = pretrain_lr if is_pretrain_group else lr
            param_group["weight_decay"] = 0.0

def make_optimizer(model, lr, pretrain_lr=None, classifier_weight_decay=0.0):
    """Build classifier/backbone optimizer groups with independent settings.

    A plain Adam optimizer is returned when both learning rates are equal and
    decay is disabled. Otherwise backbone parameters receive ``pretrain_lr``
    without decay, eligible classifier matrices receive AdamW decay, and the
    remaining classifier parameters receive no decay.
    """
    pretrain_lr = lr if pretrain_lr is None else pretrain_lr

    if pretrain_lr == lr and classifier_weight_decay == 0:
        return torch.optim.Adam(model.parameters(), lr=lr)

    pretrain_params, classifier_decay_params, classifier_no_decay_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".pretrain." in name or name.startswith("pretrain.") or name.startswith("_orig_mod.pretrain."):
            pretrain_params.append(param)
        elif classifier_weight_decay > 0 and uses_weight_decay(name, param):
            classifier_decay_params.append(param)
        else:
            classifier_no_decay_params.append(param)

    param_groups = []
    if classifier_decay_params:
        param_groups.append({"params": classifier_decay_params, "lr": lr, "weight_decay": classifier_weight_decay, "name": "classifier_decay"})
    if classifier_no_decay_params:
        param_groups.append({"params": classifier_no_decay_params, "lr": lr, "weight_decay": 0.0, "name": "classifier_no_decay"})
    if pretrain_params:
        param_groups.append({"params": pretrain_params, "lr": pretrain_lr, "weight_decay": 0.0, "name": "pretrain"})

    optimizer_cls = torch.optim.AdamW if classifier_weight_decay > 0 else torch.optim.Adam
    return optimizer_cls(param_groups, lr=lr)

def count_trainable_parameters(model):
    """Return ``(total_parameters, trainable_parameters)`` for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_next_pretrain_batch(pretrain_iter, pretrain_dataloader):
    """Fetch the next auxiliary batch, restarting a finite iterator if needed."""
    try:
        return next(pretrain_iter), pretrain_iter
    except StopIteration:
        pretrain_iter = iter(pretrain_dataloader)
        return next(pretrain_iter), pretrain_iter

def log_pretrain_metrics(pretrain_metrics, print_task_list, checkpoint, writer, device, world_size, is_main_process):
    """Reduce, print, and write auxiliary pretraining metrics by task."""
    dna_bases = ['A', 'C', 'G', 'T']
    for pretrain_task in print_task_list:
        task_metrics = pretrain_metrics[pretrain_task]

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
            writer.add_scalar(f'Pretrain/Loss/{pretrain_task}', avg_loss, checkpoint)

        num_classes = pretrain_train.CLASSIFICATION_TASK_NUM_CLASSES.get(pretrain_task)
        confusion = None
        if num_classes is not None:
            confusion = task_metrics['confusion']
            if confusion is None:
                confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
            confusion = pretrain_train.reduce_confusion_stats(confusion, device, world_size)

        if confusion is not None and confusion.sum().item() > 0:
            accuracy, precision, recall, f1, p_class, r_class, f1_class = pretrain_train.macro_metrics_from_confusion(confusion)
            if is_main_process:
                writer.add_scalar(f'Pretrain/Accuracy/{pretrain_task}', accuracy, checkpoint)
                print_str = f"  PRETRAIN {pretrain_task.upper():<7} Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}"

                if pretrain_task == 'sublm':
                    print_str += f" | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}"
                    writer.add_scalar(f'Pretrain/Precision/{pretrain_task}', precision, checkpoint)
                    writer.add_scalar(f'Pretrain/Recall/{pretrain_task}', recall, checkpoint)
                    writer.add_scalar(f'Pretrain/F1-Score/{pretrain_task}', f1, checkpoint)

                elif pretrain_task == 'masklm':
                    labels_of_interest = [0, 1, 2, 3]
                    _, macro_p, macro_r, macro_f1, p_class, r_class, f1_class = pretrain_train.macro_metrics_from_confusion(
                        confusion,
                        labels=labels_of_interest,
                    )
                    print_str += f" | Macro P: {macro_p:.4f} | Macro R: {macro_r:.4f} | Macro F1: {macro_f1:.4f}"
                    writer.add_scalar('Pretrain/Precision/masklm_macro', macro_p, checkpoint)
                    writer.add_scalar('Pretrain/Recall/masklm_macro', macro_r, checkpoint)
                    writer.add_scalar('Pretrain/F1-Score/masklm_macro', macro_f1, checkpoint)

                    for i, base in enumerate(dna_bases):
                        print_str += f" | {base} F1: {f1_class[i]:.4f}"
                        writer.add_scalar(f'Pretrain/F1-Score/masklm_{base}', f1_class[i], checkpoint)
                        writer.add_scalar(f'Pretrain/Precision/masklm_{base}', p_class[i], checkpoint)
                        writer.add_scalar(f'Pretrain/Recall/masklm_{base}', r_class[i], checkpoint)

                print(print_str, flush=True)
        elif is_main_process:
            print(f"  PRETRAIN {pretrain_task.upper():<7} Loss: {avg_loss:.4f}", flush=True)

def cross_entropy_with_logit_regularization(
    logits,
    targets,
    logit_reg_lambda=1e-3,
    label_smoothing=0.02,
    focal_loss=False,
    focal_gamma=2.0,
    focal_alpha=None,
):
    """Compute float32 cross-entropy or focal loss plus raw-logit regularization.

    ``logits`` and ``targets`` follow the corresponding PyTorch loss contract;
    floating targets and logits are converted to float32 before loss operations.
    Integer class-index targets retain their dtype. The penalty is the mean
    square of raw logits, constraining both class differences and their shared
    offset. Focal parameters are ignored unless ``focal_loss`` is true.
    """
    logits = logits.float()
    if targets.is_floating_point():
        targets = targets.float()

    if focal_loss:
        classification_loss = focal_loss_fn(
            logits,
            targets,
            gamma=focal_gamma,
            alpha=focal_alpha,
            label_smoothing=label_smoothing,
        )
    else:
        classification_loss = F.cross_entropy(
            logits,
            targets,
            label_smoothing=label_smoothing,
        )

    # Bound the shared offset as well as the class-score differences.
    logit_penalty = logits.square().mean()

    loss = classification_loss + logit_reg_lambda * logit_penalty

    return loss


def compute_supervised_loss_components(
    logits,
    targets,
    base_group_ids,
    supervision_specs,
    proportion_loss_weight=0.0,
    logit_reg_lambda=0,
    label_smoothing=0,
    focal_loss=False,
    focal_gamma=2.0,
    focal_alpha=None,
):
    """Route retained bases to classification and config-driven exclusion loss.

    Parameters
    ----------
    logits, targets : torch.Tensor
        Matching ``(N, C)`` classifier logits and target distributions for
        retained labelled bases. Both are converted to float32 before loss
        computation; gradients still propagate to the original logits.
    base_group_ids : torch.Tensor
        Length-``N`` source-config indices into ``supervision_specs``.
    supervision_specs : sequence of SupervisionSpec
        One validated policy per config group.
    proportion_loss_weight : float, default=0
        Scale for the optional allowed-class aggregate proportion penalty.
    logit_reg_lambda : float, default=0
        Scale for mean squared raw logits over all retained bases and classes.
        This constrains both class-score differences and their shared offset.
    label_smoothing : float or torch.Tensor, default=0
        Uniform-target mixing epsilon, either a scalar or shape ``(N,)`` with
        one value per retained base. Only classification targets are smoothed;
        the caller's targets remain unchanged for metrics.
    focal_loss, focal_gamma, focal_alpha
        Classification-only focal-loss controls.

    Returns
    -------
    dict
        Differentiable float32 total ``loss``; summed component losses and counts;
        classification/exclusion masks; and the excluded class per base.

    Notes
    -----
    Exclusion loss is ``logsumexp(all) - logsumexp(allowed)`` per base. When
    expectations are configured, soft probabilities are aggregated separately
    per active exclusion group, conditioned on allowed classes, and penalized
    only outside the configured tolerance. Eligible group penalties are
    averaged equally.
    """
    if logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError(
            f"Expected logits and targets with matching [N, C] shapes; "
            f"got logits={tuple(logits.shape)} targets={tuple(targets.shape)}."
        )
    if base_group_ids.ndim != 1 or len(base_group_ids) != len(logits):
        raise ValueError(
            f"Expected one group ID per retained base; got {tuple(base_group_ids.shape)} "
            f"for {len(logits)} bases."
        )
    if len(logits) == 0:
        raise ValueError("No supervised bases remain after label masking and edge trimming.")
    if len(supervision_specs) == 0:
        raise ValueError("At least one supervision specification is required.")

    # Autocast outputs keep their dtype after leaving the context. Convert
    # before log-softmax and reductions, where rounding can distort gradients.
    logits = logits.float()
    targets = targets.float()

    base_group_ids = base_group_ids.to(device=logits.device, dtype=torch.long)
    if base_group_ids.min().item() < 0 or base_group_ids.max().item() >= len(supervision_specs):
        raise ValueError("Retained base group IDs are outside the supervision-specification range.")

    exclusion_by_group = torch.tensor(
        [spec.supervision_type == 'exclusion' for spec in supervision_specs],
        device=logits.device,
        dtype=torch.bool,
    )
    excluded_class_by_group = torch.tensor(
        [spec.excluded_class_index if spec.excluded_class_index is not None else -1 for spec in supervision_specs],
        device=logits.device,
        dtype=torch.long,
    )
    is_exclusion = exclusion_by_group[base_group_ids]
    is_classification = ~is_exclusion
    excluded_classes = excluded_class_by_group[base_group_ids]

    zero = logits.sum() * 0.0
    classification_count = int(is_classification.sum().item())
    if classification_count:
        classification_logits = logits[is_classification]
        classification_targets = targets[is_classification]
        smoothing = torch.as_tensor(label_smoothing, device=logits.device, dtype=torch.float32)
        if smoothing.ndim == 1 and smoothing.shape == (len(logits),):
            smoothing = smoothing[is_classification, None]
        elif smoothing.ndim != 0:
            raise ValueError("label_smoothing must be a scalar or have shape (N,) for retained bases.")
        if not torch.all(torch.isfinite(smoothing) & (smoothing >= 0) & (smoothing < 1)):
            raise ValueError("label_smoothing must contain finite values in [0, 1).")
        # Smooth once at the loss boundary; the loader preserves source targets.
        classification_targets = (
            classification_targets * (1.0 - smoothing) + smoothing / logits.shape[1]
        )
        if focal_loss:
            classification_loss_sum = focal_loss_fn(
                classification_logits,
                classification_targets,
                gamma=focal_gamma,
                alpha=focal_alpha,
                reduction='sum',
                label_smoothing=0.0,
            )
        else:
            classification_loss_sum = F.cross_entropy(
                classification_logits,
                classification_targets,
                reduction='sum',
                label_smoothing=0.0,
            )
    else:
        classification_loss_sum = zero

    exclusion_count = int(is_exclusion.sum().item())
    exclusion_loss_values = []
    if exclusion_count:
        for group_id in torch.unique(base_group_ids[is_exclusion]).tolist():
            spec = supervision_specs[group_id]
            group_mask = is_exclusion & (base_group_ids == group_id)
            group_logits = logits[group_mask].float()
            allowed = list(spec.allowed_class_indices)
            if len(allowed) == 0 or max(allowed) >= logits.shape[1]:
                raise ValueError(f"Invalid allowed classes for exclusion config {spec.config_path}.")
            group_loss = (
                torch.logsumexp(group_logits, dim=-1)
                - torch.logsumexp(group_logits[:, allowed], dim=-1)
            )
            exclusion_loss_values.append(group_loss)
        exclusion_loss_sum = torch.cat(exclusion_loss_values).sum()
    else:
        exclusion_loss_sum = zero

    base_loss = (classification_loss_sum + exclusion_loss_sum) / len(logits)

    proportion_group_losses = []
    if proportion_loss_weight != 0 and exclusion_count:
        for group_id in torch.unique(base_group_ids[is_exclusion]).tolist():
            spec = supervision_specs[group_id]
            if spec.expected_allowed_proportions is None:
                continue
            group_logits = logits[is_exclusion & (base_group_ids == group_id)].float()
            allowed = list(spec.allowed_class_indices)
            log_probs = F.log_softmax(group_logits, dim=-1)
            log_soft_counts = torch.logsumexp(log_probs[:, allowed], dim=0)
            log_allowed_total = torch.logsumexp(log_soft_counts, dim=0)
            predicted_proportions = (log_soft_counts - log_allowed_total).exp()
            expected_proportions = torch.tensor(
                spec.expected_allowed_proportions,
                device=logits.device,
                dtype=predicted_proportions.dtype,
            )
            errors = (
                (predicted_proportions - expected_proportions).abs()
                - spec.proportion_tolerance
            ).clamp_min(0)
            proportion_group_losses.append(errors.square().mean())

    proportion_loss_count = len(proportion_group_losses)
    if proportion_loss_count:
        proportion_loss_sum = torch.stack(proportion_group_losses).sum()
        proportion_loss = proportion_loss_sum / proportion_loss_count
    else:
        proportion_loss_sum = zero
        proportion_loss = zero

    logit_penalty = logits.square().mean()
    loss = (
        base_loss
        + proportion_loss_weight * proportion_loss
        + logit_reg_lambda * logit_penalty
    )

    return {
        'loss': loss,
        'classification_loss_sum': classification_loss_sum,
        'classification_loss_count': classification_count,
        'exclusion_loss_sum': exclusion_loss_sum,
        'exclusion_loss_count': exclusion_count,
        'proportion_loss_sum': proportion_loss_sum,
        'proportion_loss_count': proportion_loss_count,
        'is_classification': is_classification,
        'is_exclusion': is_exclusion,
        'excluded_classes': excluded_classes,
    }

def training_step(
    model,
    batch,
    optimizer,
    scheduler,
    device,
    num_trim=16,
    logit_reg_lambda=0,
    focal_loss=False,
    focal_gamma=2.0,
    focal_alpha=None,
    supervision_specs=None,
    proportion_loss_weight=0.0,
    include_dummy_loss=True,
):
    """Run one supervised optimizer and scheduler update.

    ``batch`` is ``(labelled_batch, item_group_ids, item_label_smoothing)`` from
    :func:`supervised_labelled_collate_fn`. Bases outside ``num_trim`` at each
    sequence edge or false in ``labels_idx`` are excluded. The model runs under
    bfloat16 autocast with float32 final classifier projections, losses, and
    raw-logit regularization. Gradients are clipped to norm 1, and both optimizer
    and scheduler are stepped. ``include_dummy_loss`` keeps otherwise-unused model
    parameters in the distributed autograd graph.

    Returns
    -------
    tuple
        Scalar loss; CPU predictions and target indices for classification
        bases; excluded classes and success flags for exclusion bases; and a
        dictionary of component sums/counts. Metric tensors are ``None`` when
        their supervision type was absent.
    """
    labelled_batch, item_group_ids, item_label_smoothing = batch
    base_seq_input, ts_input, percentiles_input, pad_base_seq, pad_ts, labels, labels_idx, positions, strands, readnames, refnames, _min_chrom, min_start, rank = labelled_batch
    
    base_seq_input = base_seq_input.to(device, non_blocking=True)
    ts_input = ts_input.to(device, non_blocking=True)
    percentiles_input = percentiles_input.to(device, non_blocking=True)
    pad_base_seq = pad_base_seq.to(device, non_blocking=True)
    pad_ts = pad_ts.to(device, non_blocking=True)
    
    labels, labels_idx = labels.to(device, non_blocking=True), labels_idx.to(device, non_blocking=True)
    item_group_ids = item_group_ids.to(device, non_blocking=True)
    item_label_smoothing = item_label_smoothing.to(device, non_blocking=True)

    trimmed_pad_base_seq = ignore_edges(pad_base_seq, num_trim)
    filter_pos = trimmed_pad_base_seq & labels_idx
    
    optimizer.zero_grad()
    
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        out = model(base_seq_input, ts_input, percentiles_input=percentiles_input,
                    pad_base_seq=pad_base_seq, pad_ts=pad_ts)
            
    logits = out['classifier_logits'][filter_pos]
    targets = labels[filter_pos]
    base_group_ids = item_group_ids[:, None].expand_as(filter_pos)[filter_pos]
    base_label_smoothing = item_label_smoothing[:, None].expand_as(filter_pos)[filter_pos]

    loss_components = compute_supervised_loss_components(
        logits,
        targets,
        base_group_ids,
        supervision_specs,
        proportion_loss_weight=proportion_loss_weight,
        logit_reg_lambda=logit_reg_lambda,
        label_smoothing=base_label_smoothing,
        focal_loss=focal_loss,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )
    loss = loss_components['loss']
    is_standard = loss_components['is_classification']
    is_exclusion = loss_components['is_exclusion']

    std_logits = logits[is_standard]
    std_targets = targets[is_standard]
    std_preds = std_logits.argmax(dim=-1)
    std_targets_idx = std_targets.argmax(dim=-1)

    excl_logits = logits[is_exclusion]
    excluded_classes = loss_components['excluded_classes'][is_exclusion]
    excl_preds = excl_logits.argmax(dim=-1)
    
    exclusion_success = (excl_preds != excluded_classes)

    if include_dummy_loss:
        loss += get_dummy_loss(model)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    component_stats = {
        name: (
            float(loss_components[name].detach().item())
            if name.endswith('_sum')
            else loss_components[name]
        )
        for name in (
            'classification_loss_sum',
            'classification_loss_count',
            'exclusion_loss_sum',
            'exclusion_loss_count',
            'proportion_loss_sum',
            'proportion_loss_count',
        )
    }

    return loss.item(), \
           std_preds.cpu() if std_preds.numel() > 0 else None, \
           std_targets_idx.cpu() if std_targets_idx.numel() > 0 else None, \
           excluded_classes.cpu() if excluded_classes.numel() > 0 else None, \
           exclusion_success.cpu() if exclusion_success.numel() > 0 else None, \
           component_stats
    
def _get_chrom_info(contigs: List[str], ref_path: str) -> List[tuple]:
    """Return full zero-based half-open FASTA spans for selected contigs.

    An empty ``contigs`` list selects every reference in FASTA order.
    """
    fref = pysam.FastaFile(ref_path)
    
    if not contigs:
        return [(c, 0, fref.get_reference_length(c)) for c in fref.references]
    
    return [(c, 0, fref.get_reference_length(c)) for c in contigs]

def parse_bam_config(
    json_config_path: str,
    num_classes: int,
    class_labels: Optional[List[str]] = None,
    label_smoothing: float = 0.0,
):
    """Expand one supervised JSON config into iterable sample objects.

    Each ``bam_path`` may be a file or a directory recursively expanded to
    naturally sorted ``*.bam`` files. Reference-based labelled and tagged
    samples receive full contig spans from their FASTA. Read-based samples omit
    FASTA access and, without explicit contigs, use an empty region list so the
    loader streams until EOF; explicit contigs become ``(contig, 0, None)``.
    Polishing, label, and filter fields are forwarded to the sample classes and
    ``num_classes`` is injected by the model configuration.
    ``label_smoothing`` is the CLI fallback; sample/config overrides are
    resolved before directory expansion and stored as loss-only metadata.

    Returns
    -------
    list
        Labelled samples followed by tagged samples in JSON order, with
        directory expansions preserving natural filename order.
    """
    labelled_data = []
    tagged_data = []

    validation_labels = class_labels or [str(index) for index in range(num_classes)]
    config = load_supervised_config(
        json_config_path,
        class_labels=validation_labels,
    )
    smoothing_specs = resolve_sample_label_smoothing(config, label_smoothing)

    for sample_kwargs, smoothing_spec in zip(config.get("labelled_samples", []), smoothing_specs):
        kwargs = dict(sample_kwargs)
        kwargs['label_smoothing'] = smoothing_spec.value
        contigs = kwargs.pop("contigs", [])
        read_based = kwargs.get("read_based", False)
        if not isinstance(read_based, bool):
            raise TypeError(
                f"read_based must be true or false for {kwargs.get('bam_path')!r}, "
                f"got {read_based!r}"
            )
        if read_based:
            kwargs["chrom_info"] = [(contig, 0, None) for contig in contigs]
            kwargs.setdefault("ref", None)
        else:
            kwargs["chrom_info"] = _get_chrom_info(contigs, kwargs["ref"])
        kwargs['num_classes']=num_classes
        
        bam_path_str = kwargs.pop("bam_path")
        path_obj = Path(bam_path_str)
        
        if path_obj.is_dir():
            bam_files = natsorted([str(p) for p in path_obj.rglob('*.bam')])
            for bf in bam_files:
                labelled_data.append(LabelledBamSample(bam_path=bf, **kwargs))
        else:
            labelled_data.append(LabelledBamSample(bam_path=bam_path_str, **kwargs))

    for sample_kwargs, smoothing_spec in zip(config.get("tagged_samples", []), smoothing_specs):
        kwargs = dict(sample_kwargs)
        kwargs['label_smoothing'] = smoothing_spec.value
        contigs = kwargs.pop("contigs", [])
        kwargs["chrom_info"] = (
            [(contig, 0, None) for contig in contigs]
            if kwargs["label_coordinates"] == "read"
            else _get_chrom_info(contigs, kwargs["ref"])
        )
        kwargs['num_classes']=num_classes
        bam_path_str = kwargs.pop("bam_path")
        path_obj = Path(bam_path_str)
        
        if path_obj.is_dir():
            bam_files = natsorted([str(p) for p in path_obj.rglob('*.bam')])
            for bf in bam_files:
                tagged_data.append(TaggedBamSample(bam_path=bf, **kwargs))
        else:
            tagged_data.append(TaggedBamSample(bam_path=bam_path_str, **kwargs))

    return labelled_data+tagged_data

def create_model_from_pretrain(args, add_signal_cross_attn=None, enable_mtr=False, enable_pretraining_heads=False):
    """Construct a classifier and load a pretrained checkpoint backbone.

    Classifier architecture is derived from ``args`` and checkpoint config.
    Optional cross-attention/MTR constraints are delegated to
    :class:`ONTModelForClassification`. When auxiliary heads are enabled, the
    checkpoint must contain the required SUBLM/MASKLM and optional MTR states.

    Returns
    -------
    tuple
        Initialized classifier model and the pretrained checkpoint's ``meta``
        mapping. Backbone state loading is non-strict to allow the new
        classifier head.
    """
    classifier_type = args.classifier_type or 'cnn'
    if classifier_type != 'transformer_cross' and args.classifier_residual_control not in (None, 'none'):
        raise ValueError("--classifier_residual_control only applies to transformer_cross classifier heads.")

    pretrain_ckpt = torch.load(args.pretrain_load_path, map_location="cpu")
    pretrain_config_dict = get_backbone_config_from_checkpoint(
        pretrain_ckpt,
        source=args.pretrain_load_path,
    )
    
    pretrain_config=ONTBackboneConfig(**pretrain_config_dict)
    state_dict=get_pretraining_state_dict_from_checkpoint(pretrain_ckpt)
    state_dict=strip_module_prefix(state_dict)
    checkpoint_supports_signal_cross_attn = checkpoint_has_signal_cross_attn_weights(
        state_dict
    )
    if classifier_type=='cnn':
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
                                    classifier_type=classifier_type,
                                    predict_center=True,
                                    residual_control=normalize_classifier_residual_control(args.classifier_residual_control),
                                    cross_attn_temp=pretrain_config.cross_attn_temp,
                                    expected_ratio=pretrain_config.expected_ratio,
                                    pre_attention_mixer_type=pretrain_config.pre_attention_mixer_type)
        
    model = ONTModelForClassification(
        classifier_config_dict=classifier_config_dict,
        pretrain_config=pretrain_config,
        add_signal_cross_attn=add_signal_cross_attn,
        backbone_signal_cross_attn=args.backbone_signal_cross_attn,
        classifier_signal_source=args.classifier_signal_source,
        classifier_center_source=args.classifier_center_source,
        enable_mtr=enable_mtr,
        enable_pretraining_heads=enable_pretraining_heads,
        checkpoint_supports_signal_cross_attn=checkpoint_supports_signal_cross_attn,
        )

    if enable_pretraining_heads:
        required_prefixes = ['sublm_head.', 'masklm_head.']
        if enable_mtr:
            required_prefixes.extend(['ema_signal_encoder.', 'latent_head.'])
        missing_prefixes = [
            prefix for prefix in required_prefixes
            if not any(key.startswith(prefix) for key in state_dict)
        ]
        if missing_prefixes:
            raise ValueError(
                "Cannot enable auxiliary pretraining from this pretrained checkpoint because "
                f"it is missing weights with prefixes: {missing_prefixes}"
            )
    model.pretrain.load_state_dict(state_dict, strict=False)
    model.preprocessing = preprocessing_from_checkpoint(pretrain_ckpt, required=False)
    model.pretrain.preprocessing = model.preprocessing
    pretrain_meta=pretrain_ckpt.get('meta', {})
    
    return model, pretrain_meta


def resolve_auxiliary_signal_dropout_policy(args, pretrain_config):
    """Resolve signal-modality dropout used only by auxiliary pretraining.

    Classification forwards remain fixed at zero. Auxiliary probability is
    opt-in through an explicit value or an explicit request to inherit the
    checkpoint value; EMA coupling inherits unless separately overridden.
    """
    controls_used = (
        args.pretrain_signal_dropout_prob is not None
        or args.use_loaded_pretrain_signal_dropout_prob
        or args.pretrain_ema_signal_dropout_mode is not None
    )
    if not args.train_pretrain_tasks:
        if controls_used:
            raise ValueError(
                "Auxiliary signal-dropout options require --train_pretrain_tasks."
            )
        return 0.0, pretrain_config.ema_signal_dropout_mode

    if args.use_loaded_pretrain_signal_dropout_prob:
        probability = float(pretrain_config.signal_dropout_prob)
    elif args.pretrain_signal_dropout_prob is not None:
        probability = float(args.pretrain_signal_dropout_prob)
    else:
        probability = 0.0

    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Auxiliary signal dropout probability must be finite and in [0, 1].")
    if probability > 0 and not pretrain_config.add_percentile:
        raise ValueError(
            "Auxiliary signal dropout must be 0 when the loaded backbone disables percentiles."
        )

    ema_mode = (
        args.pretrain_ema_signal_dropout_mode
        or pretrain_config.ema_signal_dropout_mode
    )
    return probability, ema_mode

def train_model(args):
    """Execute supervised fine-tuning from a validated CLI namespace.

    The routine initializes distributed execution, constructs or resumes the
    classifier/optimizer/scheduler, validates JSON-wide supervision and
    whole-file sharding, then trains until ``max_steps`` while periodically
    reducing metrics and writing modular checkpoints. Optional auxiliary
    pretraining steps share the model and optimizer. Rank zero owns TensorBoard
    and checkpoint I/O; the process group is destroyed before returning.
    """
    resolve_cli_preprocessing(args)
    model_save_path = args.model_save_path
        
    checkpoint_batch_step, lr, max_steps  = args.checkpoint_batch_step, args.lr, args.max_steps
    pretrain_lr = args.pretrain_lr if args.pretrain_lr is not None else lr

    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0
    from porefm.common.run_outputs import prepare_run_outputs
    prepare_run_outputs(args, command="supervised-finetune", directories={
        "model_save_path": model_save_path,
    }, distributed=dist)

    if is_main_process:
        print(
            '\n%s:\nCommand: python %s\n'
            % (str(datetime.datetime.now()), ' '.join(sys.argv)),
            flush=True,
        )
        print(f"{str(datetime.datetime.now())}: Starting Training", flush=True)

    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    
    if is_main_process:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(model_save_path, 'runs'))
    else:
        writer = NoOpWriter()
    
    num_classes = args.num_classes
    train_pretrain_tasks = args.train_pretrain_tasks
    enable_mtr = train_pretrain_tasks and ('mtr' in args.pretrain_task_list)
    
    # Get classifier and pretrain configurations, model meta data and pretrain data
    model_ckpt = None
    if args.model_load_path:
        model_ckpt = torch.load(args.model_load_path, map_location="cpu")
        maybe_restore_lora_args_from_checkpoint(args, model_ckpt)
        model = ONTModelForClassification(
            model_load_path=args.model_load_path,
            classifier_type=args.classifier_type,
            classifier_signal_source=args.classifier_signal_source,
            classifier_center_source=args.classifier_center_source,
            backbone_signal_cross_attn=args.backbone_signal_cross_attn,
            enable_mtr=enable_mtr,
            enable_pretraining_heads=train_pretrain_tasks,
        )
        pretrain_meta = model_ckpt['pretrain_meta']

        requested_residual_control = normalize_classifier_residual_control(args.classifier_residual_control)
        saved_residual_control = getattr(model.classifier_config, 'residual_control', None)
        if args.classifier_residual_control is not None and requested_residual_control != saved_residual_control:
            raise ValueError(
                "--classifier_residual_control can only change newly initialized classifier heads from "
                "--pretrain_load_path. When using --model_load_path, omit it or match the saved classifier config "
                f"({saved_residual_control!r})."
            )
        
        assert num_classes >= model.classifier_config.out_dim
        
        if num_classes > model.classifier_config.out_dim:
            extend_classifier_head(model, new_out_dim=num_classes, copy_weights=True)
            model.classifier_config.out_dim = num_classes

    else:
        if args.pretrain_load_path is None:
            raise ValueError(
                "You must provide either `model_load_path` or `pretrain_load_path` (which includes a config dict)."
            )
        model, pretrain_meta = create_model_from_pretrain(args, enable_mtr=enable_mtr, enable_pretraining_heads=train_pretrain_tasks)
        
    lora_stats = None
    if args.freeze_pretrain and args.lora_backbone:
        raise ValueError("--lora_backbone trains adapter weights in the pretrained backbone; remove --freeze_pretrain.")

    if args.lora_backbone:
        lora_stats = add_lora_to_pretrain_backbone(
            model.pretrain,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_groups=args.lora_target_modules,
        )
        set_pretrain_lora_trainability(
            model.pretrain,
            train_base_weights=args.lora_train_base_weights,
        )
        if model_ckpt is not None:
            state_dict = strip_module_prefix(get_classification_state_dict_from_checkpoint(model_ckpt))
            lora_state_dict = {k: v for k, v in state_dict.items() if ".lora_" in k}
            model.load_state_dict(lora_state_dict, strict=False)

    if args.freeze_pretrain:
        for p in model.pretrain.parameters():
            p.requires_grad = False

    if train_pretrain_tasks and args.freeze_pretrain:
        raise ValueError("--train_pretrain_tasks requires the pretrained backbone to be trainable; remove --freeze_pretrain.")
    if train_pretrain_tasks and not args.pretrain_bam_config:
        raise ValueError("--train_pretrain_tasks requires --pretrain_bam_config for the auxiliary pretraining batches.")
    
    pretrain_config = model.pretrain.config
    if enable_mtr and not pretrain_config.add_signal_cross_attn:
        raise ValueError("--train_pretrain_tasks with task 'mtr' requires signal-to-base cross-attention in the loaded checkpoint.")
    if enable_mtr and not pretrain_config.add_ema:
        raise ValueError("--train_pretrain_tasks with task 'mtr' requires the JEPA/MTR EMA encoder to be enabled.")

    auxiliary_signal_dropout_prob, auxiliary_ema_signal_dropout_mode = (
        resolve_auxiliary_signal_dropout_policy(args, pretrain_config)
    )
    if train_pretrain_tasks:
        pretrain_config.signal_dropout_prob = auxiliary_signal_dropout_prob
        pretrain_config.ema_signal_dropout_mode = auxiliary_ema_signal_dropout_mode
    if is_main_process:
        print(
            "Auxiliary signal modality dropout: "
            f"probability={auxiliary_signal_dropout_prob:g} "
            f"ema_mode={auxiliary_ema_signal_dropout_mode} "
            f"enabled={train_pretrain_tasks}",
            flush=True,
        )

    seq_type, pore_type = args.seq_type, args.pore_type
    max_base_seq_len, max_ts_len = pretrain_config.max_base_seq_len, pretrain_config.max_ts_len
    ts_dim = pretrain_config.ts_dim
    
    num_classes = model.classifier_config.out_dim
    if len(args.class_labels) != num_classes:
        raise ValueError(
            f"--class_labels must provide exactly {num_classes} unique names; got {args.class_labels}."
        )
    if len(set(args.class_labels)) != len(args.class_labels):
        raise ValueError("--class_labels must contain unique class names.")

    args.base, args.mod_code = resolve_modification_metadata(
        args.base, args.mod_code,
        metadata=model_ckpt.get('meta', {}) if model_ckpt is not None else {},
        num_classes=num_classes,
    )

    supervision_specs = [
        load_and_validate_supervision_spec(cfg_path, args.class_labels, args.label_smoothing)
        for cfg_path in args.bam_config
    ]
    
    classifier_residual_control = getattr(model.classifier_config, 'residual_control', None)
    meta_dict = dict(
        class_labels=args.class_labels,
        base=args.base,
        mod_code=args.mod_code,
        lr=lr,
        pretrain_lr=pretrain_lr,
        classifier_weight_decay=args.classifier_weight_decay,
        classifier_residual_control=classifier_residual_control,
        classifier_loss='focal' if args.focal_loss else 'cross_entropy',
        focal_gamma=args.gamma if args.focal_loss else None,
        focal_alpha=args.alpha if args.focal_loss else None,
        label_smoothing=args.label_smoothing,
        label_smoothing_policy='sample_over_config_over_cli',
        logit_reg_lambda=args.logit_reg_lambda,
        logit_regularization='raw_mean_square',
        classifier_projection_dtype='float32',
        supervised_loss_dtype='float32',
        proportion_loss_weight=args.proportion_loss_weight,
        supervision_specs=[asdict(spec) for spec in supervision_specs],
        train_pretrain_tasks=args.train_pretrain_tasks,
        pretrain_step_interval=args.pretrain_step_interval,
        pretrain_task_list=args.pretrain_task_list,
        pretrain_weights=args.pretrain_weights,
        pretrain_mask_prob_range=args.pretrain_mask_prob_range,
        pretrain_mask_prob_warmup=args.pretrain_mask_prob_warmup,
        pretrain_mask_base_span_range=args.pretrain_mask_base_span_range,
        pretrain_mask_signal_span=args.pretrain_mask_signal_span,
        pretrain_signal_dropout_prob=auxiliary_signal_dropout_prob,
        pretrain_ema_signal_dropout_mode=auxiliary_ema_signal_dropout_mode,
        classifier_execution_contract=model.execution_contract(),
        lora_config=get_lora_config(args),
    )
    
    model = model.to(device)
    model.eval()
    
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16), torch.no_grad():
        model(torch.ones(1,max_ts_len).long().to(device),
              torch.randn(1,max_ts_len,ts_dim).to(device),
              percentiles_input=torch.randn(1,max_ts_len, ts_dim).to(device),
              pad_base_seq=torch.ones(1,max_ts_len).to(torch.bool).to(device),
              pad_ts=torch.ones(1,max_ts_len).to(torch.bool).to(device),
             );

    if args.disable_compile == False:
        model = torch.compile(model, dynamic=True)
        
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=train_pretrain_tasks)

    model_to_print = model.module if isinstance(model, DDP) else model
    model_details = str(model_to_print)
    
    if is_main_process:
        print(model_details, flush=True)
        print(
            get_unwrapped_model(model).format_execution_contract(),
            flush=True,
        )
        num_params = sum(p.numel() for p in model_to_print.parameters())
        print(f'{str(datetime.datetime.now())}: # Parameters=', num_params, flush=True)    
        total_params, trainable_params = count_trainable_parameters(model_to_print)
        print(f'{str(datetime.datetime.now())}: # Trainable Parameters=', trainable_params, 'of', total_params, flush=True)
        if lora_stats is not None:
            print(
                f"{str(datetime.datetime.now())}: LoRA backbone enabled: "
                f"rank={args.lora_rank} alpha={args.lora_alpha:g} dropout={args.lora_dropout:g} "
                f"train_base_weights={args.lora_train_base_weights} "
                f"targets={len(lora_stats['target_names'])} "
                f"lora_params={lora_stats['trainable_lora_params']} "
                f"affected_base_params={lora_stats['affected_base_params']}",
                flush=True,
            )
            print(
                f"{str(datetime.datetime.now())}: LoRA target groups: "
                + ", ".join(f"{name}={LORA_TARGET_HELP[name]}" for name in args.lora_target_modules),
                flush=True,
            )
    
    t = time.time()
    optimizer = make_optimizer(
        model,
        lr=lr,
        pretrain_lr=None if args.freeze_pretrain else pretrain_lr,
        classifier_weight_decay=args.classifier_weight_decay,
    )
    if args.model_load_path and args.load_optimizer:
        try:
            optimizer.load_state_dict(model_ckpt["optimizer_state_dict"])
            reset_optimizer_hparams(
                optimizer,
                lr=lr,
                pretrain_lr=lr if args.freeze_pretrain else pretrain_lr,
                classifier_weight_decay=args.classifier_weight_decay,
            )
        except ValueError as exc:
            if is_main_process:
                print(f"Skipping optimizer state load because parameter groups changed: {exc}", flush=True)

    scheduler = rewarmup_then_constant(
        optimizer,
        warmup_steps=checkpoint_batch_step,
    )
    model.train()
    
    sub_epoch_metrics = defaultdict(lambda: new_classifier_metrics(num_classes))
    
    class_labels = args.class_labels
    labels_of_interest = list(range(num_classes))
    
    # Checkpoint numbering is based only on completed classifier updates.
    classification_update_num, checkpoint = 0, 0
                    
    size = min(max_ts_len, args.size or max_ts_len)
    step = resolve_downstream_step(
        size,
        seq_type,
        args.step,
        context="supervised fine-tuning",
        announce=is_main_process,
    )
    
    kmer_table = resolve_cli_preprocessing(args).kmer_table

    bam_config_lists = []
        
    for cfg_path in args.bam_config:
        bam_list = parse_bam_config(cfg_path, num_classes, args.class_labels, args.label_smoothing)
        if len(bam_list) == 0:
            raise ValueError(f"{cfg_path}: config did not expand to any BAM samples.")
        bam_config_lists.append(bam_list)
    
    group_weights = args.group_weights or [1]*len(bam_config_lists)
    if len(group_weights) != len(bam_config_lists):
        raise ValueError("--group_weights must provide exactly one weight per --bam_config file.")
    for group_id, (cfg_path, bam_list, weight) in enumerate(
        zip(args.bam_config, bam_config_lists, group_weights)
    ):
        if weight > 0:
            validate_ddp_bam_sharding(
                len(bam_list),
                world_size,
                args.dataloader_workers,
                f"supervised BAM config group {group_id} ({cfg_path})",
            )

    if is_main_process:
        print("Supervision config summary:", flush=True)
        for group_id, (spec, weight) in enumerate(zip(supervision_specs, group_weights)):
            if spec.supervision_type == 'classification':
                details = "classification"
            else:
                details = f"exclusion excluded={spec.excluded_class_name}"
                if spec.expected_allowed_proportions is not None:
                    expected = ", ".join(
                        f"{class_labels[class_idx]}={value:g}"
                        for class_idx, value in zip(
                            spec.allowed_class_indices,
                            spec.expected_allowed_proportions,
                        )
                    )
                    details += f" expected=({expected}) tolerance={spec.proportion_tolerance:g}"
                elif args.proportion_loss_weight != 0:
                    details += " proportion=SKIPPED (no expected_allowed_proportions)"
            print(
                f"  [{group_id}] weight={weight:g} {details} config={spec.config_path}",
                flush=True,
            )
            for smoothing_spec in spec.sample_label_smoothing:
                print(
                    f"    sample[{smoothing_spec.sample_index}] "
                    f"label_smoothing={smoothing_spec.value:g} source={smoothing_spec.source} "
                    f"bam_path={smoothing_spec.bam_path}",
                    flush=True,
                )
        if args.proportion_loss_weight != 0:
            for spec in supervision_specs:
                if spec.supervision_type == 'exclusion' and spec.expected_allowed_proportions is None:
                    print(
                        f"WARNING: {spec.config_path} has exclusion supervision but no expected proportions; "
                        "its proportion loss will be skipped.",
                        flush=True,
                    )
            if not any(
                spec.supervision_type == 'exclusion'
                and spec.expected_allowed_proportions is not None
                for spec in supervision_specs
            ):
                print(
                    "WARNING: --proportion_loss_weight is nonzero, but no exclusion config has "
                    "expected_allowed_proportions; no proportion loss will be applied.",
                    flush=True,
                )
    
    ds = SupervisedWeightedLabelledBamIterableDataset(
        bam_lists=bam_config_lists,
        group_weights=group_weights,
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
        expected_ts_dim=pretrain_config.ts_dim,
    )

    dataloader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=supervised_labelled_collate_fn,
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

    pretrain_dataloader = None
    pretrain_iter = None
    pretrain_metrics = defaultdict(pretrain_train.new_sub_epoch_metrics)
    pretrain_print_task_list = [x for x in args.pretrain_task_list]
    if train_pretrain_tasks:
        mask_prob_start, mask_prob_end = args.pretrain_mask_prob_range
        mask_span_start, mask_span_end = args.pretrain_mask_base_span_range
        assert mask_prob_start <= mask_prob_end, 'pretrain mask_prob_range start should <= end'
        assert mask_span_start <= mask_span_end, 'pretrain mask_base_span_range start should <= end'

        pretrain_bam_list = []
        for cfg_path in args.pretrain_bam_config:
            pretrain_bam_list.extend(
                pretrain_train.load_pretraining_bam_list(cfg_path)
            )
        validate_ddp_bam_sharding(
            len(pretrain_bam_list),
            world_size,
            args.dataloader_workers,
            "auxiliary-pretraining BAM config",
        )
        
        pretrain_ds = BamIterableDataset(
            bam_list=pretrain_bam_list,
            size=size,
            move_downsample_factor=pretrain_config.move_downsample_factor,
            step=step,
            seq_type=seq_type,
            chunking_mode=resolve_chunking_mode(args), kmer_table=kmer_table,
            reads_per_chunk=args.reads_per_chunk,
            min_read_length=args.min_read_length,
            min_qscore=args.min_qscore,
            pore_type=pore_type,
            base_seed=args.seed,
            rank=rank,
            world_size=world_size,
            expected_ts_dim=pretrain_config.ts_dim,
        )

        pretrain_dataloader_kwargs = dict(dataloader_kwargs)
        pretrain_dataloader_kwargs['batch_size']=args.batch_size//4
        pretrain_dataloader_kwargs['collate_fn'] = pretrain_train.custom_collate_fn
        pretrain_dataloader = DataLoader(pretrain_ds, **pretrain_dataloader_kwargs)
        pretrain_iter = iter(pretrain_dataloader)

    task = 'Classification'

    total_instances_seen = 0
    total_base_tokens_seen = 0
    total_signal_tokens_seen = 0
    
    total_TP, total_FN, total_FP, total_TN = 0, 0, 0, 0
    
    for batch in dataloader:
        loss, std_preds, std_targets, excl_classes, excl_success, component_stats = training_step(
            model=model, batch=batch,
            optimizer=optimizer, 
            scheduler=scheduler,
            device=device,
            num_trim = args.num_trim,
            logit_reg_lambda = args.logit_reg_lambda,
            focal_loss = args.focal_loss,
            focal_gamma = args.gamma,
            focal_alpha = args.alpha,
            supervision_specs = supervision_specs,
            proportion_loss_weight = args.proportion_loss_weight,
            include_dummy_loss = not train_pretrain_tasks)

        task_metrics = sub_epoch_metrics[task]
        add_loss_metric(task_metrics, loss)
        add_supervised_loss_component_metrics(task_metrics, component_stats)
        add_confusion_metric(task_metrics, std_preds, std_targets, num_classes)
        add_exclusion_metric(task_metrics, excl_classes, excl_success, num_classes)
        classification_update_num += 1

        if train_pretrain_tasks and classification_update_num % args.pretrain_step_interval == 0:
            current_mask_prob = mask_prob_start + (mask_prob_end - mask_prob_start) * min(checkpoint, args.pretrain_mask_prob_warmup) / args.pretrain_mask_prob_warmup
            pretrain_task, current_mask_span = pretrain_train.choose_pretraining_step_controls(
                args.pretrain_task_list,
                args.pretrain_weights,
                mask_span_start,
                mask_span_end,
                device,
                rank,
                world_size,
            )
            pretrain_batch, pretrain_iter = get_next_pretrain_batch(pretrain_iter, pretrain_dataloader)
            pretrain_loss, pretrain_preds, pretrain_targets = pretrain_train.training_step(
                model = model,
                batch = pretrain_batch,
                optimizer = optimizer,
                scheduler = scheduler,
                device = device,
                task = pretrain_task,
                mask_prob = current_mask_prob,
                mask_base_span = current_mask_span,
                mask_signal_span = args.pretrain_mask_signal_span,
                add_percentile = pretrain_config.add_percentile,
                eval_only = False,
                add_signal_cross_attn = pretrain_config.add_signal_cross_attn,
                num_base_trim = args.pretrain_num_loss_trim[0],
                num_signal_trim = args.pretrain_num_loss_trim[1],
                include_dummy_loss = False,
                signal_dropout_prob = auxiliary_signal_dropout_prob,
            )
            pretrain_task_metrics = pretrain_metrics[pretrain_task]
            add_loss_metric(pretrain_task_metrics, pretrain_loss)
            pretrain_train.update_classification_stats(pretrain_task_metrics, pretrain_task, pretrain_preds, pretrain_targets)
        
        labelled_batch, _item_group_ids, _item_label_smoothing = batch
        local_batch_size = len(labelled_batch[0])
        local_base_token_count = sum(len(seq) for seq in labelled_batch[0])
        local_signal_token_count = torch.sum(labelled_batch[4]).item()
        
        total_instances_seen += local_batch_size
        total_base_tokens_seen += local_base_token_count
        total_signal_tokens_seen += local_signal_token_count

        # --- Checkpoint and Logging ---
        if classification_update_num % checkpoint_batch_step == 0:
            checkpoint = classification_update_num // checkpoint_batch_step
            global_instances, global_base_tokens, global_signal_tokens = get_global_counts(total_instances_seen, total_base_tokens_seen, total_signal_tokens_seen, device)
            
            if is_main_process:
                print(f"{str(datetime.datetime.now())}: Global Instances: {global_instances:,} | Global Base Tokens: {global_base_tokens:,} | Global Signal Tokens: {global_signal_tokens:,}", flush=True)
                print(f"{str(datetime.datetime.now())}: Checkpoint:{checkpoint}   Classification Updates:{classification_update_num}   Time={time.time()-t:.2f}", flush=True)

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

            component_averages = {}
            for component_name in ('classification', 'exclusion', 'proportion'):
                component_total, component_count = reduce_loss_stats(
                    task_metrics[f'{component_name}_loss_sum'],
                    task_metrics[f'{component_name}_loss_count'],
                    device,
                    world_size,
                )
                component_averages[component_name] = (
                    component_total / component_count if component_count else None
                )

            if is_main_process:
                if component_averages['classification'] is not None:
                    writer.add_scalar(
                        'Loss/Classification_Base',
                        component_averages['classification'],
                        checkpoint,
                    )
                if component_averages['exclusion'] is not None:
                    writer.add_scalar(
                        'Loss/Exclusion',
                        component_averages['exclusion'],
                        checkpoint,
                    )
                if args.proportion_loss_weight != 0 and component_averages['proportion'] is not None:
                    writer.add_scalar(
                        'Loss/Proportion',
                        component_averages['proportion'],
                        checkpoint,
                    )
                    writer.add_scalar(
                        'Loss/Proportion_Weighted',
                        args.proportion_loss_weight * component_averages['proportion'],
                        checkpoint,
                    )

                if component_averages['exclusion'] is not None or args.proportion_loss_weight != 0:
                    component_lines = ["  --- Supervised Loss Components ---"]
                    if component_averages['classification'] is not None:
                        component_lines.append(
                            f"  | Classification Base Loss: {component_averages['classification']:.4f}"
                        )
                    if component_averages['exclusion'] is not None:
                        component_lines.append(
                            f"  | Exclusion Loss: {component_averages['exclusion']:.4f}"
                        )
                    if args.proportion_loss_weight != 0:
                        if component_averages['proportion'] is None:
                            component_lines.append("  | Proportion Loss: N/A (0 active eligible configs)")
                        else:
                            component_lines.append(
                                f"  | Proportion Loss: {component_averages['proportion']:.6f} "
                                f"| Weighted: {args.proportion_loss_weight * component_averages['proportion']:.6f}"
                            )
                    print("\n".join(component_lines), flush=True)

            confusion = pretrain_train.reduce_confusion_stats(task_metrics['confusion'], device, world_size)
            excl_count = reduce_long_stats(task_metrics['excl_count'], device, world_size)
            excl_success = reduce_long_stats(task_metrics['excl_success'], device, world_size)

            # --- Log Standard Metrics ---
            if confusion.sum().item() > 0 and is_main_process:
                if num_classes > 1:
                    TP = confusion[1, 1].item()
                    FN = confusion[1, :].sum().item() - TP
                    FP = confusion[:, 1].sum().item() - TP
                    TN = confusion[0, 0].item()
                else:
                    TP = FN = FP = 0
                    TN = confusion[0, 0].item()

                total_TP += TP
                total_FN += FN
                total_FP += FP
                total_TN += TN

                accuracy, macro_p, macro_r, macro_f1, p_class, r_class, f1_class = pretrain_train.macro_metrics_from_confusion(
                    confusion,
                    labels=labels_of_interest,
                )
                writer.add_scalar(f'Accuracy/{task}', accuracy, checkpoint)

                print_str = f"  {task.upper():<7} Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}"

                print_str += f" | Macro P: {macro_p:.4f} | Macro R: {macro_r:.4f} | Macro F1: {macro_f1:.4f}\n"
                writer.add_scalar(f'Precision/{task}_macro', macro_p, checkpoint)
                writer.add_scalar(f'Recall/{task}_macro', macro_r, checkpoint)
                writer.add_scalar(f'F1-Score/{task}_macro', macro_f1, checkpoint)

                for i, mod in enumerate(class_labels):
                    print_str += f"| {mod} Precision: {p_class[i]:.4f}  Recall: {r_class[i]:.4f}  F1: {f1_class[i]:.4f}\n"
                    writer.add_scalar(f'F1-Score/masklm_{mod}', f1_class[i], checkpoint)
                    writer.add_scalar(f'Precision/masklm_{mod}', p_class[i], checkpoint)
                    writer.add_scalar(f'Recall/masklm_{mod}', r_class[i], checkpoint)

                print(print_str, flush=True)
                precision, recall = total_TP/max(1, total_TP+total_FP), total_TP/max(1, total_TP+total_FN)
                f1 = 2*precision*recall/(precision+recall) if precision+recall!=0 else 0
                fpr = total_FP/max(1, total_TN+total_FP)
                print(f'TP={total_TP}  FN={total_FN}  FP={total_FP}  TN={total_TN}', flush=True)
                print(f'Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}  FPR={fpr:.4f}', flush=True)
                
            # --- Log Exclusion Metrics ---
            if excl_count.sum().item() > 0 and is_main_process:
                excl_print_str = "  --- Exclusion Success Rates ---\n"
                for i, mod in enumerate(class_labels):
                    total_exclusions = excl_count[i].item()
                    
                    if total_exclusions > 0:
                        successes = excl_success[i].item()
                        success_rate = successes / total_exclusions
                        excl_print_str += f"  | {mod} Excl Success: {success_rate:.4f} ({successes}/{total_exclusions})\n"
                        writer.add_scalar(f'Exclusion_Success/{mod}', success_rate, checkpoint)
                    else:
                        excl_print_str += f"  | {mod} Excl Success: N/A (0 cases)\n"
                print(excl_print_str, flush=True)

            if is_main_process:
                print('______________________________________________________________', flush=True)

            if train_pretrain_tasks:
                log_pretrain_metrics(pretrain_metrics, pretrain_print_task_list, checkpoint, writer, device, world_size, is_main_process)

            sub_epoch_metrics = defaultdict(lambda: new_classifier_metrics(num_classes))
            pretrain_metrics = defaultdict(pretrain_train.new_sub_epoch_metrics)

            if is_main_process:
                model_to_save = model.module if isinstance(model, DDP) else model
                model_path = os.path.join(model_save_path, f'model.{checkpoint}')
                ckpt = {
                        "checkpoint_type": "ont_classification_v2",
                        "model_class": "ONTModelForClassification",
                        'optimizer_state_dict': optimizer.state_dict(),
                        "classifier_config": asdict(model_to_save.classifier_config),
                        "pretrain_config": asdict(model_to_save.pretrain.config),
                        "lora_config": get_lora_config(args),
                        "meta": meta_dict,
                        "pretrain_meta": pretrain_meta
                    }
                ckpt.update(modular_classification_payload(model_to_save))
                attach_preprocessing(ckpt, args.preprocessing)
                torch.save(ckpt, model_path)
                
            if classification_update_num >= max_steps:
                break

    if is_main_process:
        print(f"{str(datetime.datetime.now())}: Training Finished.", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    return
    
if __name__=='__main__':
    parser = build_supervised_finetune_parser()
    args = parser.parse_args()

    if len(args.pretrain_task_list) != len(args.pretrain_weights):
        raise ValueError("The number of auxiliary pretraining tasks in --pretrain_task_list must match --pretrain_weights.")
    if args.pretrain_step_interval < 1:
        raise ValueError("--pretrain_step_interval must be >= 1.")
    if args.classifier_weight_decay < 0:
        raise ValueError("--classifier_weight_decay must be >= 0.")
    if args.logit_reg_lambda < 0:
        raise ValueError("--logit_reg_lambda must be >= 0.")
    if args.proportion_loss_weight < 0:
        raise ValueError("--proportion_loss_weight must be >= 0.")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("--label_smoothing must be in the range [0, 1).")
    if args.focal_loss and args.gamma < 0:
        raise ValueError("--gamma must be >= 0 when --focal_loss is enabled.")
    if args.alpha is not None and not args.focal_loss:
        raise ValueError("--alpha requires --focal_loss.")
    if args.alpha is not None:
        if args.num_classes is not None and len(args.alpha) != args.num_classes:
            raise ValueError(
                "--alpha must provide exactly one class weight per classifier class "
                f"({args.num_classes}), got {len(args.alpha)}."
            )
        if any(weight < 0 for weight in args.alpha):
            raise ValueError("--alpha class weights must be >= 0.")
        if not any(weight > 0 for weight in args.alpha):
            raise ValueError("--alpha must contain at least one positive class weight.")

    assert args.num_classes is not None
    train_model(args)
