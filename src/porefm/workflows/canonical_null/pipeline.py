#!/usr/bin/env python3
"""Prepare a canonical reference and calibration artifacts from one command.

This is intentionally an orchestrator, not a second implementation of the
training logic.  Each stage builds an argparse-like namespace and calls
``canonical_null_predictor.train_model(args)`` directly in this Python process.

The pipeline writes deterministic, disjoint pretraining-config splits, then runs
the requested stages:

  null predictor -> cosine scale -> whitening covariance -> whitening scale
  -> empirical calibration

Cosine and whitening can be run together or separately.  Existing null,
scale, or whitening artifacts can be reused when only a downstream step needs
to change.
"""

from __future__ import annotations

from porefm.cli.preprocessing_args import add_preprocessing_arguments, resolve_cli_preprocessing
import argparse
import datetime as _datetime
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from porefm.cli.chunking_args import add_chunking_arguments, resolve_chunking_mode
from porefm.cli._parser import PoreFMArgumentParser
from porefm.common.chunking import resolve_downstream_step
from porefm.config.validation import PretrainingConfig, PretrainingSample, load_pretraining_config

SPLIT_NULL = "null"
SPLIT_COSINE_SCALE = "cosine_scale"
SPLIT_WHITENING_COV = "whitening_cov"
SPLIT_WHITENING_SCALE = "whitening_scale"
SPLIT_CALIBRATION = "calibration"

ALL_SPLITS = (
    SPLIT_NULL,
    SPLIT_COSINE_SCALE,
    SPLIT_WHITENING_COV,
    SPLIT_WHITENING_SCALE,
    SPLIT_CALIBRATION,
)

DEFAULT_SPLIT_FRACTIONS = {
    "both": {
        SPLIT_NULL: 0.50,
        SPLIT_COSINE_SCALE: 0.15,
        SPLIT_WHITENING_COV: 0.15,
        SPLIT_WHITENING_SCALE: 0.10,
        SPLIT_CALIBRATION: 0.10,
    },
    "cosine": {
        SPLIT_NULL: 0.70,
        SPLIT_COSINE_SCALE: 0.20,
        SPLIT_CALIBRATION: 0.10,
    },
    "whitening": {
        SPLIT_NULL: 0.55,
        SPLIT_WHITENING_COV: 0.20,
        SPLIT_WHITENING_SCALE: 0.15,
        SPLIT_CALIBRATION: 0.10,
    },
}

STAGE_NULL = "null_train"
STAGE_COSINE_SCALE = "cosine_scale_train"
STAGE_WHITENING_FIT = "whitening_covariance_fit"
STAGE_WHITENING_SCALE = "whitening_scale_train"
STAGE_COSINE_CALIBRATE = "cosine_calibration"
STAGE_WHITENING_CALIBRATE = "whitening_calibration"

STOP_STAGE_CHOICES = (
    "split",
    STAGE_NULL,
    STAGE_COSINE_SCALE,
    STAGE_WHITENING_FIT,
    STAGE_WHITENING_SCALE,
    STAGE_COSINE_CALIBRATE,
    STAGE_WHITENING_CALIBRATE,
)

CHECKPOINT_RE = re.compile(r"^model\.(\d+)$")


def utc_now_iso() -> str:
    """Return the current UTC timestamp in timezone-aware ISO 8601 form."""
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def stage_label_to_split_name(label: str) -> str:
    """Convert an internal underscore-delimited split label for filenames."""
    return label.replace("_", "-")


def stable_seed(base_seed: int, *parts: str) -> int:
    """Build a deterministic seed that does not depend on Python hash randomization."""

    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(base_seed)).encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def score_methods(methods: str) -> Tuple[str, ...]:
    """Expand a CLI method selection into concrete score method names."""
    if methods == "both":
        return ("cosine", "whitening")
    if methods in {"cosine", "whitening"}:
        return (methods,)
    raise ValueError(f"Unsupported methods={methods!r}")


def active_split_names(methods: str) -> Tuple[str, ...]:
    """Return the ordered data splits required by selected score methods."""
    names: List[str] = [SPLIT_NULL]
    selected = set(score_methods(methods))
    if "cosine" in selected:
        names.append(SPLIT_COSINE_SCALE)
    if "whitening" in selected:
        names.extend([SPLIT_WHITENING_COV, SPLIT_WHITENING_SCALE])
    names.append(SPLIT_CALIBRATION)
    return tuple(names)


def parse_split_fractions(text: Optional[str], methods: str) -> Dict[str, float]:
    """Parse ``name=fraction`` pairs and normalize over active split names."""

    active = active_split_names(methods)
    if text is None or text.strip() == "":
        fractions = dict(DEFAULT_SPLIT_FRACTIONS[methods])
    else:
        fractions = {}
        for raw_part in text.split(","):
            part = raw_part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(
                    "--split_fractions must use named entries like "
                    "null=0.50,cosine_scale=0.15,whitening_cov=0.15,"
                    "whitening_scale=0.10,calibration=0.10"
                )
            key, value = part.split("=", 1)
            key = key.strip()
            if key not in ALL_SPLITS:
                raise ValueError(f"Unknown split name {key!r}; expected one of {ALL_SPLITS}")
            try:
                fractions[key] = float(value)
            except ValueError as exc:
                raise ValueError(f"Invalid fraction for split {key!r}: {value!r}") from exc

    missing = [name for name in active if name not in fractions]
    if missing:
        raise ValueError(f"Missing split fractions for active splits: {missing}")

    active_fractions = {name: float(fractions[name]) for name in active}
    for name, value in active_fractions.items():
        if value < 0.0:
            raise ValueError(f"Split fraction for {name} must be nonnegative, got {value}")
    total = sum(active_fractions.values())
    if total <= 0.0:
        raise ValueError("At least one active split fraction must be positive")
    return {name: value / total for name, value in active_fractions.items()}


def drop_null_split_and_renormalize(fractions: Mapping[str, float]) -> Dict[str, float]:
    """Use all records for downstream stages when a null checkpoint is reused."""

    downstream = {name: value for name, value in fractions.items() if name != SPLIT_NULL}
    total = sum(downstream.values())
    if total <= 0.0:
        raise ValueError(
            "Cannot reuse a null model with all downstream split fractions set to zero"
        )
    return {name: value / total for name, value in downstream.items()}


def load_bam_samples(path: Path) -> List[PretrainingSample]:
    """Return validated BAM samples from a pretraining JSON config."""
    return list(load_pretraining_config(path).samples)


def deterministic_split_samples(
    samples: Sequence[PretrainingSample],
    fractions: Mapping[str, float],
    seed: int,
) -> Dict[str, List[PretrainingSample]]:
    """Split samples by deterministic random shuffling only.

    The splitter deliberately avoids interpreting path names or directory
    conventions, keeping splits reproducible and disjoint for arbitrary users.
    """

    names = tuple(fractions.keys())
    splits: Dict[str, List[PretrainingSample]] = {name: [] for name in names}
    shuffled = list(samples)
    random.Random(stable_seed(seed, "bam-records")).shuffle(shuffled)
    n = len(shuffled)

    cumulative: List[Tuple[str, float]] = []
    running = 0.0
    for name in names:
        running += float(fractions[name])
        cumulative.append((name, running))

    start = 0
    for idx, (name, boundary) in enumerate(cumulative):
        end = n if idx == len(cumulative) - 1 else int(round(boundary * n))
        splits[name].extend(shuffled[start:end])
        start = end
    return splits


def write_split_configs(
    split_dir: Path,
    run_name: str,
    splits: Mapping[str, Sequence[PretrainingSample]],
) -> Dict[str, str]:
    """Write one validated pretraining JSON config per nonempty split.

    Split provenance remains in the pipeline manifest because strict input
    configs allow only schema-defined fields. The split directory is created
    if needed.

    Returns
    -------
    dict[str, str]
        Mapping from split name to its written configuration path.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for name, samples in splits.items():
        if not samples:
            continue
        out_path = split_dir / f"{run_name}.{stage_label_to_split_name(name)}.pretraining.json"
        write_json(out_path, PretrainingConfig(samples=tuple(samples)).to_dict())
        paths[name] = str(out_path)
    return paths


def latest_model_checkpoint(model_dir: Path) -> Optional[Path]:
    """Find the numerically latest ``model.N`` checkpoint in a directory."""
    if not model_dir.exists():
        return None
    best: Optional[Tuple[int, Path]] = None
    for child in model_dir.iterdir():
        match = CHECKPOINT_RE.match(child.name)
        if not match or not child.is_file():
            continue
        step_idx = int(match.group(1))
        if best is None or step_idx > best[0]:
            best = (step_idx, child)
    return best[1] if best is not None else None


def predicted_model_checkpoint(model_dir: Path, max_steps: int, checkpoint_batch_step: int) -> Optional[Path]:
    """Predict the last periodic checkpoint path produced by a dry run.

    Returns ``None`` when checkpointing is disabled or the requested stage
    ends before its first checkpoint boundary.
    """
    if checkpoint_batch_step <= 0:
        return None
    checkpoint_idx = int(max_steps) // int(checkpoint_batch_step)
    if checkpoint_idx <= 0:
        return None
    return model_dir / f"model.{checkpoint_idx}"


def require_existing_file(path: Optional[str], name: str) -> Path:
    """Validate a required artifact argument and return it as a path.

    Parameters
    ----------
    path : str or None
        User-supplied artifact path.
    name : str
        Argument label used in validation errors.
    """
    if not path:
        raise ValueError(f"{name} is required")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{name} does not exist or is not a file: {p}")
    return p


def warn_checkpoint_boundary(stage: str, max_steps: int, checkpoint_batch_step: int) -> None:
    """Warn when a stage ends after its last reusable checkpoint boundary."""
    if max_steps <= 0 or checkpoint_batch_step <= 0:
        return
    if max_steps % checkpoint_batch_step != 0:
        last_saved = (max_steps // checkpoint_batch_step) * checkpoint_batch_step
        print(
            f"[pipeline] Warning: {stage} max_steps={max_steps} is not divisible by "
            f"checkpoint_batch_step={checkpoint_batch_step}. The latest reusable checkpoint "
            f"will be from step {last_saved}, unless an older checkpoint is reused.",
            flush=True,
        )


def positive_step_limit(max_steps: int, stage: str) -> int:
    """Return a stage limit as an integer, requiring it to be positive."""
    max_steps = int(max_steps)
    if max_steps <= 0:
        raise ValueError(f"{stage} max_steps must be positive")
    return max_steps


def resolve_run_name(args: argparse.Namespace) -> str:
    """Choose an explicit run name or derive a timestamped artifact-based name."""
    if args.run_name:
        return args.run_name
    source = args.reuse_null_model or args.pretrain_load_path or "canonical_null"
    source_path = Path(source)
    if source_path.name.startswith("model.") and source_path.parent.name:
        base = source_path.parent.name
    else:
        base = source_path.stem or source_path.name or "canonical_null"
    safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-") or "canonical_null"
    timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_base}.{args.methods}.{timestamp}"


def ensure_single_process_orchestrator() -> None:
    """Reject distributed launch environments unsupported by this orchestrator."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        raise RuntimeError(
            "porefm zero-shot pipeline is an in-process orchestrator. Run it as a "
            "single Python process, not under torchrun. This script runs several stages "
            "sequentially in one interpreter and does not spawn a separate distributed "
            "rank group for its calibration stages. Launch "
            "torchrun --module porefm.cli.main zero-shot calibrate for "
            "distributed histogram calibration."
        )


def add_common_stage_args(ns: SimpleNamespace, args: argparse.Namespace) -> None:
    """Fill arguments expected by canonical_null_predictor.train_model."""

    ns.size = int(args.size)
    ns.preprocessing = getattr(args, 'preprocessing', None)
    ns.seq_type = args.seq_type
    ns.chunking_mode = resolve_chunking_mode(args)
    ns.kmer_dict_path = args.kmer_dict_path
    ns.eval_only = False
    ns.force_overwrite = bool(getattr(args, "force_overwrite", False))
    ns.disable_compile = bool(args.disable_compile)
    ns.min_read_length = int(args.min_read_length)
    ns.min_qscore = int(args.min_qscore)
    ns.lr = float(args.lr)
    ns.batch_size = int(args.batch_size)
    ns.scale_dropout = float(args.scale_dropout)
    ns.scale_kmer_embedding_dim = int(args.scale_kmer_embedding_dim)
    ns.scale_context_embedding_dim = int(args.scale_context_embedding_dim)
    ns.scale_context_channels = int(args.scale_context_channels)
    ns.scale_context_kernel_size = int(args.scale_context_kernel_size)
    ns.scale_context_dilations = tuple(int(value) for value in args.scale_context_dilations)
    ns.scale_context_expansion = int(args.scale_context_expansion)
    ns.scale_context_dropout = float(args.scale_context_dropout)
    ns.scale_h_null_feature_dim = int(args.scale_h_null_feature_dim)
    ns.disable_scale_sequence_context = bool(args.disable_scale_sequence_context)
    ns.disable_scale_h_null_features = bool(args.disable_scale_h_null_features)
    ns.use_base_quality_scale_features = bool(args.use_base_quality_scale_features)
    ns.scale_context_quality = bool(getattr(args, 'scale_context_quality', False))
    ns.scale_rna_structure_features = bool(getattr(args, 'scale_rna_structure_features', False))
    ns.null_rna_structure_features = bool(getattr(args, 'null_rna_structure_features', False))
    ns.null_sequence_context = bool(getattr(args, 'null_sequence_context', False))
    for name in ('embedding_dim', 'channels', 'kernel_size', 'dilations', 'expansion', 'dropout'):
        setattr(ns, f'null_context_{name}', getattr(args, f'null_context_{name}'))
    ns.checkpoint_batch_step = int(args.checkpoint_batch_step)
    ns.seed = int(args.seed)
    ns.reads_per_chunk = int(args.reads_per_chunk)
    ns.pore_type = args.pore_type
    ns.dataloader_workers = int(args.dataloader_workers)
    ns.prefetch_factor = int(args.prefetch_factor)
    ns.reference_match_kmer_size = int(args.reference_match_kmer_size)
    ns.log_every = int(args.log_every)
    ns.kmer_size = int(args.kmer_size)
    ns.scale_kmer_size = int(args.scale_kmer_size) if args.scale_kmer_size is not None else None
    ns.calibration_kmer_size = (
        int(args.calibration_kmer_size) if args.calibration_kmer_size is not None else None
    )
    ns.whitening_kmer_size = int(args.whitening_kmer_size)
    ns.max_scores_per_kmer = int(args.max_scores_per_kmer)
    ns.calibration_distribution_type = args.calibration_distribution_type
    ns.hist_min = float(args.hist_min)
    ns.hist_max = float(args.hist_max)
    ns.hist_bins = int(args.hist_bins)
    ns.quantile_body_count = int(args.quantile_body_count)
    ns.quantile_tail_count = int(args.quantile_tail_count)
    ns.quantile_tail_min_prob = float(args.quantile_tail_min_prob)
    ns.whitening_artifact = ""
    ns.allow_whitening_checkpoint_mismatch = bool(args.allow_whitening_checkpoint_mismatch)
    ns.whitening_output = ""
    ns.covariance_mode = args.covariance_mode
    ns.shrinkage_mode = args.shrinkage_mode
    ns.shrinkage = float(args.shrinkage)
    ns.whitening_parent_shrinkage_strength = float(args.whitening_parent_shrinkage_strength)
    ns.whitening_covariance_buffer_rows = int(args.whitening_covariance_buffer_rows)
    ns.relative_eigenvalue_floor = float(args.relative_eigenvalue_floor)
    ns.max_condition_number = float(args.max_condition_number)
    ns.max_samples_per_base = int(args.max_samples_per_base)
    ns.max_tokens_per_read = int(args.max_tokens_per_read)
    ns.whitening_token_sample_rate = float(args.whitening_token_sample_rate)
    ns.sampling_seed = int(args.sampling_seed)
    ns.robust_fit = bool(args.robust_fit)
    ns.robust_cutoff = float(args.robust_cutoff)
    ns.robust_quantile = float(args.robust_quantile)
    ns.robust_quantile_sample_size = int(args.robust_quantile_sample_size)
    ns.robust_eps = float(args.robust_eps)
    ns.norm_eps = float(args.norm_eps)
    ns.score_log_eps = float(args.score_log_eps)
    ns.whitening_score_chunk_size = int(args.whitening_score_chunk_size)
    ns.include_raw_covariance = bool(args.include_raw_covariance)
    ns.require_valid_kmer_for_whitening_fit = bool(args.require_valid_kmer_for_whitening_fit)
    ns.num_loss_trim = [int(args.num_loss_trim[0]), int(args.num_loss_trim[1])]
    ns.calibration_output = ""


def canonical_stage_args(
    args: argparse.Namespace,
    *,
    mode: str,
    bam_config: str,
    max_steps: int,
    model_load_path: Optional[str] = None,
    model_save_path: Optional[str] = None,
    pretrain_load_path: Optional[str] = None,
    scale_score_type: str = "cosine",
    whitening_artifact: Optional[str] = None,
    whitening_output: Optional[str] = None,
    calibration_output: Optional[str] = None,
    batch_size: Optional[int] = None,
    reads_per_chunk: Optional[int] = None,
    step: Optional[int] = None,
    prefetch_factor: Optional[int] = None,
    log_every: Optional[int] = None,
) -> SimpleNamespace:
    """Build the argument namespace consumed by one predictor stage.

    Shared data, optimization, calibration, and whitening settings are copied
    from ``args``. Keyword parameters identify the stage mode and its artifacts;
    optional throughput values override the shared defaults for that stage.

    Returns
    -------
    types.SimpleNamespace
        Complete arguments for
        :func:`porefm.workflows.canonical_null.predictor.train_model`.
    """
    ns = SimpleNamespace()
    add_common_stage_args(ns, args)
    ns.mode = mode
    requested_step = step if step is not None else args.step
    ns.chunk_step_source = "explicit" if requested_step is not None else "automatic"
    if requested_step is None and (mode in {'train-scale', 'calibrate'} or ns.null_rna_structure_features):
        # The required overlap depends on the null/scale checkpoint's serialized
        # context radius and the backbone's signal/base ratio. Defer automatic
        # resolution until train_model has loaded both configurations.
        ns.step = None
    else:
        ns.step = resolve_downstream_step(
            ns.size,
            ns.seq_type,
            requested_step,
            context=f"zero-shot pipeline stage {mode}",
        )
    ns.bam_config = bam_config
    ns.model_load_path = model_load_path
    ns.model_save_path = model_save_path or ""
    ns.pretrain_load_path = pretrain_load_path or ""
    ns.max_steps = int(max_steps)
    ns.scale_score_type = scale_score_type
    if whitening_artifact:
        ns.whitening_artifact = whitening_artifact
    if whitening_output:
        ns.whitening_output = whitening_output
    if calibration_output:
        ns.calibration_output = calibration_output
    if batch_size is not None:
        ns.batch_size = int(batch_size)
    if reads_per_chunk is not None:
        ns.reads_per_chunk = int(reads_per_chunk)
    if prefetch_factor is not None:
        ns.prefetch_factor = int(prefetch_factor)
    if log_every is not None:
        ns.log_every = int(log_every)
    return ns


def namespace_to_jsonable(ns: SimpleNamespace) -> Dict[str, Any]:
    """Convert a stage namespace to stable JSON-compatible scalar/list values."""
    out: Dict[str, Any] = {}
    for key, value in sorted(vars(ns).items()):
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = list(value)
        else:
            out[key] = str(value)
    return out


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON file using a sibling temporary path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def cleanup_after_stage() -> None:
    """Collect Python garbage and release unused CUDA cache after a stage."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def call_train_model(stage: str, stage_args: SimpleNamespace) -> None:
    """Import lazily so split helpers can be unit-tested without model deps."""

    from porefm.workflows.canonical_null.predictor import train_model

    print(f"[pipeline] Starting stage {stage} at {utc_now_iso()}", flush=True)
    train_model(stage_args)
    cleanup_after_stage()
    print(f"[pipeline] Finished stage {stage} at {utc_now_iso()}", flush=True)


def stage_arg_dir(run_dir: Path) -> Path:
    """Return the directory used for serialized per-stage arguments."""
    return run_dir / "stage_args"


def record_stage(
    manifest: MutableMapping[str, Any],
    manifest_path: Path,
    *,
    stage: str,
    status: str,
    output: Optional[str] = None,
    args_path: Optional[str] = None,
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Append a stage event to the manifest and persist it immediately.

    A stage may have multiple events across planned, reused, skipped, failed,
    or completed execution paths. Optional timestamps, output, argument path,
    and error text are included only when supplied.
    """
    entry: Dict[str, Any] = {
        "stage": stage,
        "status": status,
    }
    if output is not None:
        entry["output"] = output
    if args_path is not None:
        entry["args_path"] = args_path
    if started_at is not None:
        entry["started_at"] = started_at
    if ended_at is not None:
        entry["ended_at"] = ended_at
    if error is not None:
        entry["error"] = error
    manifest.setdefault("stages", []).append(entry)
    manifest["updated_at"] = utc_now_iso()
    write_json(manifest_path, manifest)


def execute_stage(
    args: argparse.Namespace,
    manifest: MutableMapping[str, Any],
    manifest_path: Path,
    *,
    stage: str,
    stage_args: SimpleNamespace,
    expected_output: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Plan, reuse, or execute one canonical-null pipeline stage.

    Stage arguments are always serialized first. In dry-run mode the expected
    artifact is recorded without running training. With ``skip_existing``, an
    existing explicit artifact or latest checkpoint is reused. Otherwise the
    predictor entry point is invoked and its output is validated.

    Returns
    -------
    pathlib.Path or None
        Resolved artifact/checkpoint path, when the stage has an output.

    Raises
    ------
    FileNotFoundError
        If execution finishes without producing its required artifact.
    """
    args_path = stage_arg_dir(Path(manifest["run_dir"])) / f"{stage}.args.json"
    write_json(args_path, namespace_to_jsonable(stage_args))

    if args.dry_run:
        output_path = expected_output
        if output_path is None and checkpoint_dir is not None:
            output_path = predicted_model_checkpoint(
                checkpoint_dir,
                int(stage_args.max_steps),
                int(stage_args.checkpoint_batch_step),
            )
            if output_path is None:
                raise ValueError(
                    f"{stage} dry-run cannot predict a reusable checkpoint because "
                    f"max_steps={stage_args.max_steps} and checkpoint_batch_step="
                    f"{stage_args.checkpoint_batch_step} do not produce a model.N checkpoint."
                )
        output = str(output_path or "")
        record_stage(
            manifest,
            manifest_path,
            stage=stage,
            status="planned",
            output=output or None,
            args_path=str(args_path),
        )
        return output_path

    if args.skip_existing:
        if expected_output is not None and expected_output.is_file():
            record_stage(
                manifest,
                manifest_path,
                stage=stage,
                status="skipped_existing",
                output=str(expected_output),
                args_path=str(args_path),
            )
            return expected_output
        if checkpoint_dir is not None:
            latest = latest_model_checkpoint(checkpoint_dir)
            if latest is not None:
                record_stage(
                    manifest,
                    manifest_path,
                    stage=stage,
                    status="skipped_existing",
                    output=str(latest),
                    args_path=str(args_path),
                )
                return latest

    started_at = utc_now_iso()
    try:
        call_train_model(stage, stage_args)
    except Exception as exc:
        record_stage(
            manifest,
            manifest_path,
            stage=stage,
            status="failed",
            args_path=str(args_path),
            started_at=started_at,
            ended_at=utc_now_iso(),
            error=repr(exc),
        )
        raise

    output_path: Optional[Path] = expected_output
    if checkpoint_dir is not None:
        output_path = latest_model_checkpoint(checkpoint_dir)
        if output_path is None:
            raise FileNotFoundError(
                f"{stage} finished but no model.N checkpoint was found in {checkpoint_dir}. "
                "Decrease --checkpoint_batch_step or make max_steps land on a checkpoint boundary."
            )
    if output_path is not None and not output_path.exists():
        raise FileNotFoundError(f"{stage} finished but expected output was not created: {output_path}")

    record_stage(
        manifest,
        manifest_path,
        stage=stage,
        status="completed",
        output=str(output_path) if output_path is not None else None,
        args_path=str(args_path),
        started_at=started_at,
        ended_at=utc_now_iso(),
    )
    return output_path


def default_output_paths(args: argparse.Namespace, run_dir: Path) -> Dict[str, Path]:
    """Construct conventional model, whitening, and calibration output paths."""
    scale_kmer_size = args.scale_kmer_size if args.scale_kmer_size is not None else args.kmer_size
    calibration_kmer_size = (
        args.calibration_kmer_size
        if args.calibration_kmer_size is not None
        else scale_kmer_size
    )
    kmer_tag = f"kmer{int(calibration_kmer_size)}"
    whitening_tag = f"kmer{int(args.whitening_kmer_size)}"
    dist = args.calibration_distribution_type
    return {
        "null_model_dir": run_dir / "null_model",
        "cosine_scale_model_dir": run_dir / "cosine_scale_model",
        "whitening_artifact": run_dir / "whitening" / f"{whitening_tag}.{args.covariance_mode}.whitening.pt",
        "whitening_scale_model_dir": run_dir / "whitening_scale_model",
        "cosine_calibration": run_dir / "calibrations" / f"{kmer_tag}_{dist}.z_cos.pt",
        "whitening_calibration": run_dir / "calibrations" / f"{kmer_tag}_{dist}.z_whiten.pt",
    }


def resolve_output_paths(args: argparse.Namespace, run_dir: Path) -> Dict[str, Path]:
    """Apply explicit CLI output overrides to the conventional stage paths."""
    paths = default_output_paths(args, run_dir)
    overrides = {
        "null_model_dir": args.null_model_dir,
        "cosine_scale_model_dir": args.cosine_scale_model_dir,
        "whitening_artifact": args.whitening_artifact_output,
        "whitening_scale_model_dir": args.whitening_scale_model_dir,
        "cosine_calibration": args.cosine_calibration_output,
        "whitening_calibration": args.whitening_calibration_output,
    }
    for key, value in overrides.items():
        if value:
            paths[key] = Path(value)
    return paths


def minimum_bams_by_split(args: argparse.Namespace) -> Dict[str, int]:
    """Resolve per-split minimum BAM counts with the shared value as fallback."""
    return {
        SPLIT_NULL: int(args.min_null_bams if args.min_null_bams is not None else args.min_bams_per_split),
        SPLIT_COSINE_SCALE: int(
            args.min_cosine_scale_bams
            if args.min_cosine_scale_bams is not None
            else args.min_bams_per_split
        ),
        SPLIT_WHITENING_COV: int(
            args.min_whitening_cov_bams
            if args.min_whitening_cov_bams is not None
            else args.min_bams_per_split
        ),
        SPLIT_WHITENING_SCALE: int(
            args.min_whitening_scale_bams
            if args.min_whitening_scale_bams is not None
            else args.min_bams_per_split
        ),
        SPLIT_CALIBRATION: int(
            args.min_calibration_bams
            if args.min_calibration_bams is not None
            else args.min_bams_per_split
        ),
    }


def validate_split_sizes(
    splits: Mapping[str, Sequence[PretrainingSample]],
    args: argparse.Namespace,
    *,
    train_null: bool,
) -> None:
    """Require enough BAM samples for every stage that will actually run.

    Reused model or whitening artifacts remove their associated training split
    from validation. Calibration is always required, and the null split is
    required only when a new null model will be trained.
    """
    mins = minimum_bams_by_split(args)
    selected = set(score_methods(args.methods))
    required: List[str] = []
    if train_null:
        required.append(SPLIT_NULL)
    if "cosine" in selected and not args.reuse_cosine_scale_model:
        required.append(SPLIT_COSINE_SCALE)
    if "whitening" in selected:
        if not args.reuse_whitening_artifact:
            required.append(SPLIT_WHITENING_COV)
        if not args.reuse_whitening_scale_model:
            required.append(SPLIT_WHITENING_SCALE)
    required.append(SPLIT_CALIBRATION)

    for name in required:
        observed = len(splits.get(name, ()))
        minimum = int(mins[name])
        if observed < minimum:
            raise ValueError(
                f"Split {name!r} has {observed} BAM samples, below required minimum {minimum}. "
                "Adjust --split_fractions or the --min_*_bams threshold."
            )


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the selected canonical-null stages and maintain a run manifest.

    The input pretraining samples are deterministically partitioned into
    disjoint stage inputs. The null predictor, score standardizers, whitening
    artifact, and calibrations are then trained or reused in dependency order.
    Manifest updates are written throughout so planned, completed, reused, and
    failed stages remain inspectable after interruption.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments produced by :func:`build_parser`.

    Returns
    -------
    dict[str, Any]
        In-memory form of the final or deliberately stopped pipeline manifest.
    """
    resolve_cli_preprocessing(args, checkpoint_paths=[
        args.pretrain_load_path, args.reuse_null_model,
        args.reuse_cosine_scale_model, args.reuse_whitening_scale_model,
    ])
    if getattr(args, 'scale_rna_structure_features', False) and args.seq_type != 'rna':
        raise ValueError('--scale_rna_structure_features requires --seq_type rna')
    if getattr(args, 'null_rna_structure_features', False) and args.seq_type != 'rna':
        raise ValueError('--null_rna_structure_features requires --seq_type rna')
    ensure_single_process_orchestrator()
    if (
        float(args.whitening_token_sample_rate) <= 0.0
        or float(args.whitening_token_sample_rate) > 1.0
        or not math.isfinite(float(args.whitening_token_sample_rate))
    ):
        raise ValueError(
            f"--whitening_token_sample_rate must be in (0, 1], got {args.whitening_token_sample_rate}"
        )

    if not args.reuse_null_model and not args.pretrain_load_path:
        raise ValueError("--pretrain_load_path is required unless --reuse_null_model is set")
    if args.reuse_null_model:
        require_existing_file(args.reuse_null_model, "--reuse_null_model")
    if args.reuse_cosine_scale_model:
        require_existing_file(args.reuse_cosine_scale_model, "--reuse_cosine_scale_model")
    if args.reuse_whitening_artifact:
        require_existing_file(args.reuse_whitening_artifact, "--reuse_whitening_artifact")
    if args.reuse_whitening_scale_model:
        require_existing_file(args.reuse_whitening_scale_model, "--reuse_whitening_scale_model")

    fractions = parse_split_fractions(args.split_fractions, args.methods)
    if args.reuse_null_model and not args.split_fractions:
        fractions = drop_null_split_and_renormalize(fractions)
    input_bam_config = Path(args.bam_config)
    samples = load_bam_samples(input_bam_config)

    run_name = resolve_run_name(args)
    output_dir = Path(args.output_dir)
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    split_dir = Path(args.split_dir) if args.split_dir else run_dir / "splits"
    splits = deterministic_split_samples(samples, fractions, int(args.split_seed))
    train_null = not bool(args.reuse_null_model)
    validate_split_sizes(splits, args, train_null=train_null)
    split_paths = write_split_configs(split_dir, run_name, splits)

    paths = resolve_output_paths(args, run_dir)
    for key, path in paths.items():
        if key.endswith("_dir"):
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "pipeline_manifest.json"
    manifest: Dict[str, Any] = {
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "argv": sys.argv,
        "methods": args.methods,
        "input": {
            "bam_config": str(input_bam_config),
            "pretrain_load_path": args.pretrain_load_path or "",
            "reuse_null_model": args.reuse_null_model or "",
            "preprocessing": args.preprocessing,
            "seq_type": args.seq_type,
            "kmer_dict_path": args.kmer_dict_path,
        },
        "split": {
            "seed": int(args.split_seed),
            "fractions": fractions,
            "paths": split_paths,
            "counts": {name: len(values) for name, values in splits.items()},
        },
        "outputs": {key: str(path) for key, path in paths.items()},
        "stages": [],
    }
    write_json(manifest_path, manifest)

    print(f"[pipeline] Run directory: {run_dir}", flush=True)
    print(f"[pipeline] Manifest: {manifest_path}", flush=True)
    print(f"[pipeline] Split counts: {manifest['split']['counts']}", flush=True)
    if args.stop_after_stage == "split":
        record_stage(manifest, manifest_path, stage="split", status="completed")
        return manifest

    selected = set(score_methods(args.methods))
    null_checkpoint: Optional[Path]

    if args.reuse_null_model:
        null_checkpoint = Path(args.reuse_null_model)
        record_stage(
            manifest,
            manifest_path,
            stage=STAGE_NULL,
            status="reused",
            output=str(null_checkpoint),
        )
    else:
        null_max_steps = positive_step_limit(args.null_max_steps, STAGE_NULL)
        warn_checkpoint_boundary(STAGE_NULL, null_max_steps, args.checkpoint_batch_step)
        null_args = canonical_stage_args(
            args,
            mode="train",
            bam_config=split_paths[SPLIT_NULL],
            pretrain_load_path=args.pretrain_load_path,
            model_save_path=str(paths["null_model_dir"]),
            max_steps=null_max_steps,
        )
        null_checkpoint = execute_stage(
            args,
            manifest,
            manifest_path,
            stage=STAGE_NULL,
            stage_args=null_args,
            checkpoint_dir=paths["null_model_dir"],
        )
        if args.stop_after_stage == STAGE_NULL:
            return manifest

    if null_checkpoint is None:
        raise RuntimeError("No null checkpoint is available for downstream stages")

    cosine_scale_checkpoint: Optional[Path] = None
    if "cosine" in selected:
        if args.reuse_cosine_scale_model:
            cosine_scale_checkpoint = Path(args.reuse_cosine_scale_model)
            record_stage(
                manifest,
                manifest_path,
                stage=STAGE_COSINE_SCALE,
                status="reused",
                output=str(cosine_scale_checkpoint),
            )
        else:
            cosine_max_steps = positive_step_limit(args.cosine_scale_max_steps, STAGE_COSINE_SCALE)
            warn_checkpoint_boundary(STAGE_COSINE_SCALE, cosine_max_steps, args.checkpoint_batch_step)
            cosine_args = canonical_stage_args(
                args,
                mode="train-scale",
                bam_config=split_paths[SPLIT_COSINE_SCALE],
                model_load_path=str(null_checkpoint),
                model_save_path=str(paths["cosine_scale_model_dir"]),
                max_steps=cosine_max_steps,
                scale_score_type="cosine",
            )
            cosine_scale_checkpoint = execute_stage(
                args,
                manifest,
                manifest_path,
                stage=STAGE_COSINE_SCALE,
                stage_args=cosine_args,
                checkpoint_dir=paths["cosine_scale_model_dir"],
            )
        if args.stop_after_stage == STAGE_COSINE_SCALE:
            return manifest

    whitening_artifact: Optional[Path] = None
    whitening_scale_checkpoint: Optional[Path] = None
    if "whitening" in selected:
        if args.reuse_whitening_artifact:
            whitening_artifact = Path(args.reuse_whitening_artifact)
            record_stage(
                manifest,
                manifest_path,
                stage=STAGE_WHITENING_FIT,
                status="reused",
                output=str(whitening_artifact),
            )
        else:
            whitening_fit_max = positive_step_limit(args.whitening_fit_max_batches, STAGE_WHITENING_FIT)
            fit_args = canonical_stage_args(
                args,
                mode="fit-whitening",
                bam_config=split_paths[SPLIT_WHITENING_COV],
                model_load_path=str(null_checkpoint),
                whitening_output=str(paths["whitening_artifact"]),
                max_steps=whitening_fit_max,
                batch_size=args.whitening_fit_batch_size,
                reads_per_chunk=args.whitening_fit_reads_per_chunk,
                log_every=args.whitening_fit_log_every,
            )
            whitening_artifact = execute_stage(
                args,
                manifest,
                manifest_path,
                stage=STAGE_WHITENING_FIT,
                stage_args=fit_args,
                expected_output=paths["whitening_artifact"],
            )
        if args.stop_after_stage == STAGE_WHITENING_FIT:
            return manifest
        if whitening_artifact is None:
            raise RuntimeError("No whitening artifact is available for whitening scale/calibration")

        if args.reuse_whitening_scale_model:
            whitening_scale_checkpoint = Path(args.reuse_whitening_scale_model)
            record_stage(
                manifest,
                manifest_path,
                stage=STAGE_WHITENING_SCALE,
                status="reused",
                output=str(whitening_scale_checkpoint),
            )
        else:
            whitening_scale_max = positive_step_limit(args.whitening_scale_max_steps, STAGE_WHITENING_SCALE)
            warn_checkpoint_boundary(STAGE_WHITENING_SCALE, whitening_scale_max, args.checkpoint_batch_step)
            whitening_scale_args = canonical_stage_args(
                args,
                mode="train-scale",
                bam_config=split_paths[SPLIT_WHITENING_SCALE],
                model_load_path=str(null_checkpoint),
                model_save_path=str(paths["whitening_scale_model_dir"]),
                max_steps=whitening_scale_max,
                scale_score_type="whitening",
                whitening_artifact=str(whitening_artifact),
            )
            whitening_scale_checkpoint = execute_stage(
                args,
                manifest,
                manifest_path,
                stage=STAGE_WHITENING_SCALE,
                stage_args=whitening_scale_args,
                checkpoint_dir=paths["whitening_scale_model_dir"],
            )
        if args.stop_after_stage == STAGE_WHITENING_SCALE:
            return manifest

    calibration_step = args.calibration_step if args.calibration_step is not None else args.step
    calibration_max = positive_step_limit(args.calibration_max_batches, "calibration")

    if "cosine" in selected:
        if cosine_scale_checkpoint is None:
            raise RuntimeError("No cosine scale checkpoint is available for calibration")
        cosine_cal_args = canonical_stage_args(
            args,
            mode="calibrate",
            bam_config=split_paths[SPLIT_CALIBRATION],
            model_load_path=str(cosine_scale_checkpoint),
            calibration_output=str(paths["cosine_calibration"]),
            max_steps=calibration_max,
            scale_score_type="cosine",
            batch_size=args.calibration_batch_size,
            reads_per_chunk=args.calibration_reads_per_chunk,
            step=calibration_step,
            prefetch_factor=args.calibration_prefetch_factor,
            log_every=args.calibration_log_every,
        )
        execute_stage(
            args,
            manifest,
            manifest_path,
            stage=STAGE_COSINE_CALIBRATE,
            stage_args=cosine_cal_args,
            expected_output=paths["cosine_calibration"],
        )
        if args.stop_after_stage == STAGE_COSINE_CALIBRATE:
            return manifest

    if "whitening" in selected:
        if whitening_scale_checkpoint is None or whitening_artifact is None:
            raise RuntimeError("No whitening scale checkpoint/artifact is available for calibration")
        whitening_cal_args = canonical_stage_args(
            args,
            mode="calibrate",
            bam_config=split_paths[SPLIT_CALIBRATION],
            model_load_path=str(whitening_scale_checkpoint),
            calibration_output=str(paths["whitening_calibration"]),
            max_steps=calibration_max,
            scale_score_type="whitening",
            whitening_artifact=str(whitening_artifact),
            batch_size=args.calibration_batch_size,
            reads_per_chunk=args.calibration_reads_per_chunk,
            step=calibration_step,
            prefetch_factor=args.calibration_prefetch_factor,
            log_every=args.calibration_log_every,
        )
        execute_stage(
            args,
            manifest,
            manifest_path,
            stage=STAGE_WHITENING_CALIBRATE,
            stage_args=whitening_cal_args,
            expected_output=paths["whitening_calibration"],
        )

    manifest["completed_at"] = utc_now_iso()
    manifest["updated_at"] = manifest["completed_at"]
    write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the end-to-end canonical-null pipeline command-line parser."""
    parser = PoreFMArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run canonical-null training, whitening fitting, scalar standardizer "
            "training, and empirical calibration from one Python command."
        ),
    )

    core = parser.add_argument_group("Core inputs")
    core.add_argument("--bam_config", required=True, help="Versioned pretraining JSON config to split deterministically by sample.")
    core.add_argument("--pretrain_load_path", default="", help="Starting pretrained model for null training.")
    core.add_argument(
        "--reuse_null_model",
        default="",
        help="Existing canonical-null checkpoint. If set, null training is skipped.",
    )
    core.add_argument("--output_dir", required=True, help="Parent directory for this pipeline run.")
    core.add_argument("--run_name", default="", help="Run subdirectory name. Defaults to model name plus timestamp.")
    core.add_argument("--methods", choices=["both", "cosine", "whitening"], default="both")
    core.add_argument("--seq_type", choices=["dna", "rna"], required=True)
    add_chunking_arguments(core)
    core.add_argument("--kmer_dict_path", required=True)
    core.add_argument("--size", type=int, default=2048)
    core.add_argument("--step", type=int, default=None)
    core.add_argument(
        "--calibration_step",
        type=int,
        default=None,
        help="Optional step override for held-out calibration. If omitted, uses --step.",
    )
    core.add_argument("--dry_run", action="store_true", help="Write splits/manifest/stage args but do not train.")
    core.add_argument("--skip_existing", action="store_true", help="Reuse existing stage outputs in the run directory.")
    from porefm.common.run_outputs import add_output_safety_argument
    add_output_safety_argument(core)
    core.add_argument("--disable_compile", action="store_true", help="Disable model compilation in each stage.")

    split = parser.add_argument_group("Deterministic BAM splitting")
    split.add_argument(
        "--split_fractions",
        default="",
        help=(
            "Comma-separated named fractions. Example for both methods: "
            "null=0.50,cosine_scale=0.15,whitening_cov=0.15,"
            "whitening_scale=0.10,calibration=0.10"
        ),
    )
    split.add_argument("--split_seed", type=int, default=0)
    split.add_argument("--split_dir", default="", help="Optional output directory for generated pretraining JSON configs.")
    split.add_argument("--min_bams_per_split", type=int, default=1)
    split.add_argument("--min_null_bams", type=int, default=None)
    split.add_argument("--min_cosine_scale_bams", type=int, default=None)
    split.add_argument("--min_whitening_cov_bams", type=int, default=None)
    split.add_argument("--min_whitening_scale_bams", type=int, default=None)
    split.add_argument("--min_calibration_bams", type=int, default=None)

    training = parser.add_argument_group("Shared training/data settings")
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--batch_size", type=int, default=512)
    training.add_argument("--checkpoint_batch_step", type=int, default=10000)
    training.add_argument("--reads_per_chunk", type=int, default=100)
    training.add_argument("--dataloader_workers", type=int, default=8)
    training.add_argument("--prefetch_factor", type=int, default=2)
    training.add_argument(
        "--reference_match_kmer_size",
        type=int,
        default=0,
        help=(
            "Require this centered read k-mer to match the reference exactly in "
            "whitening/scale/calibration stages; zero disables the filter."
        ),
    )
    training.add_argument("--seed", type=int, default=0, help="Data-loader/training seed passed to each stage.")
    training.add_argument("--min_read_length", type=int, default=100)
    training.add_argument("--min_qscore", type=int, default=10)
    training.add_argument("--pore_type", type=str, default="R10/RNA004")
    training.add_argument("--num_loss_trim", nargs=2, type=int, default=[8, 16])
    training.add_argument(
        "--kmer_size",
        type=int,
        default=9,
        help="Legacy default for scale/calibration context size.",
    )
    training.add_argument("--scale_kmer_size", type=int, default=None)
    training.add_argument(
        "--calibration_kmer_size",
        type=int,
        default=None,
        help="Defaults to --scale_kmer_size (or the legacy --kmer_size fallback).",
    )
    training.add_argument("--whitening_kmer_size", type=int, default=1)
    training.add_argument("--scale_dropout", type=float, default=0.0)
    training.add_argument("--scale_kmer_embedding_dim", type=int, default=64)
    training.add_argument("--scale_context_embedding_dim", type=int, default=32)
    training.add_argument("--scale_context_channels", type=int, default=128)
    training.add_argument("--scale_context_kernel_size", type=int, default=5)
    training.add_argument(
        "--scale_context_dilations",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
    )
    training.add_argument("--scale_context_expansion", type=int, default=4)
    training.add_argument("--scale_context_dropout", type=float, default=0.05)
    training.add_argument("--scale_h_null_feature_dim", type=int, default=128)
    training.add_argument("--disable_scale_sequence_context", action="store_true")
    training.add_argument("--disable_scale_h_null_features", action="store_true")
    training.add_argument("--use_base_quality_scale_features", action="store_true")
    training.add_argument("--scale_context_quality", action="store_true",
                          help="Route scaled per-base quality through the scaler CNN; requires quality features.")
    training.add_argument("--scale_rna_structure_features", action="store_true",
                          help="Add RNALfold structure inputs to RNA scale models; requires rnalocal_ext.")
    training.add_argument('--null_sequence_context', action='store_true',
                          help='Enable the independent sequence CNN for a new null predictor.')
    training.add_argument('--null_rna_structure_features', action='store_true',
                          help='Add RNALfold inputs before the null CNN; requires RNA and --null_sequence_context for a new null.')
    training.add_argument('--null_context_embedding_dim', type=int, default=32)
    training.add_argument('--null_context_channels', type=int, default=128)
    training.add_argument('--null_context_kernel_size', type=int, default=5)
    training.add_argument('--null_context_dilations', nargs='+', type=int, default=[1, 1, 2, 4, 8, 16])
    training.add_argument('--null_context_expansion', type=int, default=4)
    training.add_argument('--null_context_dropout', type=float, default=0.05)
    training.add_argument(
        "--stop_after_stage",
        choices=STOP_STAGE_CHOICES,
        default="",
        help="Run through this stage, then stop. Useful for checking a split or partial run.",
    )

    limits = parser.add_argument_group("Stage limits")
    limits.add_argument("--null_max_steps", type=int, default=500000)
    limits.add_argument("--cosine_scale_max_steps", type=int, default=500000)
    limits.add_argument("--whitening_scale_max_steps", type=int, default=500000)
    limits.add_argument("--whitening_fit_max_batches", type=int, default=1000000)
    limits.add_argument("--calibration_max_batches", type=int, default=1000000)

    stage = parser.add_argument_group("Stage-specific throughput")
    stage.add_argument("--whitening_fit_batch_size", type=int, default=None)
    stage.add_argument("--whitening_fit_reads_per_chunk", type=int, default=None)
    stage.add_argument("--whitening_fit_log_every", type=int, default=None)
    stage.add_argument("--calibration_batch_size", type=int, default=128)
    stage.add_argument("--calibration_reads_per_chunk", type=int, default=50)
    stage.add_argument("--calibration_prefetch_factor", type=int, default=4)
    stage.add_argument("--calibration_log_every", type=int, default=1000)
    stage.add_argument("--log_every", type=int, default=100)

    reuse = parser.add_argument_group("Reuse existing downstream artifacts")
    reuse.add_argument("--reuse_cosine_scale_model", default="", help="Existing cosine scale checkpoint.")
    reuse.add_argument("--reuse_whitening_artifact", default="", help="Existing whitening artifact.")
    reuse.add_argument("--reuse_whitening_scale_model", default="", help="Existing whitening scale checkpoint.")

    outputs = parser.add_argument_group("Output overrides")
    outputs.add_argument("--null_model_dir", default="")
    outputs.add_argument("--cosine_scale_model_dir", default="")
    outputs.add_argument("--whitening_artifact_output", default="")
    outputs.add_argument("--whitening_scale_model_dir", default="")
    outputs.add_argument("--cosine_calibration_output", default="")
    outputs.add_argument("--whitening_calibration_output", default="")

    calibration = parser.add_argument_group("Empirical calibration")
    calibration.add_argument("--calibration_distribution_type", choices=["reservoir", "histogram", "quantile"], default="histogram")
    calibration.add_argument("--max_scores_per_kmer", type=int, default=20000)
    calibration.add_argument("--hist_min", type=float, default=-8.0)
    calibration.add_argument("--hist_max", type=float, default=8.0)
    calibration.add_argument("--hist_bins", type=int, default=512)
    calibration.add_argument("--quantile_body_count", type=int, default=201)
    calibration.add_argument("--quantile_tail_count", type=int, default=200)
    calibration.add_argument("--quantile_tail_min_prob", type=float, default=1e-4)

    whitening = parser.add_argument_group("Full-space whitening")
    whitening.add_argument("--covariance_mode", choices=["full", "diagonal", "isotropic"], default="full")
    whitening.add_argument("--shrinkage_mode", choices=["fixed", "oas"], default="fixed")
    whitening.add_argument("--shrinkage", type=float, default=0.03)
    whitening.add_argument("--whitening_parent_shrinkage_strength", type=float, default=0.0)
    whitening.add_argument("--whitening_covariance_buffer_rows", type=int, default=128)
    whitening.add_argument("--relative_eigenvalue_floor", type=float, default=1e-3)
    whitening.add_argument("--max_condition_number", type=float, default=1e4)
    whitening.add_argument("--max_samples_per_base", type=int, default=0)
    whitening.add_argument("--max_tokens_per_read", type=int, default=0)
    whitening.add_argument("--whitening_token_sample_rate", type=float, default=1.0)
    whitening.add_argument("--sampling_seed", type=int, default=0)
    whitening.add_argument("--robust_fit", dest="robust_fit", default=True, action="store_true")
    whitening.add_argument("--disable_robust_fit", dest="robust_fit", action="store_false")
    whitening.add_argument("--robust_cutoff", type=float, default=0.0)
    whitening.add_argument("--robust_quantile", type=float, default=0.995)
    whitening.add_argument("--robust_quantile_sample_size", type=int, default=200000)
    whitening.add_argument("--robust_eps", type=float, default=1e-12)
    whitening.add_argument("--norm_eps", type=float, default=1e-8)
    whitening.add_argument("--score_log_eps", type=float, default=1e-8)
    whitening.add_argument("--whitening_score_chunk_size", type=int, default=262144)
    whitening.add_argument("--include_raw_covariance", action="store_true")
    whitening.add_argument(
        "--require_valid_kmer_for_whitening_fit",
        dest="require_valid_kmer_for_whitening_fit",
        default=True,
        action="store_true",
    )
    whitening.add_argument(
        "--allow_invalid_kmer_for_whitening_fit",
        dest="require_valid_kmer_for_whitening_fit",
        action="store_false",
    )
    whitening.add_argument("--allow_whitening_checkpoint_mismatch", action="store_true")

    add_preprocessing_arguments(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments, run the pipeline, and return a successful exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    started = time.time()
    print(f"[pipeline] Command: python {' '.join(sys.argv)}", flush=True)
    manifest = run_pipeline(args)
    print(
        f"[pipeline] Pipeline metadata written to {Path(manifest['run_dir']) / 'pipeline_manifest.json'}",
        flush=True,
    )
    print(f"[pipeline] Elapsed seconds: {time.time() - started:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
