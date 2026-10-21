"""Versioned checkpoint manifests, state partitioning, and Git provenance."""

from porefm.preprocessing import attach_preprocessing
from dataclasses import asdict, is_dataclass
from pathlib import Path
import subprocess
from typing import Any, Dict, Optional, Tuple


REPOSITORY_PROVENANCE_KEY = "repository_provenance"

BACKBONE_PREFIXES = (
    "signal_projector.",
    "base_seq_projector.",
    "cross_attn.",
    "final_norm_base_seq.",
    "final_norm_ts.",
)

PRETRAIN_HEAD_PREFIXES = {
    "sublm_head": "sublm_head.",
    "masklm_head": "masklm_head.",
    "ema_signal_encoder": "ema_signal_encoder.",
    "latent_head": "latent_head.",
}


def _run_git(repository_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repository_path), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )


def build_repository_provenance(repository_path: Optional[str] = None) -> Dict[str, Any]:
    """Describe the Git checkout responsible for a saved model checkpoint.

    Git metadata is deliberately stored outside model state dictionaries so it
    cannot become an unexpected parameter during strict ``load_state_dict``.
    Failure to query Git is recorded in the metadata and never blocks saving.

    ``repository_path`` may name a file or directory inside the checkout; when
    omitted, discovery starts beside this module. The returned schema records
    commit, branch, repository root, and separate tracked/untracked dirty flags.
    """
    source_path = Path(repository_path).resolve() if repository_path else Path(__file__).resolve().parent
    provenance = {
        "schema_version": 1,
        "git_available": False,
        "commit_sha": None,
        "branch": None,
        "repository_root": None,
        "worktree_dirty": None,
        "tracked_files_dirty": None,
        "untracked_files_present": None,
    }

    try:
        root_result = _run_git(source_path, "rev-parse", "--show-toplevel")
        if root_result.returncode != 0:
            error = root_result.stderr.strip() or root_result.stdout.strip() or "not a Git checkout"
            provenance["error"] = error[:500]
            return provenance

        repository_root = Path(root_result.stdout.strip()).resolve()
        commit_result = _run_git(repository_root, "rev-parse", "HEAD")
        if commit_result.returncode != 0:
            error = commit_result.stderr.strip() or commit_result.stdout.strip() or "cannot resolve HEAD"
            provenance["error"] = error[:500]
            return provenance

        provenance["git_available"] = True
        provenance["repository_root"] = str(repository_root)
        provenance["commit_sha"] = commit_result.stdout.strip()

        branch_result = _run_git(repository_root, "symbolic-ref", "--quiet", "--short", "HEAD")
        if branch_result.returncode == 0:
            provenance["branch"] = branch_result.stdout.strip() or None

        status_result = _run_git(
            repository_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        )
        if status_result.returncode == 0:
            status_lines = [line for line in status_result.stdout.splitlines() if line]
            untracked_files_present = any(line.startswith("?? ") for line in status_lines)
            tracked_files_dirty = any(not line.startswith("?? ") for line in status_lines)
            provenance["untracked_files_present"] = untracked_files_present
            provenance["tracked_files_dirty"] = tracked_files_dirty
            provenance["worktree_dirty"] = tracked_files_dirty or untracked_files_present
        else:
            error = status_result.stderr.strip() or status_result.stdout.strip()
            if error:
                provenance["status_error"] = error[:500]
    except (OSError, subprocess.TimeoutExpired) as exc:
        provenance["error"] = f"{type(exc).__name__}: {exc}"[:500]

    return provenance


def attach_repository_provenance(
    checkpoint: Dict[str, Any],
    repository_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach repository metadata to a checkpoint payload in place.

    The same mutable mapping is returned for convenient payload construction.
    """
    checkpoint[REPOSITORY_PROVENANCE_KEY] = build_repository_provenance(repository_path)
    return checkpoint


def normalize_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip repeated DDP and ``torch.compile`` wrappers from state keys.

    Leading ``module.`` and ``_orig_mod.`` prefixes are removed in any order
    until neither remains. Tensor values are shared with the input mapping.
    """
    normalized = {}
    for key, value in state_dict.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod."):]
        normalized[key] = value
    return normalized


def config_to_dict(config: Any) -> Optional[Dict[str, Any]]:
    """Convert a dataclass or mapping-like configuration to a plain dictionary."""
    if config is None:
        return None
    if is_dataclass(config):
        return asdict(config)
    return dict(config)


def _strip_prefix(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _select_prefixes(state_dict: Dict[str, Any], prefixes: Tuple[str, ...]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in state_dict.items()
        if any(key.startswith(prefix) for prefix in prefixes)
    }


def split_pretraining_state_dict(state_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Partition a pretraining state dict into backbone and named head states.

    Backbone keys retain their model-level prefixes. Head keys have their head
    prefix removed and empty heads are omitted.
    """
    state_dict = normalize_state_dict_keys(state_dict)
    backbone_state_dict = _select_prefixes(state_dict, BACKBONE_PREFIXES)
    head_state_dicts = {}
    for head_name, prefix in PRETRAIN_HEAD_PREFIXES.items():
        head_state_dict = _strip_prefix(state_dict, prefix)
        if head_state_dict:
            head_state_dicts[head_name] = head_state_dict
    return backbone_state_dict, head_state_dicts


def split_classification_state_dict(state_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Partition a classifier state dict into backbone and all available heads.

    The leading ``pretrain.`` namespace is removed before splitting the
    backbone/pretraining heads; classifier-head keys are also stored without
    their ``classifier_head.`` prefix.
    """
    state_dict = normalize_state_dict_keys(state_dict)
    pretrain_state_dict = _strip_prefix(state_dict, "pretrain.")
    backbone_state_dict, pretraining_heads = split_pretraining_state_dict(pretrain_state_dict)

    head_state_dicts = {}
    classifier_head_state_dict = _strip_prefix(state_dict, "classifier_head.")
    if classifier_head_state_dict:
        head_state_dicts["classifier_head"] = classifier_head_state_dict
    head_state_dicts.update(pretraining_heads)

    return backbone_state_dict, head_state_dicts


def compose_pretraining_state_dict(
    backbone_state_dict: Dict[str, Any],
    head_state_dicts: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compose the legacy flat pretraining state dict from modular components.

    Unknown head names are ignored. Input mappings and tensor objects are not
    mutated or copied deeply.
    """
    state_dict = dict(backbone_state_dict or {})
    for head_name, head_state_dict in (head_state_dicts or {}).items():
        if head_name in PRETRAIN_HEAD_PREFIXES:
            prefix = PRETRAIN_HEAD_PREFIXES[head_name]
            state_dict.update({f"{prefix}{key}": value for key, value in head_state_dict.items()})
    return state_dict


def compose_classification_state_dict(
    backbone_state_dict: Dict[str, Any],
    head_state_dicts: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compose the legacy flat classifier state dict from modular components.

    Backbone and pretraining heads are placed under ``pretrain.``; the
    classifier head uses its top-level namespace. Unknown heads are ignored.
    """
    state_dict = {
        f"pretrain.{key}": value
        for key, value in (backbone_state_dict or {}).items()
    }
    for head_name, head_state_dict in (head_state_dicts or {}).items():
        if head_name == "classifier_head":
            prefix = "classifier_head."
        elif head_name in PRETRAIN_HEAD_PREFIXES:
            prefix = f"pretrain.{PRETRAIN_HEAD_PREFIXES[head_name]}"
        else:
            continue
        state_dict.update({f"{prefix}{key}": value for key, value in head_state_dict.items()})
    return state_dict


def get_pretraining_state_dict_from_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Return a loadable flat pretraining state from legacy or modular payloads.

    ``model_state_dict`` takes precedence when present. ``KeyError`` is raised
    when neither that key nor ``backbone_state_dict`` is available.
    """
    if checkpoint.get("model_state_dict") is not None:
        return checkpoint["model_state_dict"]
    if checkpoint.get("backbone_state_dict") is None:
        raise KeyError("Checkpoint has neither model_state_dict nor backbone_state_dict")
    return compose_pretraining_state_dict(
        checkpoint["backbone_state_dict"],
        checkpoint.get("head_state_dicts"),
    )


def get_classification_state_dict_from_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Return a loadable flat classifier state from legacy or modular payloads."""
    if checkpoint.get("model_state_dict") is not None:
        return checkpoint["model_state_dict"]
    if checkpoint.get("backbone_state_dict") is None:
        raise KeyError("Checkpoint has neither model_state_dict nor backbone_state_dict")
    return compose_classification_state_dict(
        checkpoint["backbone_state_dict"],
        checkpoint.get("head_state_dicts"),
    )


def build_backbone_manifest(backbone_config: Any) -> Dict[str, Any]:
    """Describe backbone configuration and outputs available at inference.

    ``contextual_ts`` is advertised only when signal-to-base cross-attention is
    enabled and has at least one layer. The manifest also records EMA support.
    """
    config = config_to_dict(backbone_config) or {}
    has_signal_cross_attn = bool(config.get("add_signal_cross_attn", True)) and int(config.get("num_layers_signal_cross", 0)) > 0
    supported_outputs = [
        "contextual_base_seq",
        "fixed_sa_base",
        "fixed_sa_ts",
        "prenorm_contextual_base_seq",
    ]
    if has_signal_cross_attn:
        supported_outputs.append("contextual_ts")

    return {
        "config": config,
        "has_signal_to_base_cross_attn": has_signal_cross_attn,
        "has_ema": bool(config.get("add_ema", False)),
        "supported_outputs": supported_outputs,
    }


def build_model_manifest(
    model_class: str,
    checkpoint_type: str,
    backbone_config: Any,
    head_state_dicts: Dict[str, Dict[str, Any]],
    classifier_config: Any = None,
    null_config: Any = None,
    scale_config: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the version-2 model manifest stored beside modular state dicts.

    Dataclass or mapping configurations are normalized to plain dictionaries.
    ``extra`` fields are merged at the manifest top level and may therefore
    extend or deliberately override default metadata.
    """
    manifest = {
        "format_version": 2,
        "model_class": model_class,
        "checkpoint_type": checkpoint_type,
        "backbone": build_backbone_manifest(backbone_config),
        "heads": sorted(head_state_dicts.keys()),
        "head_configs": {},
    }

    classifier_config = config_to_dict(classifier_config)
    null_config = config_to_dict(null_config)
    scale_config = config_to_dict(scale_config)
    if classifier_config is not None:
        manifest["head_configs"]["classifier_head"] = classifier_config
    if null_config is not None:
        manifest["head_configs"]["null_predictor"] = null_config
    if scale_config is not None:
        manifest["head_configs"]["scale_model"] = scale_config
    if extra:
        manifest.update(extra)

    return manifest


def _with_preprocessing(payload, model):
    """Carry non-parameter preprocessing metadata through modular exports."""
    profile = getattr(model, "preprocessing", None)
    if profile is None and hasattr(model, "pretrain"):
        profile = getattr(model.pretrain, "preprocessing", None)
    return attach_preprocessing(payload, profile) if profile is not None else payload


def modular_pretraining_payload(model: Any, checkpoint_type: str = "ont_pretraining_v2") -> Dict[str, Any]:
    """Create modular pretraining state, manifest, and repository provenance."""
    full_state_dict = model.state_dict()
    backbone_state_dict, head_state_dicts = split_pretraining_state_dict(full_state_dict)
    return _with_preprocessing(attach_repository_provenance({
        "backbone_state_dict": backbone_state_dict,
        "head_state_dicts": head_state_dicts,
        "model_manifest": build_model_manifest(
            model_class="ONTModelForPreTraining",
            checkpoint_type=checkpoint_type,
            backbone_config=model.config,
            head_state_dicts=head_state_dicts,
        ),
    }), model)


def modular_classification_payload(model: Any, checkpoint_type: str = "ont_classification_v2") -> Dict[str, Any]:
    """Create modular classifier state, manifest, and repository provenance."""
    full_state_dict = model.state_dict()
    backbone_state_dict, head_state_dicts = split_classification_state_dict(full_state_dict)
    return _with_preprocessing(attach_repository_provenance({
        "backbone_state_dict": backbone_state_dict,
        "head_state_dicts": head_state_dicts,
        "model_manifest": build_model_manifest(
            model_class="ONTModelForClassification",
            checkpoint_type=checkpoint_type,
            backbone_config=model.pretrain.config,
            classifier_config=model.classifier_config,
            head_state_dicts=head_state_dicts,
            extra={
                "classifier_signal_source": getattr(model.classifier_config, "signal_source", None),
                "resolved_signal_key": getattr(model, "signal_key", None),
                "effective_add_signal_cross_attn": bool(
                    model.pretrain.config.add_signal_cross_attn
                ),
                "classifier_execution_contract": model.execution_contract(),
            },
        ),
    }), model)


def modular_canonical_null_payload(
    backbone: Any,
    null_predictor: Any,
    scale_model: Any = None,
    checkpoint_type: str = "ont_canonical_null_v2",
) -> Dict[str, Any]:
    """Create a modular canonical-null checkpoint payload.

    The backbone, required null predictor, and optional scale model are stored
    as separate state dictionaries with their matching configurations in the
    manifest. Repository provenance is attached before returning.
    """
    head_state_dicts = {
        "null_predictor": null_predictor.state_dict(),
    }
    if scale_model is not None:
        head_state_dicts["scale_model"] = scale_model.state_dict()

    return _with_preprocessing(attach_repository_provenance({
        "backbone_state_dict": backbone.state_dict(),
        "head_state_dicts": head_state_dicts,
        "model_manifest": build_model_manifest(
            model_class="OntModelForCanonicalNull",
            checkpoint_type=checkpoint_type,
            backbone_config=backbone.config,
            null_config=null_predictor.config,
            scale_config=scale_model.config if scale_model is not None else None,
            head_state_dicts=head_state_dicts,
        ),
    }), backbone)
