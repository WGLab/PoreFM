"""JSON Schema validation and typed default normalization for PoreFM configs."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, best_match

from porefm.config.schemas import CONFIG_TYPES, SCHEMA_VERSION, get_schema


class ConfigValidationError(ValueError):
    """Report a configuration contract violation with its JSON location."""

    def __init__(self, source: str | Path, location: str, message: str):
        self.source = str(source)
        self.location = location
        self.message = message
        super().__init__(f"{self.source}: {location}: {message}")


class _DuplicateKeyError(ValueError):
    """Carry a duplicate JSON object key out of ``object_pairs_hook``."""

    def __init__(self, key: str):
        self.key = key
        super().__init__(key)


@dataclass(frozen=True)
class PretrainingSample:
    """One unlabelled signal-BAM input.

    Parameters
    ----------
    bam_path : str
        BAM path used as written by sequential pretraining readers.
    sampling_probability : float, default=1.0
        Independent probability in ``(0, 1]`` of retaining each eligible BAM
        record.
    """

    bam_path: str
    sampling_probability: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Return the normalized JSON representation."""
        return {
            "bam_path": self.bam_path,
            "sampling_probability": self.sampling_probability,
        }


@dataclass(frozen=True)
class PretrainingConfig:
    """Validated v1 pretraining/canonical-null BAM manifest.

    ``samples`` is nonempty. The schema version and type discriminator are
    normalized even when their optional JSON fields were omitted.
    """

    samples: tuple[PretrainingSample, ...]
    schema_version: int = SCHEMA_VERSION
    config_type: str = "pretraining"

    @property
    def bam_list(self) -> list[tuple[str, float]]:
        """Return the tuple contract consumed by ``BamIterableDataset``."""
        return [
            (sample.bam_path, sample.sampling_probability)
            for sample in self.samples
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return normalized JSON with every default made explicit."""
        return {
            "config_type": self.config_type,
            "schema_version": self.schema_version,
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True)
class ShardingInput:
    """One BAM input and its coordinate-sharding restriction.

    An empty ``contigs`` tuple means stream all records rather than making
    indexed per-contig fetches.
    """

    bam_path: str
    contigs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the normalized JSON representation."""
        return {"bam_path": self.bam_path, "contigs": list(self.contigs)}


@dataclass(frozen=True)
class ShardingConfig:
    """Validated v1 BAM sharding manifest containing at least one input."""

    inputs: tuple[ShardingInput, ...]
    schema_version: int = SCHEMA_VERSION
    config_type: str = "bam-sharding"

    def to_dict(self) -> dict[str, Any]:
        """Return normalized JSON with every default made explicit."""
        return {
            "config_type": self.config_type,
            "schema_version": self.schema_version,
            "inputs": [item.to_dict() for item in self.inputs],
        }


@dataclass(frozen=True)
class ValidatedConfig:
    """Generic validated config returned by the public validation command."""

    config_type: str
    schema_version: int
    data: Mapping[str, Any]


def _fail(source: str | Path, location: str, message: str):
    raise ConfigValidationError(source, location, message)


_SCHEMA_VALIDATORS = {
    config_type: Draft202012Validator(get_schema(config_type))
    for config_type in CONFIG_TYPES
}


def _json_location(path) -> str:
    """Render a jsonschema path deque as a readable JSON location."""
    location = "$"
    for part in path:
        if isinstance(part, int):
            location += f"[{part}]"
        elif str(part).isidentifier():
            location += f".{part}"
        else:
            location += f"[{json.dumps(str(part))}]"
    return location


def _schema_message(error: ValidationError, config_type: str) -> str:
    """Translate common validator failures into concise contract language."""
    if error.validator == "additionalProperties" and isinstance(error.instance, dict):
        allowed = set(error.schema.get("properties", {}))
        unknown = sorted(set(error.instance) - allowed)
        return f"contains unsupported fields: {unknown}"
    if error.validator == "maximum":
        return f"must be at most {error.validator_value}"
    if error.validator == "minimum":
        return f"must be at least {error.validator_value}"
    if error.validator == "exclusiveMinimum":
        return f"must be greater than {error.validator_value}"
    if error.validator == "exclusiveMaximum":
        return f"must be less than {error.validator_value}"
    return error.message


def _validate_schema(raw: Any, config_type: str, source: str | Path) -> None:
    """Validate structural fields and relationships with Draft 2020-12."""
    if isinstance(raw, dict):
        declared_type = raw.get("config_type", config_type)
        if declared_type != config_type:
            _fail(
                source,
                "$.config_type",
                f"must be {config_type!r}, got {declared_type!r}",
            )
        version = raw.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            _fail(
                source,
                "$.schema_version",
                f"unsupported {config_type!r} schema version {version}; supported version is {SCHEMA_VERSION}",
            )
        if config_type == "supervised":
            has_labelled = "labelled_samples" in raw
            has_tagged = "tagged_samples" in raw
            if has_labelled == has_tagged:
                _fail(
                    source,
                    "$",
                    "must define exactly one of labelled_samples or tagged_samples",
                )

    error = best_match(_SCHEMA_VALIDATORS[config_type].iter_errors(raw))
    if error is not None:
        _fail(
            source,
            _json_location(error.absolute_path),
            _schema_message(error, config_type),
        )


def _text(value: Any, source: str | Path, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(source, location, "must be a nonempty string")
    return value.strip()


def _optional_text(value: Any, source: str | Path, location: str) -> str | None:
    if value is None:
        return None
    return value.strip() or None


def _string_list(value: Any, source: str | Path, location: str) -> list[str]:
    normalized = [
        _text(item, source, f"{location}[{idx}]")
        for idx, item in enumerate(value)
    ]
    if len(normalized) != len(set(normalized)):
        _fail(source, location, "must not contain duplicate values")
    return normalized


def _read_json(path: str | Path) -> Any:
    """Decode strict JSON, rejecting duplicate keys and non-standard constants."""

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateKeyError(key)
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value!r}")

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_constant,
            )
    except _DuplicateKeyError as exc:
        raise ConfigValidationError(
            source,
            "$",
            f"contains duplicate object field {exc.key!r}",
        ) from exc
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(
            source,
            f"line {exc.lineno}, column {exc.colno}",
            exc.msg,
        ) from exc
    except ValueError as exc:
        raise ConfigValidationError(source, "$", str(exc)) from exc
    except OSError as exc:
        raise ConfigValidationError(source, "$", str(exc)) from exc


def _validate_pretraining_data(raw: Any, source: str | Path) -> PretrainingConfig:
    _validate_schema(raw, "pretraining", source)
    version = raw.get("schema_version", SCHEMA_VERSION)
    samples = []
    for idx, sample in enumerate(raw["samples"]):
        location = f"$.samples[{idx}]"
        samples.append(
            PretrainingSample(
                bam_path=_text(sample["bam_path"], source, f"{location}.bam_path"),
                sampling_probability=float(sample.get("sampling_probability", 1.0)),
            )
        )
    return PretrainingConfig(samples=tuple(samples), schema_version=version)


def _validate_sharding_data(raw: Any, source: str | Path) -> ShardingConfig:
    _validate_schema(raw, "bam-sharding", source)
    version = raw.get("schema_version", SCHEMA_VERSION)
    inputs = []
    for idx, item in enumerate(raw["inputs"]):
        location = f"$.inputs[{idx}]"
        inputs.append(
            ShardingInput(
                bam_path=_text(item["bam_path"], source, f"{location}.bam_path"),
                contigs=tuple(
                    _string_list(item.get("contigs", []), source, f"{location}.contigs")
                ),
            )
        )
    return ShardingConfig(inputs=tuple(inputs), schema_version=version)


def _validate_polishing(
    sample: Mapping[str, Any],
    source: str | Path,
    location: str,
) -> tuple[bool, int, str | None]:
    enabled = sample.get("polish", False)
    indel_limit = sample.get("polish_indel_limit", 5)
    vcf_path = _optional_text(
        sample.get("polish_vcf_path"),
        source,
        f"{location}.polish_vcf_path",
    )
    if vcf_path and not enabled:
        _fail(source, f"{location}.polish_vcf_path", "requires polish=true")
    return enabled, indel_limit, vcf_path


def _motif_labels(value: Any, source: str | Path, location: str) -> list[int]:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        _fail(source, location, "must be an integer or semicolon-delimited integer string")
    try:
        labels = [int(part.strip()) for part in str(value).split(";")]
    except ValueError as exc:
        raise ConfigValidationError(
            source,
            location,
            "must contain only semicolon-delimited integer class indices",
        ) from exc
    if not labels or any(label < 0 for label in labels):
        _fail(source, location, "must contain nonnegative class indices")
    return labels


def _normalize_labelled_sample(
    raw: Any,
    source: str | Path,
    location: str,
    class_labels: Sequence[str] | None,
) -> dict[str, Any]:
    sample = raw

    bam_path = _text(sample["bam_path"], source, f"{location}.bam_path")
    if "bam_index" in sample and "pod5" not in sample:
        _fail(source, f"{location}.bam_index", "requires a pod5 signal input")
    contigs = _string_list(sample.get("contigs", []), source, f"{location}.contigs")
    read_based = sample.get("read_based", False)
    ref = _optional_text(sample.get("ref"), source, f"{location}.ref")
    pos_file = _optional_text(sample.get("pos_file"), source, f"{location}.pos_file")
    motif = _optional_text(sample.get("motif"), source, f"{location}.motif")

    if read_based:
        if pos_file:
            _fail(source, f"{location}.pos_file", "is incompatible with read_based=true")
        if not motif:
            _fail(source, f"{location}.motif", "is required when read_based=true")
    else:
        if not ref:
            _fail(source, f"{location}.ref", "is required for reference-based samples")
        if bool(pos_file) == bool(motif):
            _fail(source, location, "must define exactly one of motif or pos_file")

    motif_label = sample.get("motif_label", 0)
    labels = _motif_labels(motif_label, source, f"{location}.motif_label")
    if class_labels is not None and any(label >= len(class_labels) for label in labels):
        _fail(
            source,
            f"{location}.motif_label",
            f"indices must be in [0, {len(class_labels)})",
        )

    polish, polish_indel_limit, polish_vcf_path = _validate_polishing(
        sample, source, location
    )
    return {
        "bam_path": bam_path,
        **({"pod5": _text(sample["pod5"], source, f"{location}.pod5")}
           if "pod5" in sample else {}),
        **({"bam_index": _text(sample["bam_index"], source, f"{location}.bam_index")}
           if "bam_index" in sample else {}),
        "contigs": contigs,
        "ref": ref,
        "pos_file": pos_file,
        "motif": motif,
        "motif_label": motif_label,
        **_normalize_label_smoothing(sample, source, location),
        "flag": sample.get("flag", 0),
        "sample_labels": float(sample.get("sample_labels", 1.0)),
        "polish": polish,
        "read_based": read_based,
        "polish_indel_limit": polish_indel_limit,
        "polish_vcf_path": polish_vcf_path,
    }


def _normalize_label_smoothing(
    raw: Mapping[str, Any], source: str | Path, location: str,
) -> dict[str, float]:
    """Preserve epsilon omission and reject legacy target transformations.

    Numeric ``soft_tags: 0`` is accepted only as a deprecated no-op. It does
    not imply ``label_smoothing: 0`` and therefore does not override a config
    or CLI default. Nonzero legacy values require an explicit migration.
    """
    if "soft_tags" in raw:
        if raw["soft_tags"] != 0:
            _fail(
                source, f"{location}.soft_tags",
                "nonzero soft_tags is retired; migrate to label_smoothing. "
                "For one legacy transformation, epsilon = soft_tags * C / (C - 1) "
                "matches its formula only when epsilon < 1. Reference-position and "
                "motif targets were not effectively smoothed; scalar tagged dictionary "
                "targets could be transformed twice. Review any CLI smoothing too; "
                "the new values override rather than compound it. See docs/configuration.md.",
            )
        warnings.warn(
            f"{source}: {location}.soft_tags: soft_tags=0 is deprecated and ignored; "
            "remove it. Use label_smoothing=0 to override config/CLI smoothing.",
            FutureWarning,
            stacklevel=3,
        )
    if "label_smoothing" not in raw:
        return {}
    value = raw["label_smoothing"]
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value < 1):
        _fail(source, f"{location}.label_smoothing", "must be a finite number in [0, 1)")
    return {"label_smoothing": float(value)}


def _normalize_tagged_sample(
    raw: Any,
    source: str | Path,
    location: str,
    class_labels: Sequence[str] | None,
) -> dict[str, Any]:
    sample = raw

    ml_tag = sample.get("ML_tag", False)
    label_dict = _optional_text(
        sample.get("read_label_dict_path"),
        source,
        f"{location}.read_label_dict_path",
    )
    if ml_tag and label_dict:
        _fail(source, location, "ML_tag=true cannot be combined with read_label_dict_path")
    mm_code = sample.get("MM_code")
    if mm_code is not None and not ml_tag:
        _fail(source, f"{location}.MM_code", "requires ML_tag=true")
    if not ml_tag and not label_dict:
        _fail(
            source,
            f"{location}.read_label_dict_path",
            "is required when ML_tag=false",
        )

    label = sample.get("label", 1)
    if class_labels is not None and label >= len(class_labels):
        _fail(source, f"{location}.label", f"must be in [0, {len(class_labels)})")
    negative_label = sample.get("negative_label")
    if negative_label is not None:
        if not ml_tag:
            _fail(source, f"{location}.negative_label", "requires ML_tag=true")
        if class_labels is not None and negative_label >= len(class_labels):
            _fail(source, f"{location}.negative_label", f"must be in [0, {len(class_labels)})")
        if negative_label == label:
            _fail(source, f"{location}.negative_label", "must differ from label")
    polish, polish_indel_limit, polish_vcf_path = _validate_polishing(
        sample, source, location
    )
    return {
        "bam_path": _text(sample["bam_path"], source, f"{location}.bam_path"),
        "contigs": _string_list(sample.get("contigs", []), source, f"{location}.contigs"),
        "ref": _optional_text(sample.get("ref"), source, f"{location}.ref"),
        "label_coordinates": sample.get("label_coordinates"),
        "read_label_dict_path": label_dict,
        **_normalize_label_smoothing(sample, source, location),
        "label": label,
        "negative_label": negative_label,
        "ML_tag": ml_tag,
        "MM_code": mm_code,
        "ML_threshold": float(sample.get("ML_threshold", 0.5)),
        "min_read_frac": float(sample.get("min_read_frac", 0.0)),
        "flag": sample.get("flag", 0),
        "polish": polish,
        "polish_indel_limit": polish_indel_limit,
        "polish_vcf_path": polish_vcf_path,
    }


def _normalize_supervision(
    raw: Any,
    source: str | Path,
) -> dict[str, Any]:
    location = "$.supervision"
    supervision = raw
    supervision_type = supervision.get("type", "classification")
    if supervision_type == "classification":
        return {"type": "classification"}
    normalized: dict[str, Any] = {
        "type": "exclusion",
        "excluded_class": _text(
            supervision["excluded_class"], source, f"{location}.excluded_class"
        ),
    }

    expected = supervision.get("expected_allowed_proportions")
    tolerance = supervision.get("proportion_tolerance")
    if expected is None:
        return normalized

    expected_mapping = expected
    if not expected_mapping:
        _fail(
            source,
            f"{location}.expected_allowed_proportions",
            "must not be empty",
        )
    normalized_expected = {
        _text(name, source, f"{location}.expected_allowed_proportions key"): float(value)
        for name, value in expected_mapping.items()
    }
    if not math.isclose(sum(normalized_expected.values()), 1.0, rel_tol=0, abs_tol=1e-6):
        _fail(
            source,
            f"{location}.expected_allowed_proportions",
            f"values must sum to 1; got {sum(normalized_expected.values()):.8g}",
        )
    normalized["expected_allowed_proportions"] = normalized_expected
    normalized["proportion_tolerance"] = float(tolerance)
    return normalized


def _validate_supervised_data(
    raw: Any,
    source: str | Path,
    *,
    class_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    _validate_schema(raw, "supervised", source)
    root = raw
    version = root.get("schema_version", SCHEMA_VERSION)
    has_labelled = "labelled_samples" in root
    has_tagged = "tagged_samples" in root
    if class_labels is not None:
        if not class_labels or len(class_labels) != len(set(class_labels)):
            _fail(source, "$", "class_labels must contain unique class names")

    normalized: dict[str, Any] = {
        "config_type": "supervised",
        "schema_version": version,
        "supervision": _normalize_supervision(
            root.get("supervision", {"type": "classification"}), source
        ),
        **_normalize_label_smoothing(root, source, "$"),
    }
    if has_labelled:
        normalized["labelled_samples"] = [
            _normalize_labelled_sample(
                sample,
                source,
                f"$.labelled_samples[{idx}]",
                class_labels,
            )
            for idx, sample in enumerate(root["labelled_samples"])
        ]
    else:
        normalized["tagged_samples"] = [
            _normalize_tagged_sample(
                sample,
                source,
                f"$.tagged_samples[{idx}]",
                class_labels,
            )
            for idx, sample in enumerate(root["tagged_samples"])
        ]
    if normalized["supervision"]["type"] == "exclusion":
        if normalized.get("label_smoothing", 0) != 0:
            _fail(source, "$.label_smoothing", "must be zero or omitted for exclusion supervision")
        for idx, sample in enumerate(normalized["labelled_samples"]):
            if sample.get("label_smoothing", 0) != 0:
                _fail(
                    source,
                    f"$.labelled_samples[{idx}].label_smoothing",
                    "must be zero or omitted for exclusion supervision",
                )

        if class_labels is not None:
            excluded_class = normalized["supervision"]["excluded_class"]
            if excluded_class not in class_labels:
                _fail(
                    source,
                    "$.supervision.excluded_class",
                    f"must be one of {list(class_labels)}",
                )
            excluded_index = class_labels.index(excluded_class)
            for idx, sample in enumerate(normalized["labelled_samples"]):
                if sample["motif"] is None:
                    continue
                labels = _motif_labels(
                    sample["motif_label"],
                    source,
                    f"$.labelled_samples[{idx}].motif_label",
                )
                if any(label != excluded_index for label in labels):
                    _fail(
                        source,
                        f"$.labelled_samples[{idx}].motif_label",
                        f"motif labels must all match excluded class index {excluded_index}",
                    )

            expected = normalized["supervision"].get(
                "expected_allowed_proportions"
            )
            if expected is not None:
                allowed_classes = {
                    label for label in class_labels if label != excluded_class
                }
                if set(expected) != allowed_classes:
                    _fail(
                        source,
                        "$.supervision.expected_allowed_proportions",
                        f"keys must exactly match allowed classes {sorted(allowed_classes)}",
                    )
    return normalized


def infer_config_type(raw: Any, source: str | Path = "<config>") -> str:
    """Infer a config type from its discriminator or unique top-level field.

    Raises :class:`ConfigValidationError` when the discriminator is unknown or
    the structure is absent or ambiguous.
    """
    if not isinstance(raw, dict):
        _fail(source, "$", "cannot infer config type from a non-object JSON value")
    root = raw
    declared = root.get("config_type")
    if declared is not None:
        if declared not in CONFIG_TYPES:
            _fail(source, "$.config_type", f"must be one of {list(CONFIG_TYPES)}")
        return declared
    matches = []
    if "samples" in root:
        matches.append("pretraining")
    if "inputs" in root:
        matches.append("bam-sharding")
    if "labelled_samples" in root or "tagged_samples" in root:
        matches.append("supervised")
    if len(matches) != 1:
        _fail(
            source,
            "$",
            "cannot infer config type; add config_type or pass --type",
        )
    return matches[0]


def validate_config_data(
    raw: Any,
    *,
    expected_type: str | None = None,
    source: str | Path = "<config>",
    class_labels: Sequence[str] | None = None,
) -> ValidatedConfig:
    """Validate already-decoded JSON and apply all schema defaults.

    Parameters
    ----------
    raw : Any
        Value returned by ``json.load``.
    expected_type : str or None
        Required config type. When omitted, the discriminator or top-level
        structure is used to infer the type.
    source : str or pathlib.Path
        Label included in validation errors.
    class_labels : sequence of str or None
        Optional classifier-index order for validating supervised class IDs.

    Returns
    -------
    ValidatedConfig
        Normalized data containing explicit schema headers and defaults.
    """
    config_type = expected_type or infer_config_type(raw, source)
    if config_type not in CONFIG_TYPES:
        _fail(source, "$", f"unknown config type {config_type!r}")
    if config_type == "pretraining":
        config = _validate_pretraining_data(raw, source)
        data = config.to_dict()
    elif config_type == "bam-sharding":
        config = _validate_sharding_data(raw, source)
        data = config.to_dict()
    else:
        data = _validate_supervised_data(raw, source, class_labels=class_labels)
    return ValidatedConfig(config_type, SCHEMA_VERSION, data)


def validate_config_file(
    path: str | Path,
    *,
    expected_type: str | None = None,
    class_labels: Sequence[str] | None = None,
) -> ValidatedConfig:
    """Load, validate, and normalize one JSON configuration file.

    Parameters
    ----------
    path : str or pathlib.Path
        JSON file to decode with duplicate-key detection.
    expected_type : str or None
        Required config type, or ``None`` to infer it from the JSON header or
        unique top-level collection.
    class_labels : sequence of str or None
        Optional classifier order used for supervised class-name and index
        checks.

    Returns
    -------
    ValidatedConfig
        Detected type, supported version, and normalized data with defaults.

    Raises
    ------
    ConfigValidationError
        If JSON decoding or any type-specific contract check fails.
    """
    return validate_config_data(
        _read_json(path),
        expected_type=expected_type,
        source=path,
        class_labels=class_labels,
    )


def load_pretraining_config(path: str | Path) -> PretrainingConfig:
    """Load a strict v1 pretraining BAM manifest.

    Parameters
    ----------
    path : str or pathlib.Path
        JSON object containing a nonempty ``samples`` array.

    Returns
    -------
    PretrainingConfig
        Immutable samples with explicit sampling-probability defaults.
    """
    raw = _read_json(path)
    return _validate_pretraining_data(raw, path)


def load_sharding_config(path: str | Path) -> ShardingConfig:
    """Load a strict v1 BAM-sharding manifest.

    Returns an immutable configuration whose omitted per-input ``contigs``
    values have been normalized to empty tuples.
    """
    raw = _read_json(path)
    return _validate_sharding_data(raw, path)


def load_supervised_config(
    path: str | Path,
    *,
    class_labels: Sequence[str] | None = None,
    allow_pod5: bool = False,
) -> dict[str, Any]:
    """Load and normalize a labelled/tagged supervised dataset config.

    Parameters
    ----------
    path : str or pathlib.Path
        Version 1 supervised JSON file.
    class_labels : sequence of str or None
        Classifier labels in index order. When provided, motif/tag labels and
        exclusion-supervision class names are checked against this order.
    allow_pod5 : bool, default=False
        Enable inference-only labelled-sample ``pod5`` inputs. Only the
        ``porefm detect-modification`` entry point (alias ``porefm infer``) opts in;
        training and other consumers reject this field rather than silently
        expecting SG tags.

    Returns
    -------
    dict
        Strict normalized mapping with exactly one nonempty sample section.
    """
    raw = _read_json(path)
    config = _validate_supervised_data(raw, path, class_labels=class_labels)
    if not allow_pod5:
        for idx, sample in enumerate(config.get("labelled_samples", [])):
            if "pod5" in sample:
                _fail(path, f"$.labelled_samples[{idx}].pod5",
                      "is supported only by porefm detect-modification (alias: porefm infer); other modes require SG signal BAMs")
    return config
