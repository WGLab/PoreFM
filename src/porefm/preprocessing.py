"""Resolve portable signal preprocessing independently of model execution.

Built-in chemistry profiles use checksum-verified package resources. Custom
profiles carry their exact float32 base-4 lookup table. No network access or
model construction occurs here; only ``load_preprocessing`` imports torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
import hashlib
import json
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class MissingPreprocessingError(ValueError):
    """A checkpoint lacks enough information to identify its preprocessing."""


def _word(value):
    return str(value).strip().lower() if value not in (None, "") else None


def get_kmer_size_from_lookup_table(lookup_table):
    """Return odd k for a one-dimensional dense table of exactly ``4**k`` values."""
    import numpy as np
    shape = np.shape(lookup_table)
    if len(shape) != 1:
        raise ValueError(f"Expected a one-dimensional k-mer lookup table, got shape {shape}.")
    size, k, expected = shape[0], 0, 1
    while expected < size:
        expected *= 4
        k += 1
    if expected != size or k < 1:
        raise ValueError(f"K-mer lookup table length must equal 4**k; got {size}.")
    if k % 2 == 0:
        raise ValueError(f"Expected an odd k-mer size for centered signal extraction; got k={k}.")
    return k


def _parse_table(text, source):
    """Parse the established headerless KMER/LEVEL format, retaining sparse zeros."""
    import numpy as np
    lookup, k = None, None
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Expected two columns in {source} at line {number}; got {len(fields)}.")
        kmer, value = fields
        kmer = kmer.upper()
        if k is None:
            k = len(kmer)
            if k < 1 or k % 2 == 0:
                raise ValueError(f"Expected an odd k-mer size in {source}; got k={k}.")
            lookup = np.zeros(4 ** k, dtype=np.float32)
        if len(kmer) != k:
            raise ValueError(f"Mixed k-mer lengths in {source} at line {number}.")
        idx = 0
        for base in kmer:
            if base not in "ACGT":
                raise ValueError(f"Invalid base {base!r} in {source} at line {number}.")
            idx = (idx << 2) | "ACGT".index(base)
        lookup[idx] = float(value)
    if lookup is None:
        raise ValueError(f"K-mer signal table is empty: {source}")
    return lookup


def _array(table):
    import numpy as np
    if hasattr(table, "detach"):
        table = table.detach().cpu().numpy()
    array = np.asarray(table, dtype=np.float32)
    get_kmer_size_from_lookup_table(array)
    if not np.isfinite(array).all():
        raise ValueError("K-mer table contains non-finite expected levels.")
    return array


def table_digest(table):
    """Hash dense A/C/G/T base-4 levels as canonical little-endian float32 bytes."""
    return hashlib.sha256(_array(table).astype("<f4", copy=False).tobytes()).hexdigest()


@lru_cache(maxsize=1)
def _registry():
    root = files("porefm").joinpath("resources/kmer_models")
    return json.loads(root.joinpath("manifest.json").read_text())


def _chemistry(chemistry=None, seq_type=None, pore_type=None):
    """Resolve explicit chemistry or unambiguous historical pore/modality pairs."""
    chemistry, seq, pore = _word(chemistry), _word(seq_type), _word(pore_type)
    if chemistry:
        aliases = {"r10": "dna_r10.4.1_e8.2_400bps", "rna_r9.4_180mv_70bps": "rna002"}
        chemistry = aliases.get(chemistry, chemistry)
        if chemistry != "custom" and chemistry not in _registry()["tables"]:
            raise ValueError(f"Unknown chemistry {chemistry!r}; choose {list(_registry()['tables'])} or custom.")
        return chemistry
    if pore in {"rna002", "rna004"}:
        return pore
    if pore in {"r9", "r9.4", "r9.4.1"} and seq == "rna":
        return "rna002"
    if pore in {"r10", "r10.4.1", "dna_r10.4.1_e8.2_400bps"} and seq == "dna":
        return "dna_r10.4.1_e8.2_400bps"
    if pore == "r10/rna004" and seq in {"dna", "rna"}:
        return "rna004" if seq == "rna" else "dna_r10.4.1_e8.2_400bps"
    return None


@dataclass(frozen=True, eq=False)
class PreprocessingSpec:
    """Resolved chemistry and immutable expected-current lookup.

    ``kmer_table`` is a read-only float32 vector of length ``4**k`` indexed in
    A/C/G/T base-4 order. RNA tables retain upstream 5'-to-3' sequence order;
    existing BAM decoders handle RNA signal orientation. Use the resolver to
    construct profiles and ``to_dict`` for portable checkpoint metadata.
    """

    chemistry: str
    seq_type: str
    pore_type: str | None
    kmer_table: np.ndarray = field(repr=False)
    table_sha256: str = field(init=False)
    kmer_size: int = field(init=False)

    def __post_init__(self):
        import numpy as np
        table = _array(self.kmer_table)
        # Immutable bytes also prevent consumers re-enabling numpy writeability.
        table = np.frombuffer(table.astype("<f4", copy=False).tobytes(), dtype="<f4")
        object.__setattr__(self, "kmer_table", table)
        object.__setattr__(self, "kmer_size", get_kmer_size_from_lookup_table(table))
        object.__setattr__(self, "table_sha256", table_digest(table))
        if self.seq_type not in {"dna", "rna"}:
            raise ValueError("seq_type must be dna or rna")
        if self.pore_type in {"rna002", "rna004"} and self.seq_type != "rna":
            raise ValueError(f"pore_type={self.pore_type!r} requires seq_type='rna'.")
        if self.chemistry != "custom":
            row = _registry()["tables"].get(self.chemistry)
            if row is None or (self.seq_type, self.pore_type, self.table_sha256) != (
                row["seq_type"], row["pore_type"], row["table_sha256"]
            ):
                raise ValueError("Preprocessing profile conflicts with its registered chemistry/table.")

    def __reduce__(self):
        # DataLoader spawn must restore read-only tables too.
        return (preprocessing_from_dict, (self.to_dict(),))

    def to_dict(self, *, include_table=True):
        """Return JSON-safe metadata; custom checkpoints must include the table."""
        result = dict(schema_version=1, chemistry=self.chemistry, seq_type=self.seq_type,
                      pore_type=self.pore_type, kmer_size=self.kmer_size,
                      table_sha256=self.table_sha256)
        if self.chemistry == "custom":
            if include_table:
                result["kmer_table"] = self.kmer_table.tolist()
        else:
            registry = _registry()
            row = registry["tables"][self.chemistry]
            result.update(repository=registry["repository"], upstream_commit=registry["commit"],
                          upstream_path=row["path"], file_sha256=row["file_sha256"])
        return result


@lru_cache(maxsize=3)
def _builtin(chemistry):
    row = _registry()["tables"][chemistry]
    raw = files("porefm").joinpath("resources/kmer_models", row["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["file_sha256"]:
        raise ValueError(f"Packaged k-mer table checksum mismatch: {row['path']}")
    return PreprocessingSpec(chemistry, row["seq_type"], row["pore_type"],
                             _parse_table(raw.decode(), row["path"]))


def _check(spec, *, chemistry=None, seq_type=None, pore_type=None, table=None):
    """Treat explicitly supplied settings as assertions, never overrides."""
    seq, pore = _word(seq_type), _word(pore_type)
    if seq and seq != spec.seq_type:
        raise ValueError(f"seq_type={seq!r} conflicts with preprocessing seq_type={spec.seq_type!r}.")
    if chemistry and _chemistry(chemistry) != spec.chemistry:
        raise ValueError(f"chemistry={chemistry!r} conflicts with preprocessing chemistry={spec.chemistry!r}.")
    if pore and pore != spec.pore_type:
        resolved = _chemistry(seq_type=spec.seq_type, pore_type=pore)
        if resolved != spec.chemistry:
            raise ValueError(f"pore_type={pore!r} conflicts with preprocessing chemistry={spec.chemistry!r}.")
    if table is not None and table_digest(table) != spec.table_sha256:
        raise ValueError("Explicit k-mer table conflicts with checkpoint/chemistry preprocessing.")
    return spec


def preprocessing_from_dict(record):
    """Validate a version-1 preprocessing record, including table/provenance hashes."""
    if not isinstance(record, Mapping) or record.get("schema_version") != 1:
        raise ValueError("Unsupported preprocessing metadata; expected schema_version=1.")
    chemistry = _chemistry(record.get("chemistry"))
    if not chemistry:
        raise ValueError("Preprocessing record is missing chemistry.")
    if chemistry == "custom":
        if record.get("kmer_table") is None:
            raise MissingPreprocessingError("Custom preprocessing metadata is missing its embedded kmer_table.")
        spec = PreprocessingSpec("custom", _word(record.get("seq_type")), _word(record.get("pore_type")),
                                 record["kmer_table"])
    else:
        spec = _builtin(chemistry)
    expected = spec.to_dict(include_table=False)
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Preprocessing metadata {key}={record.get(key)!r} conflicts with expected {value!r}.")
    if "kmer_table" in record:
        _check(spec, table=record["kmer_table"])
    return spec


def _legacy_profile(metadata):
    table = metadata.get("kmer_table")
    seq, pore = _word(metadata.get("seq_type")), _word(metadata.get("pore_type"))
    if table is not None:
        digest = table_digest(table)
        match = next((name for name, row in _registry()["tables"].items()
                      if digest == row["table_sha256"]), None)
        if match:
            return _check(_builtin(match), seq_type=seq, pore_type=pore)
        if seq:
            return PreprocessingSpec("custom", seq, pore, table)
    chemistry = _chemistry(metadata.get("chemistry"), seq_type=seq, pore_type=pore)
    if chemistry and chemistry != "custom":
        return _check(_builtin(chemistry), seq_type=seq, pore_type=pore)
    return None


def preprocessing_from_checkpoint(checkpoint, *, required=True):
    """Resolve legacy/new checkpoint metadata without inspecting tensor architecture.

    All usable metadata copies must agree. Historical file paths are never
    opened or used to infer table identity. Missing metadata may return None
    for tensor-only loading; data workflows leave ``required=True``.
    """
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping.")
    profiles, metadata = [], []
    if checkpoint.get("preprocessing") is not None:
        profiles.append(preprocessing_from_dict(checkpoint["preprocessing"]))
    for key in ("pretrain_meta", "meta"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            metadata.append(value)
            if isinstance(value.get("args"), Mapping):
                metadata.append(value["args"])
    for value in metadata:
        if value.get("preprocessing") is not None:
            profiles.append(preprocessing_from_dict(value["preprocessing"]))
        # Generic legacy labels are weaker evidence than an explicit contract
        # or an embedded table; validate those against the resolved profile below.
        if value.get("kmer_table") is not None:
            profile = _legacy_profile(value)
            if profile is not None:
                profiles.append(profile)
    if not profiles:
        profiles = [profile for value in metadata if (profile := _legacy_profile(value)) is not None]
    if not profiles:
        if required:
            raise MissingPreprocessingError(
                "Checkpoint has no unambiguous preprocessing metadata. Use "
                "'porefm utils migrate-preprocessing INPUT OUTPUT --chemistry ...'."
            )
        return None
    spec = profiles[0]
    for other in profiles[1:]:
        assert_compatible_preprocessing(spec, other)
    for value in metadata:
        _check(spec, chemistry=value.get("chemistry"),
               seq_type=value.get("seq_type"), pore_type=value.get("pore_type"),
               table=value.get("kmer_table"))
    return spec


def assert_compatible_preprocessing(first, second):
    """Reject mixing models/artifacts whose resolved preprocessing differs."""
    if (first.chemistry, first.seq_type, first.pore_type, first.table_sha256) != (
        second.chemistry, second.seq_type, second.pore_type, second.table_sha256
    ):
        raise ValueError("Conflicting preprocessing contracts: chemistry or k-mer table differs.")


def validate_artifact_preprocessing(artifact, preprocessing=None):
    """Check recorded contracts in derived artifacts and nested source metadata.

    Legacy artifacts without contracts remain readable. Tensor/state payloads
    and embedded level lists are not traversed. Returns the shared profile, or
    None when neither the caller nor the artifact supplies one.
    """
    spec = resolve_preprocessing(preprocessing=preprocessing) if preprocessing is not None else None

    def visit(value):
        nonlocal spec
        if isinstance(value, Mapping):
            if value.get("preprocessing") is not None:
                candidate = resolve_preprocessing(preprocessing=value["preprocessing"])
                if spec is not None:
                    assert_compatible_preprocessing(spec, candidate)
                spec = candidate
            for key, child in value.items():
                if key not in {"preprocessing", "kmer_table", "model_state_dict", "backbone_state_dict",
                               "head_state_dicts", "optimizer_state_dict"}:
                    if isinstance(child, (Mapping, list, tuple)):
                        visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                if isinstance(child, (Mapping, list, tuple)):
                    visit(child)

    visit(artifact)
    return spec


def load_preprocessing(checkpoint_path, *, required=True):
    """Read metadata from a trusted checkpoint on CPU, without constructing a model."""
    import torch
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as exc:
        if "mmap" not in str(exc):
            raise
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return preprocessing_from_checkpoint(checkpoint, required=required)


def resolve_preprocessing(*, preprocessing=None, checkpoint=None, chemistry=None,
                          seq_type=None, pore_type=None, kmer_table=None, kmer_dict_path=None):
    """Resolve one profile from a checkpoint, chemistry, or explicit custom table.

    ``checkpoint`` accepts a trusted path or loaded mapping. ``preprocessing``
    accepts a resolved object or versioned record. Additional values are
    assertions. Without a checkpoint/profile, an explicit table plus sequence
    type defines a custom profile unless its values identify a built-in table.
    """
    if kmer_dict_path:
        supplied = _parse_table(Path(kmer_dict_path).read_text(), kmer_dict_path)
        if kmer_table is not None and table_digest(supplied) != table_digest(kmer_table):
            raise ValueError("kmer_table and kmer_dict_path disagree.")
        kmer_table = supplied
    spec = None
    if preprocessing is not None:
        spec = preprocessing if isinstance(preprocessing, PreprocessingSpec) else preprocessing_from_dict(preprocessing)
    if checkpoint is not None:
        other = (preprocessing_from_checkpoint(checkpoint) if isinstance(checkpoint, Mapping)
                 else load_preprocessing(checkpoint))
        if spec is not None:
            assert_compatible_preprocessing(spec, other)
        spec = other
    selected = _chemistry(chemistry, seq_type, pore_type)
    if spec is None:
        if chemistry and selected != "custom":
            spec = _builtin(selected)
        elif kmer_table is not None:
            if selected == "custom":
                spec = PreprocessingSpec("custom", _word(seq_type), _word(pore_type), kmer_table)
            else:
                spec = _legacy_profile(dict(seq_type=seq_type, pore_type=pore_type, kmer_table=kmer_table))
        elif selected and selected != "custom":
            spec = _builtin(selected)
    if spec is None:
        raise MissingPreprocessingError(
            "Specify a checkpoint/preprocessing profile, an unambiguous chemistry "
            "(dna_r10.4.1_e8.2_400bps, rna004, rna002), or seq_type plus a custom table."
        )
    return _check(spec, chemistry=chemistry, seq_type=seq_type, pore_type=pore_type, table=kmer_table)


def build_lookup_table(kmer_dict_path=None, *, chemistry=None, preprocessing=None):
    """Load explicit KMER/LEVEL text or return a cached built-in/profile table.

    Explicit files retain the historical sparse-zero format and writable
    float32 result. Resolved profile tables are read-only. No table download
    occurs; supported chemistry resources ship with the installed package.
    """
    if kmer_dict_path is not None and chemistry is None and preprocessing is None:
        return _parse_table(Path(kmer_dict_path).read_text(), kmer_dict_path)
    return resolve_preprocessing(chemistry=chemistry, preprocessing=preprocessing,
                                 kmer_dict_path=kmer_dict_path).kmer_table


def attach_preprocessing(checkpoint, preprocessing):
    """Write one canonical record and remove redundant legacy signal settings.

    Other training metadata, parameters, and optimizer state are preserved.
    Existing metadata mappings are copied before removing chemistry fields.
    """
    spec = resolve_preprocessing(preprocessing=preprocessing)
    checkpoint["preprocessing"] = spec.to_dict()
    for key in ("meta", "pretrain_meta"):
        if isinstance(checkpoint.get(key), Mapping):
            value = dict(checkpoint[key])
            for field_name in ("chemistry", "seq_type", "pore_type", "kmer_table", "kmer_dict_path", "preprocessing"):
                value.pop(field_name, None)
            if isinstance(value.get("args"), Mapping):
                value["args"] = dict(value["args"])
                for field_name in ("chemistry", "seq_type", "pore_type", "kmer_table", "kmer_dict_path", "preprocessing",
                                   "_preprocessing_resolved_paths", "_preprocessing_resolved_table_path"):
                    value["args"].pop(field_name, None)
            checkpoint[key] = value
    return checkpoint
