"""Portable inference bundles containing checksummed model components.

A bundle is an uncompressed ZIP64 archive (conventionally named ``*.pt``),
with a JSON manifest and embedded model artifacts. Streaming copies bound
memory use even for multi-GB calibrations. Components may first be compacted
by ``model_bundle_export``; this layer preserves those bytes and their hashes.
Existing model loaders remain unchanged.
"""

from contextlib import contextmanager
from copy import copy
from functools import wraps
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile


FORMAT = 'porefm_inference_bundle'
VERSION = 1
MANIFEST = 'porefm_bundle.json'
SCORERS = ('nearest-prototype', 'centroid-l2', 'centroid-cosine', 'centroid-rbf',
           'diagonal-gaussian', 'per-kmer-lda', 'residual-effects')
_CHUNK_BYTES = 8 * 1024 * 1024


def _copy_and_hash(source, destination):
    """Stream bytes between open files and return their SHA-256 and size."""
    digest, size = hashlib.sha256(), 0
    while chunk := source.read(_CHUNK_BYTES):
        destination.write(chunk)
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _validate_manifest(manifest):
    """Reject unknown versions, incomplete model sets, and unsafe members."""
    if not isinstance(manifest, dict) or manifest.get('format') != FORMAT:
        raise ValueError('Not a PoreFM inference bundle')
    if manifest.get('version') != VERSION:
        raise ValueError(f"Unsupported inference bundle version: {manifest.get('version')}")
    components = manifest.get('components')
    settings = manifest.get('settings', {})
    if not isinstance(components, dict) or not isinstance(settings, dict):
        raise ValueError('Bundle components and settings must be mappings')
    if manifest.get('kind') == 'zero-shot':
        required = {'canonical_null_model_path', 'calibration_path'}
        allowed = required | {'whitening_artifact'}
        if settings:
            raise ValueError('Zero-shot bundles do not define prototype settings')
    elif manifest.get('kind') == 'prototype':
        scorer = settings.get('scorer')
        if scorer not in SCORERS:
            raise ValueError('Prototype bundle requires a supported scorer')
        if scorer == 'residual-effects':
            required = allowed = {'model_path', 'effect_model'}
            if set(settings) - {'scorer'}:
                raise ValueError('Unknown residual-effects bundle settings')
        else:
            labels = settings.get('class_labels')
            if (not isinstance(labels, list) or len(labels) < 2
                    or any(not isinstance(label, str) or not label.strip() for label in labels)
                    or len(set(labels)) != len(labels)):
                raise ValueError('Prototype bundle requires at least two unique class labels')
            required = allowed = {'model_path', *(f'prototype_{i}' for i in range(len(labels)))}
            if set(settings) - {'scorer', 'class_labels'}:
                raise ValueError('Unknown prototype bundle settings')
    else:
        raise ValueError(f"Unsupported inference bundle kind: {manifest.get('kind')!r}")
    if not required <= components.keys() or components.keys() - allowed:
        raise ValueError(f'Bundle requires components {sorted(required)}; got {sorted(components)}')
    for name, entry in components.items():
        if not isinstance(entry, dict) or entry.get('member') != f'artifacts/{name}.pt':
            raise ValueError(f'Invalid archive member for bundle component {name}')
        digest, size = entry.get('sha256'), entry.get('size')
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)
                or not isinstance(size, int) or size < 0):
            raise ValueError(f'Invalid checksum/size for bundle component {name}')


def read_bundle_manifest(path, *, required=False):
    """Read lightweight bundle metadata without importing torch or unpickling.

    Ordinary checkpoints return ``None`` unless ``required`` is true. A ZIP
    containing our manifest is always validated; malformed bundles cannot fall
    back to separate model arguments. Component hashes are checked on extraction.
    """
    path = Path(path).expanduser()
    if not path.is_file() or not zipfile.is_zipfile(path):
        if required:
            raise ValueError(f'{path} is not a PoreFM inference bundle')
        return None
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if MANIFEST not in names:
            if required:
                raise ValueError(f'{path} is not a PoreFM inference bundle')
            return None
        if len(names) != len(set(names)):
            raise ValueError(f'Duplicate members in bundle {path}')
        if archive.getinfo(MANIFEST).file_size > 1024 * 1024:
            raise ValueError('Bundle manifest exceeds 1 MiB')
        manifest = json.loads(archive.read(MANIFEST))
        _validate_manifest(manifest)
        for name, entry in manifest['components'].items():
            if entry['member'] not in names or archive.getinfo(entry['member']).file_size != entry['size']:
                raise ValueError(f'Missing or truncated bundle component {name}')
    return manifest


def create_model_bundle(output, *, kind, components, settings=None, source_provenance=None):
    """Package existing artifacts without changing or overwriting any file.

    Parameters
    ----------
    output : path-like
        New bundle path; existing destinations are always rejected.
    kind : {'zero-shot', 'prototype'}
        Inference workflow that consumes this bundle.
    components : mapping of str to path-like
        CLI model-path fields, or numbered ``prototype_N`` files for per-class
        prototypes. Every source is embedded in full, with a recorded SHA-256.
    settings : dict, optional
        Prototype scorer and class labels identifying the embedded model set.
        Runtime scoring options are not stored in the bundle.
    source_provenance : dict, optional
        Original source paths and SHA-256 values for compacted components.
        Embedded checksums always describe the bytes actually packaged.

    Returns
    -------
    dict
        Manifest stored inside the archive. A private staging file prevents
        incomplete archives from appearing at the requested destination.
    """
    output = Path(output).expanduser()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    sources = {name: Path(path).expanduser().resolve(strict=True) for name, path in components.items()}
    manifest = dict(format=FORMAT, version=VERSION, kind=kind, settings=settings or {}, components={
        name: dict(member=f'artifacts/{name}.pt', sha256='0' * 64, size=path.stat().st_size,
                   source=str(path)) for name, path in sources.items()
    })
    for name, provenance in (source_provenance or {}).items():
        manifest['components'][name].update(source=provenance['source'],
                                           source_sha256=provenance['source_sha256'])
    _validate_manifest(manifest)
    # Staging beside the output permits an atomic, no-clobber hard-link publish.
    with tempfile.TemporaryDirectory(prefix='.porefm-bundle-', dir=output.parent) as staging:
        pending = Path(staging) / 'bundle.pt'
        with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, path in sources.items():
                entry = manifest['components'][name]
                print(f'Bundling {name}: {path} ({entry["size"]:,} bytes)', flush=True)
                with path.open('rb') as source, archive.open(entry['member'], 'w', force_zip64=True) as dest:
                    entry['sha256'], entry['size'] = _copy_and_hash(source, dest)
            archive.writestr(MANIFEST, json.dumps(manifest, indent=2) + '\n')
        output.hardlink_to(pending)
    return manifest


def apply_bundle_arguments(args, kind):
    """Make embedded model selection authoritative before CLI validation.

    Model fields become descriptive bundle member references until extraction.
    Input/output and runtime scoring options remain caller-controlled.
    Return the manifest or ``None`` for the ordinary separate-file interface.
    """
    path = getattr(args, 'model_bundle_path', None) or getattr(args, 'model_path', None)
    if not path:
        return None
    manifest = read_bundle_manifest(path, required=(kind == 'zero-shot'))
    if manifest is None:
        return None
    if manifest['kind'] != kind:
        raise ValueError(f"Expected a {kind} bundle, got {manifest['kind']}")
    args.model_bundle_path = str(Path(path).expanduser().resolve())
    args.model_bundle_manifest = manifest
    members = {name: f'{args.model_bundle_path}::{entry["member"]}'
               for name, entry in manifest['components'].items()}
    if kind == 'zero-shot':
        args.canonical_null_model_path = members['canonical_null_model_path']
        args.calibration_path = members['calibration_path']
        args.whitening_artifact = members.get('whitening_artifact', '')
    else:
        settings = manifest['settings']
        args.scorer = settings['scorer']
        args.effect_model = members.get('effect_model')
        args.prototype_embeddings = None
        args.class_labels = None
        if args.scorer == 'residual-effects':
            args.class_priors = None
            args.classifier_output = None
            args.metric = None
        else:
            args.class_labels = settings['class_labels']
            args.prototype_embeddings = [members[f'prototype_{i}'] for i in range(len(args.class_labels))]
    return manifest


@contextmanager
def resolved_model_bundle(args, kind):
    """Yield inference arguments backed by verified private temporary files.

    Temporary storage (under ``TMPDIR`` when set) must hold the embedded files.
    It is released on success or failure. No original artifact is needed at
    inference time. Non-bundle calls retain the original Namespace and behavior.
    """
    resolved = copy(args)
    manifest = apply_bundle_arguments(resolved, kind)
    if manifest is None:
        yield args
        return
    with tempfile.TemporaryDirectory(prefix='porefm-model-') as directory:
        paths = {}
        with zipfile.ZipFile(resolved.model_bundle_path) as archive:
            for name, entry in manifest['components'].items():
                dest = Path(directory) / f'{name}.pt'
                with archive.open(entry['member']) as source, dest.open('xb') as target:
                    digest, size = _copy_and_hash(source, target)
                if digest != entry['sha256'] or size != entry['size']:
                    raise ValueError(f'Checksum mismatch for bundle component {name}')
                paths[name] = str(dest)
        if kind == 'zero-shot':
            # Preprocessing must inspect the embedded checkpoint, not the ZIP container.
            resolved.model_path = None
            resolved.canonical_null_model_path = paths['canonical_null_model_path']
            resolved.calibration_path = paths['calibration_path']
            resolved.whitening_artifact = paths.get('whitening_artifact', '')
        else:
            resolved.model_path = paths['model_path']
            resolved.effect_model = paths.get('effect_model')
            if resolved.scorer != 'residual-effects':
                resolved.prototype_embeddings = [paths[f'prototype_{i}'] for i in range(len(resolved.class_labels))]
        yield resolved


def use_model_bundle(kind):
    """Wrap an inference entry point with bundle resolution and cleanup."""
    def decorate(function):
        @wraps(function)
        def run(args):
            with resolved_model_bundle(args, kind) as resolved:
                return function(resolved)
        return run
    return decorate
