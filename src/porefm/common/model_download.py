"""Download versioned public model files over HTTPS using only the standard library.

The cache belongs to PoreFM, not the Hugging Face SDK. A release manifest selects
the checkpoint and supplies its exact size and SHA-256. Downloads are pinned to
the manifest's resolved commit, serialized across processes, and atomically
published only after verification. Cached releases never require network access.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


_REFERENCE = re.compile(
    r'(?P<repo>[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)'
    r'@(?P<tag>v(?P<version>(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)'
    r'(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?(?:\+[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?))'
)
_HEX256 = re.compile(r'[0-9a-f]{64}')
_COMMIT = re.compile(r'[0-9a-f]{40}')
_CHUNK = 8 * 1024 * 1024
_MANIFEST_LIMIT = 1024 * 1024
_ROLES = {'backbone', 'pretraining', 'supervised_classifier', 'zero-shot', 'prototype'}


@dataclass(frozen=True)
class ModelReference:
    """A public repository and explicit release tag, such as ``WGLab/model@v0.1.0``."""

    repository: str
    tag: str
    version: str


@dataclass(frozen=True)
class DownloadedModel:
    """Verified local checkpoint plus portable release provenance."""

    path: Path
    provenance: dict


def parse_model_reference(value):
    """Validate ``OWNER/REPOSITORY@vMAJOR.MINOR.PATCH`` and return its parts.

    Optional prerelease/build suffixes are accepted. Local paths, unversioned
    names, branch names, and URL/path traversal syntax are rejected.
    """
    match = _REFERENCE.fullmatch(value) if isinstance(value, str) else None
    if match is None or '..' in value or '--' in value.split('@', 1)[0]:
        raise ValueError('Model must be OWNER/REPOSITORY@vVERSION, for example '
                         'WGLab/PoreFM-RNA_prototype@v0.1.0')
    return ModelReference(match['repo'], match['tag'], match['version'])


def model_cache_directory(cache_dir=None):
    """Resolve CLI override, POREFM_MODEL_CACHE, then the XDG/home cache default."""
    value = cache_dir or os.environ.get('POREFM_MODEL_CACHE')
    if value:
        return Path(value).expanduser().absolute()
    root = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache')
    return root.expanduser().absolute() / 'porefm' / 'models'


def _url(reference, revision, filename):
    return ('https://huggingface.co/' + reference.repository + '/resolve/'
            + quote(revision, safe='') + '/' + quote(filename, safe=''))


@contextmanager
def _release_lock(directory):
    """Serialize POSIX processes sharing a release cache; release on process exit."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _request(url):
    """Open an anonymous public-file request, following HTTPS download redirects."""
    return urlopen(Request(url, headers={'User-Agent': 'PoreFM/model-download'}), timeout=60)


def _retry(operation, url):
    """Retry transient transfers three times; fail clearly on missing releases."""
    for attempt in range(3):
        try:
            return operation()
        except HTTPError as exc:
            if exc.code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                raise ValueError(f'Cannot download {url}: HTTP {exc.code}. '
                                 'Check the public repository, release tag, and filename.') from exc
        except (URLError, TimeoutError, IncompleteRead) as exc:
            if attempt == 2:
                raise OSError(f'Cannot download {url}: {exc}. '
                              'Prefetch on a connected machine or supply --model-path.') from exc
        time.sleep(attempt + 1)


def _atomic_bytes(path, data):
    """Publish a small cache metadata file without exposing partial JSON."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as handle:
        pending = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    try:
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _validate_manifest(manifest, reference):
    """Validate public release metadata before selecting or downloading any weights."""
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1:
        raise ValueError('Unsupported model release manifest schema_version')
    if manifest.get('model_version') != reference.version:
        raise ValueError(f'Release manifest version does not match {reference.tag}')
    artifacts = manifest.get('artifacts')
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError('Release manifest must contain checkpoint artifacts')
    names = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError('Invalid release artifact')
        name, size, digest = (artifact.get(key) for key in ('file', 'size_bytes', 'file_sha256'))
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.pt', name)
                or '..' in name or name in names):
            raise ValueError('Release checkpoint filenames must be unique plain .pt filenames')
        if (type(size) is not int or size <= 0 or not isinstance(digest, str)
                or _HEX256.fullmatch(digest) is None or artifact.get('role') not in _ROLES):
            raise ValueError(f'Invalid size, SHA-256, or role for {name}')
        expected_format = ('porefm_inference_bundle' if artifact['role'] in {'zero-shot', 'prototype'}
                           else 'pytorch_checkpoint')
        if artifact.get('format') != expected_format:
            raise ValueError(f'Unexpected artifact format for {name}')
        names.add(name)


def _release_manifest(directory, reference, offline):
    """Reuse local metadata or fetch it once and pin downloads to its HF commit."""
    manifest_path, source_path = directory / 'manifest.json', directory / 'source.json'
    if manifest_path.is_file() and source_path.is_file():
        try:
            raw = manifest_path.read_bytes()
            source = json.loads(source_path.read_bytes())
            if (source['repository'] != reference.repository or source['tag'] != reference.tag
                    or _COMMIT.fullmatch(source['commit']) is None
                    or source['manifest_sha256'] != hashlib.sha256(raw).hexdigest()):
                raise ValueError('Invalid cached release metadata')
            manifest = json.loads(raw)
            _validate_manifest(manifest, reference)
            return manifest, source
        except (ValueError, KeyError, TypeError) as exc:
            if offline:
                raise ValueError(f'Invalid cached release metadata in {directory}') from exc
    if offline:
        raise FileNotFoundError(f'Model release is not cached in {directory}; '
                                'run porefm models download --model OWNER/REPOSITORY@vVERSION first')
    url = _url(reference, reference.tag, 'manifest.json')

    def read():
        with _request(url) as response:
            raw = response.read(_MANIFEST_LIMIT + 1)
            commit = response.headers.get('X-Repo-Commit', '')
        if len(raw) > _MANIFEST_LIMIT:
            raise ValueError('Model release manifest exceeds 1 MiB')
        if _COMMIT.fullmatch(commit) is None:
            raise ValueError('Download response is missing the resolved repository commit')
        manifest = json.loads(raw)
        _validate_manifest(manifest, reference)
        return raw, commit, manifest

    raw, commit, manifest = _retry(read, url)
    source = dict(repository=reference.repository, tag=reference.tag, commit=commit,
                  manifest_sha256=hashlib.sha256(raw).hexdigest())
    _atomic_bytes(manifest_path, raw)
    _atomic_bytes(source_path, (json.dumps(source, indent=2) + '\n').encode())
    return manifest, source


def _select_artifact(manifest, roles, filename):
    """Select one compatible artifact, defaulting base-model downloads to backbone."""
    artifacts = manifest['artifacts']
    candidates = [item for item in artifacts if (not roles or item['role'] in roles)
                  and (filename is None or item['file'] == filename)]
    if filename is None and len(candidates) > 1:
        candidates = [item for item in candidates if item['role'] == 'backbone']
    if len(candidates) != 1:
        expected = ', '.join(roles) if roles else 'a single checkpoint (or --file FILENAME)'
        raise ValueError(f'Release must provide {expected}; available artifacts: '
                         + ', '.join(f"{item['file']} ({item['role']})" for item in artifacts))
    return candidates[0]


def _verified(path, artifact):
    if not path.is_file() or path.stat().st_size != artifact['size_bytes']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest() == artifact['file_sha256']


def _download_checkpoint(url, path, artifact):
    """Stream and hash a bounded-memory transfer; publish only complete valid bytes."""
    def transfer():
        pending = None
        try:
            with _request(url) as response, tempfile.NamedTemporaryFile(
                    dir=path.parent, prefix='.' + path.name, suffix='.part', delete=False) as target:
                pending = Path(target.name)
                digest, size, last_report = hashlib.sha256(), 0, time.monotonic()
                while chunk := response.read(_CHUNK):
                    size += len(chunk)
                    if size > artifact['size_bytes']:
                        raise ValueError(f'Download exceeds expected size for {path.name}')
                    target.write(chunk)
                    digest.update(chunk)
                    if time.monotonic() - last_report >= 10:
                        print(f"Downloading {path.name}: {size:,}/{artifact['size_bytes']:,} bytes",
                              file=sys.stderr, flush=True)
                        last_report = time.monotonic()
                if size != artifact['size_bytes']:
                    raise IncompleteRead(b'', artifact['size_bytes'] - size)
                if digest.hexdigest() != artifact['file_sha256']:
                    raise ValueError(f'SHA-256 mismatch for {path.name}; download was not installed')
                target.flush()
                os.fsync(target.fileno())
            os.replace(pending, path)
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)
    _retry(transfer, url)


def fetch_model(model, *, cache_dir=None, offline=False, roles=None, filename=None):
    """Return a verified checkpoint from a public ``OWNER/REPO@vVERSION`` release.

    Parameters
    ----------
    model : str
        Explicit versioned model reference. Never falls back to ``main``.
    cache_dir : path-like, optional
        Override POREFM_MODEL_CACHE or the default XDG/home PoreFM model cache.
    offline : bool
        Prohibit every network request; missing/corrupt cached data raises.
    roles : iterable of str, optional
        Allowed manifest artifact roles for the consuming inference command.
    filename : str, optional
        Select a specific manifest artifact, e.g. ``pretraining.pt``. Otherwise
        select the only compatible artifact, preferring ``backbone`` when needed.

    Returns
    -------
    DownloadedModel
        Absolute local path and JSON-serializable repository, version, commit,
        filename, role, size, and checksum provenance. SHA-256 is checked on
        cache hits as well as downloads; no tensors are loaded or modified.
    """
    reference = parse_model_reference(model)
    roles = tuple(roles or ())
    if set(roles) - _ROLES:
        raise ValueError(f'Unknown model roles: {roles}')
    directory = model_cache_directory(cache_dir) / ('models--' + reference.repository.replace('/', '--')) / reference.tag
    with _release_lock(directory):
        manifest, source = _release_manifest(directory, reference, offline)
        artifact = _select_artifact(manifest, roles, filename)
        path = directory / artifact['file']
        if path.exists():
            print(f'Checking cached model: {path}', file=sys.stderr, flush=True)
        if not _verified(path, artifact):
            if offline:
                raise FileNotFoundError(f'Checkpoint missing or checksum mismatch in offline cache: {path}')
            print(f"Downloading {model}: {artifact['file']} ({artifact['size_bytes']:,} bytes)",
                  file=sys.stderr, flush=True)
            _download_checkpoint(_url(reference, source['commit'], artifact['file']), path, artifact)
    provenance = dict(source, model=model, model_version=reference.version,
                      filename=artifact['file'], role=artifact['role'],
                      size_bytes=artifact['size_bytes'], sha256=artifact['file_sha256'])
    return DownloadedModel(path, provenance)
