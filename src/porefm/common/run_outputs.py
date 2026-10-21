"""Reserve fresh run destinations and record invocation provenance.

These guards belong at workflow startup, before models, logs, or writers are
opened. They do not change checkpoint contents or model execution.
"""

from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import uuid


def add_output_safety_argument(parser):
    """Expose explicit permission to reuse occupied output destinations."""
    parser.add_argument(
        "--force-overwrite", action="store_true",
        help=("Check existing output destinations, warn, and allow overwriting matching files "
              "and run manifests directly. No backup or directory cleanup is "
              "performed. Otherwise destinations must be fresh, including "
              "when loading a checkpoint."),
    )


def _path(value):
    """Resolve aliases without accepting a missing/blank destination."""
    if value is None or not str(value).strip():
        raise ValueError("An explicit output destination is required")
    return Path(value).expanduser().absolute()


def _strings(value):
    """Walk config/argument containers without interpreting dictionary keys."""
    if isinstance(value, (str, os.PathLike)):
        yield os.fspath(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _input_records(arguments, output_arguments):
    """Snapshot JSON configs and identify existing directly declared inputs.

    Large input files are identified by canonical path, size, and mtime, not
    hashed. JSON config text is embedded and hashed so later edits cannot erase
    the invocation's configuration. Existing paths inside configs are protected
    too; relative paths are checked against both cwd and the config directory.
    """
    records = {}

    def record(value, relative_to=None):
        if not value:
            return None
        path = Path(value).expanduser()
        if relative_to is not None and not path.is_absolute():
            path = relative_to / path
        try:
            if not path.exists():
                return None
        except OSError as exc:
            # Config strings can contain long biological sequences rather than
            # paths. Ignore only invalid-path errors; preserve real I/O errors.
            if exc.errno in {errno.ENAMETOOLONG, errno.EINVAL}:
                return None
            raise
        path = path.resolve()
        key = str(path)
        if key not in records:
            stat = path.stat()
            records[key] = {
                "path": key, "kind": "directory" if path.is_dir() else "file",
                "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            }
        return path

    for key, value in arguments.items():
        if key in output_arguments or key == "force_overwrite":
            continue
        for text in _strings(value):
            path = record(text)
            is_config = key.endswith("_config") or key == "manifest"
            if path is not None and path.is_file() and (is_config or path.suffix.lower() == ".json"):
                raw = path.read_bytes()
                config = json.loads(raw)
                records[str(path)].update(
                    sha256=hashlib.sha256(raw).hexdigest(),
                    content=config,
                )
                for nested in _strings(config):
                    record(nested)
                    record(nested, path.parent)
    return list(records.values())


def _occupied(path, directory):
    if path.is_symlink():
        raise ValueError(f"Output destination must not be a symlink: {path}")
    if directory and path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Expected an output directory: {path}")
        return any(path.iterdir())
    if not directory and path.is_dir():
        raise IsADirectoryError(f"Expected an output file: {path}")
    return path.exists()


def prepare_run_outputs(args, *, command, directories=None, files=None,
                        output_arguments=(), distributed=None):
    """Reserve outputs and write startup manifests on rank zero.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments, including optional ``force_overwrite``.
    command : str
        Stable workflow identifier recorded separately from process argv.
    directories, files : dict[str, path-like], optional
        Output option names mapped to their active destinations. Directories
        must be empty; files and their ``.run_manifest.json`` sidecars must be
        absent unless ``force_overwrite`` is set. BAM index companions also
        count as occupied outputs. Nested directories are supported and
        duplicates merged.
    output_arguments : iterable[str]
        Additional output-only argument names to exclude from input discovery.
    distributed : module-like, optional
        Initialized ``torch.distributed`` interface. Rank zero performs I/O and
        broadcasts success or failure before any rank starts a writer/model.

    Returns
    -------
    dict
        Startup manifest. A failed/aborted run keeps its reservation; retry in
        a fresh destination or explicitly reuse it with ``--force-overwrite``.
        Forced runs warn about occupied paths, overwrite manifests, and permit writers to replace matching
        files. They do not back up or delete files; stale outputs and indexes
        remain the caller's responsibility.
    """
    if distributed is not None and distributed.is_initialized():
        result = [None]
        if distributed.get_rank() == 0:
            try:
                result[0] = {"manifest": prepare_run_outputs(
                    args, command=command, directories=directories, files=files,
                    output_arguments=output_arguments,
                )}
            except Exception as exc:
                result[0] = {"error": f"{type(exc).__name__}: {exc}"}
        distributed.broadcast_object_list(result, src=0)
        if "error" in result[0]:
            raise RuntimeError("Output preparation failed: " + result[0]["error"])
        return result[0]["manifest"]

    directories, files = directories or {}, files or {}
    if not directories and not files:
        return {}
    arguments = vars(args).copy()
    excluded = set(output_arguments) | set(directories) | set(files)
    inputs = _input_records(arguments, excluded)
    dirs = {_path(value).resolve() for value in directories.values()}
    # Check the final path itself before resolving so symlinks cannot silently
    # turn a result destination into another run's directory or artifact.
    for value in [*directories.values(), *files.values()]:
        if _path(value).is_symlink():
            raise ValueError(f"Output destination must not be a symlink: {value}")
    artifacts = {_path(value).resolve() for value in files.values()}
    companions = set()
    for path in artifacts:
        if path.suffix.lower() == ".bam":
            companions.update({Path(str(path) + ".bai"), Path(str(path) + ".csi"), path.with_suffix(".bai")})
    sidecars = {path.with_name(path.name + ".run_manifest.json") for path in artifacts}
    manifests = {path / "run_manifest.json" for path in dirs} | sidecars
    if artifacts & (dirs | manifests) or dirs & (artifacts | manifests | companions) or manifests & companions:
        raise ValueError("Output file, directory, and manifest destinations collide")
    # Reject file ancestors as well as equality (e.g. --output foo and foo/bar).
    for path in artifacts | manifests | companions:
        if any(path in other.parents for other in dirs | artifacts | manifests | companions):
            raise ValueError(f"An output file is also an output parent: {path}")
    destinations = dirs | artifacts | manifests | companions
    force = bool(getattr(args, "force_overwrite", False))
    input_paths = {Path(item["path"]) for item in inputs}
    protected_roots = {Path.cwd().resolve(), Path(__file__).resolve().parents[3]}
    for path in destinations:
        contains_input = any(path == source or path in source.parents for source in input_paths)
        contains_root = any(path == source or path in source.parents for source in protected_roots)
        # Direct file aliases remain invalid. Forced directory reuse can load a
        # checkpoint from that directory because preparation no longer moves it.
        if contains_root or (contains_input and (path not in dirs or not force)):
            raise ValueError(f"Output would replace a declared input or working/repository directory: {path}")

    # Preflight every destination before writing any manifests.
    occupied = []
    for path in sorted(destinations):
        if _occupied(path, path in dirs):
            occupied.append(path)
            if not force:
                raise FileExistsError(
                    f"Output destination is non-empty or already exists: {path}. "
                    "Choose a fresh path (also when continuing from a checkpoint), "
                    "or use --force-overwrite to allow direct overwrites without backups."
                )

    # Model package imports initialize PyTorch; CLI help only needs the flag.
    from porefm.models.checkpointing import build_repository_provenance

    run_id = uuid.uuid4().hex
    manifest = {
        "schema_version": 1, "run_id": run_id, "command": command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv), "cwd": str(Path.cwd().resolve()),
        "hostname": socket.gethostname(), "pid": os.getpid(),
        "arguments": arguments, "inputs": inputs,
        "outputs": {"directories": {key: str(_path(value).resolve()) for key, value in directories.items()},
                    "files": {key: str(_path(value).resolve()) for key, value in files.items()},
                    "companions": sorted(map(str, companions))},
        "repository_provenance": build_repository_provenance(),
    }

    # Serialize before any mutation so unsupported argument values fail early.
    serialized = json.dumps(manifest, indent=2, sort_keys=True, default=os.fspath) + "\n"
    if occupied:
        print(
            "WARNING: --force-overwrite permits reuse of occupied output destinations:\n"
            + "\n".join(f"  {path}" for path in occupied)
            + "\nMatching files and run manifests may be overwritten. No backups or "
              "cleanup will be performed; stale files may remain. Continuing.",
            file=sys.stderr, flush=True,
        )
    for path in sorted(dirs):
        path.mkdir(parents=True, exist_ok=True)
    for path in sorted(manifests):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents two normal invocations from claiming
        # the same empty directory/file between the check and manifest write.
        with path.open("w" if force else "x", encoding="utf-8") as handle:
            handle.write(serialized)
    return manifest
