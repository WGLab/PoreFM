"""Persistent, unfiltered BAM offsets for POD5 inference joins."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

import pysam


def bam_signature(path):
    """Identify a BAM by size, nanosecond mtime and a hash of its first/last MiB.

    This inexpensive stale-index check deliberately avoids rescanning the BAM.
    It is not a full-file checksum; callers must not modify BAMs during use.
    """
    path = Path(os.fsdecode(path))
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        handle.seek(max(0, stat.st_size - 1024 * 1024))
        digest.update(handle.read())
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                edge_sha256=digest.hexdigest())


def default_bam_index_path(bam_path):
    """Append ``.pfmi`` to the complete BAM path, preserving its filename."""
    return Path(str(Path(bam_path)) + ".pfmi")


def build_bam_index(bam_path, output=None, *, threads=1):
    """Save every BAM record's BGZF virtual offset in a new SQLite index.

    No flags, quality, length, region, alignment or tag-presence filters apply.
    Lookup keys are ``pi`` parent IDs for Dorado children, otherwise query names.
    Duplicate keys retain every record. Records with missing names are retained
    with a NULL key. The index contains no signal, sequence or inference settings.

    Parameters
    ----------
    bam_path : path-like
        One BAM file, opened read-only.
        Neither a coordinate index nor coordinate sorting is required.
    output : path-like, optional
        New index path. Defaults to ``<bam_path>.pfmi`` alongside the BAM.
        Existing outputs are never overwritten, and paths aliasing the input
        BAM (including symbolic and hard links) are rejected.
    threads : int, default=1
        BAM decompression threads used for the sequential build.

    Returns
    -------
    dict
        Index metadata, record/child counts, build seconds and output size.

    Notes
    -----
    Construction uses a temporary sibling file and publishes the complete
    index atomically. Input changes during construction abort publication.
    Each inference worker opens its own read-only SQLite connection.
    """
    bam_path = Path(bam_path)
    output = default_bam_index_path(bam_path) if output is None else Path(output)
    if not bam_path.is_file():
        raise ValueError(f"BAM indexing requires one BAM file: {bam_path}")
    if threads < 1:
        raise ValueError("threads must be at least 1")
    if output.resolve() == bam_path.resolve() or (output.exists() and output.samefile(bam_path)):
        raise ValueError(f"Index output must be different from the input BAM: {output}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Index output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    signature = bam_signature(bam_path)
    started = time.perf_counter()
    count = children = 0
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".partial", dir=output.parent)
    os.close(fd)
    connection = None
    try:
        connection = sqlite3.connect(temporary)
        # Only the unpublished file uses relaxed durability; no existing index
        # can be damaged by an interrupted build.
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-65536")
        connection.execute("CREATE TABLE reads (voffset INTEGER PRIMARY KEY, read_id TEXT)")
        connection.execute("CREATE TABLE metadata (value TEXT NOT NULL)")
        print(f"Indexing every BAM record in {bam_path}", flush=True)
        with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False, threads=threads) as bam:
            if not bam.is_bam:
                raise ValueError("BAM indexing supports BAM files only")
            rows = []
            while True:
                offset = bam.tell()
                try:
                    record = next(bam)
                except StopIteration:
                    break
                child = record.has_tag("pi")
                key = record.get_tag("pi") if child else record.query_name
                rows.append((offset, key))
                count += 1
                children += int(child)
                if len(rows) == 8192:
                    connection.executemany("INSERT INTO reads VALUES (?, ?)", rows)
                    rows.clear()
                if count % 1_000_000 == 0:
                    print(f"Indexed {count:,} records in {time.perf_counter() - started:.1f}s", flush=True)
            connection.executemany("INSERT INTO reads VALUES (?, ?)", rows)
        print(f"Building lookup table for {count:,} records", flush=True)
        connection.execute("CREATE INDEX by_read_id ON reads (read_id)")
        if bam_signature(bam_path) != signature:
            raise ValueError("BAM changed while its index was being built")
        metadata = dict(format="porefm-bam-index", version=1, bam=str(bam_path.resolve()),
                        signature=signature, records=count, children=children)
        connection.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
        connection.commit()
        connection.close()
        connection = None
        # Unlike replace(), link() cannot overwrite a competing build's output.
        os.link(temporary, output)
        return dict(metadata, seconds=time.perf_counter() - started,
                    index_bytes=output.stat().st_size, output=str(output))
    finally:
        if connection is not None:
            connection.close()
        Path(temporary).unlink(missing_ok=True)


class BamReadIndex:
    """Read-only offset lookup tied to an unchanged BAM file.

    Open a separate instance in each process. SQLite's bounded cache and mapped
    pages avoid reconstructing the complete index as a Python dictionary.
    """

    def __init__(self, path, bam_path):
        self.connection = None
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"BAM read index does not exist: {path}")
        try:
            self.connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            self.connection.execute("PRAGMA cache_size=-16384")
            self.connection.execute("PRAGMA mmap_size=268435456")
            # Keep one read snapshot for this immutable-by-contract index.
            # Otherwise every lookup repeats locking and file-change checks,
            # which is particularly expensive on network filesystems.
            self.connection.execute("BEGIN")
            row = self.connection.execute("SELECT value FROM metadata").fetchone()
            self.metadata = json.loads(row[0]) if row else {}
            if self.metadata.get("format") != "porefm-bam-index" or self.metadata.get("version") != 1:
                raise ValueError(f"Unsupported BAM index format: {path}")
            if self.metadata.get("signature") != bam_signature(bam_path):
                raise ValueError(f"BAM index is stale or belongs to a different BAM: {path}")
        except (sqlite3.DatabaseError, json.JSONDecodeError) as exc:
            self.close()
            raise ValueError(f"Invalid BAM read index {path}: {exc}") from exc
        except BaseException:
            self.close()
            raise

    def find(self, read_id):
        """Return all matching record offsets in original BAM order."""
        return [row[0] for row in self.connection.execute(
            "SELECT voffset FROM reads WHERE read_id = ? ORDER BY voffset", (str(read_id),)
        )]

    def close(self):
        """Release this process's connection and mapped pages."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
