"""Sequential POD5 signal access joined to primary BAM records for inference."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pod5
import pysam

from porefm.data.bam_index import BamReadIndex


def find_pod5_files(source: str | Path) -> list[Path]:
    """Resolve one POD5 file or recursively searched directory, deterministically.

    Missing paths, non-POD5 files and directories without POD5 files are errors.
    Paths are used literally, without expanding environment variables or ``~``.
    """
    source = Path(source)
    if source.is_dir():
        files = sorted(path for path in source.rglob("*.pod5") if path.is_file())
        # Resolve aliases so a symlink to an input cannot schedule it twice.
        files = list({path.resolve(): path for path in files}.values())
        if not files:
            raise ValueError(f"No .pod5 files found recursively in {source}")
        return files
    if not source.is_file():
        raise FileNotFoundError(f"POD5 input does not exist: {source}")
    if source.suffix != ".pod5":
        raise ValueError(f"POD5 input must be a .pod5 file or directory: {source}")
    return [source]


def validate_pod5_samples(samples):
    """Return discovered files for one POD5 sample; reject mixed/multiple BAMs."""
    if not any(getattr(sample, "pod5", None) for sample in samples):
        return []
    if len(samples) != 1:
        raise ValueError("POD5 inference requires exactly one BAM sample; mixed SG/POD5 inputs are not supported")
    sample = samples[0]
    if not Path(sample.bam_path).is_file():
        raise ValueError("POD5 inference requires a single BAM file, not a directory")
    return find_pod5_files(sample.pod5)


def pod5_worker_count(files, requested, rank=0, world_size=1):
    """Cap DataLoader processes to this rank's whole POD5 files; zero stays zero."""
    if requested < 0:
        raise ValueError("dataloader_workers must be nonnegative")
    return min(requested, len(files[rank::world_size]))


def signal_for_bam_record(record: pysam.AlignedSegment, signal: np.ndarray, *, chunking_mode="auto") -> np.ndarray:
    """Return raw ADC samples anchored at this BAM read's untrimmed origin.

    ``signal`` is the full one-dimensional int16 POD5 parent signal. For a
    Dorado child (``pi``), ``sp`` is a zero-based parent-signal sample offset.
    The returned view starts there; the shared BAM decoder subsequently applies
    the selected move-based or inferred trimming, then RNA reversal. ``ns`` bounds a
    child's signal extent when present. Ordinary reads retain the full signal,
    exactly as the SG converter stores it. No calibration or scaling is applied.

    Invalid offsets and move spans fail explicitly rather than allowing NumPy
    slicing to silently return a truncated signal.
    """
    tags = {key: value for key, value, kind in record.get_tags(with_value_type=True)
            if not (key == "ts" and kind == "A")}
    child = "pi" in tags
    if child and "sp" not in tags:
        raise ValueError(f"Dorado child {record.query_name} has pi but no sp tag")
    offset = tags["sp"] if child else 0
    from porefm.data.chunk_boundaries import uses_inferred_chunks
    if uses_inferred_chunks(record, chunking_mode):
        if record.flag & 0x900:
            raise ValueError("Inferred chunking accepts primary/unmapped records only")
        if not isinstance(offset, int) or offset < 0 or offset >= len(signal):
            raise ValueError(f"Invalid sp offset for read {record.query_name}: {offset!r}")
        extent = tags.get("ns", len(signal) - offset) if child else len(signal)
        if not isinstance(extent, int) or extent <= 0 or offset + extent > len(signal):
            raise ValueError(f"Invalid POD5 signal extent for read {record.query_name}")
        return signal[offset:offset + extent] if child else signal
    start = tags["ts"]
    moves = tags["mv"]
    if not isinstance(offset, int) or offset < 0:
        raise ValueError(f"Invalid sp offset for read {record.query_name}: {offset!r}")
    if not isinstance(start, int) or start < 0 or len(moves) < 2 or moves[0] <= 0:
        raise ValueError(f"Invalid ts/mv signal span for read {record.query_name}")
    required = start + (len(moves) - 1) * moves[0]
    extent = tags.get("ns", required) if child else len(signal)
    if not isinstance(extent, int) or extent < required or offset + extent > len(signal):
        raise ValueError(
            f"Signal bounds for read {record.query_name} exceed POD5 signal: "
            f"offset={offset}, extent={extent}, required={required}, samples={len(signal)}"
        )
    return signal[offset:offset + extent] if child else signal


def iter_pod5_bam_records(
    bam: pysam.AlignmentFile,
    source: str | Path,
    *,
    accept_record: Callable[[pysam.AlignedSegment], bool] | None = None,
    index_path: str | Path | None = None,
    assigned_files: list[Path] | None = None,
    chunking_mode: str = "auto",
) -> Iterator[tuple[pysam.AlignedSegment, np.ndarray]]:
    """Join sequential POD5 reads to primary BAM records, including split children.

    Parameters
    ----------
    bam : pysam.AlignmentFile
        Seekable BAM open for reading. Its position is reset and subsequently
        moved by this generator; no other iterator may use it concurrently.
        Neither coordinate sorting nor a BAI/CSI index is required.
    source : str or pathlib.Path
        POD5 file or directory searched recursively.
    accept_record : callable, optional
        Inference filters evaluated before signal decoding. With a saved index,
        filters apply to retrieved records; otherwise they apply during indexing.
        Secondary/supplementary records are excluded. Records lacking ``mv``
        are excluded only in explicit mv mode.
    index_path : path-like, optional
        Unfiltered SQLite index created by ``porefm utils index-bam``. Opened read-only
        in this worker. Omission builds an in-memory index in this worker.
    assigned_files : list of pathlib.Path, optional
        This worker's disjoint subset of discovered POD5 files. An empty list
        produces no records and does not open or build an index.
    chunking_mode : {"auto", "mv", "hierarchical", "dtw", "inferred"}, default="auto"
        Match the shared BAM decoder's boundary policy. Inferred/auto joins
        retain primary records without move or trimming tags; split-child
        parent offsets are still applied before anchor estimation.

    Yields
    ------
    tuple of pysam.AlignedSegment and numpy.ndarray
        Original BAM record and one-dimensional int16 raw signal in that read's
        untrimmed coordinate system. Children retain their original query names
        and tags. Records follow POD5 order, not genomic order.

    Notes
    -----
    Parent IDs (``pi``) or ordinary query names map to BAM virtual offsets.
    Multiple primary children of one parent are all retained.
    Each matching parent signal is decompressed once and shared by its children;
    it is released as iteration advances. IDs absent from the BAM index are
    skipped without signal decompression. Missing BAM reads in the supplied
    POD5 subset are naturally omitted. Repeated IDs within this worker are
    suppressed. Files assigned to different workers must have disjoint IDs.
    """
    files = find_pod5_files(source) if assigned_files is None else assigned_files
    if not files:
        return
    if index_path is not None:
        with BamReadIndex(index_path, bam.filename) as index:
            print(f"{datetime.datetime.now()}: Using saved BAM index {index_path}; traversing {len(files)} POD5 files.", flush=True)
            seen = set()
            for path in files:
                with pod5.Reader(path) as reader:
                    for pod5_read in reader.reads():
                        key = str(pod5_read.read_id)
                        if key in seen:
                            continue
                        positions = index.find(key)
                        if not positions:
                            continue
                        seen.add(key)
                        signal = None
                        for position in positions:
                            bam.seek(position)
                            record = next(bam)
                            if record.flag & 0x900 or (chunking_mode == "mv" and not record.has_tag("mv")):
                                continue
                            if accept_record is not None and not accept_record(record):
                                continue
                            if signal is None:
                                signal = pod5_read.signal
                            yield record, signal_for_bam_record(record, signal, chunking_mode=chunking_mode)
        return
    offsets: dict[str, int | list[int]] = {}
    count = 0
    children = 0
    bam.reset()
    print(f"{datetime.datetime.now()}: Indexing primary BAM records for POD5 inference.", flush=True)
    while True:
        offset = bam.tell()
        try:
            record = next(bam)
        except StopIteration:
            break
        if record.flag & 0x900 or (chunking_mode == "mv" and not record.has_tag("mv")):
            continue
        if accept_record is not None and not accept_record(record):
            continue
        is_child = record.has_tag("pi")
        key = record.get_tag("pi") if is_child else record.query_name
        previous = offsets.get(key)
        if previous is None:
            offsets[key] = offset
        elif isinstance(previous, int):
            offsets[key] = [previous, offset]
        else:
            previous.append(offset)
        count += 1
        children += int(is_child)
    print(
        f"{datetime.datetime.now()}: Indexed {count} primary records "
        f"({children} Dorado children); traversing {len(files)} POD5 files.",
        flush=True,
    )
    if not offsets:
        return
    for path in files:
        with pod5.Reader(path) as reader:
            for pod5_read in reader.reads():
                positions = offsets.pop(str(pod5_read.read_id), None)
                if positions is None:
                    continue
                signal = pod5_read.signal
                for position in [positions] if isinstance(positions, int) else positions:
                    bam.seek(position)
                    record = next(bam)
                    yield record, signal_for_bam_record(record, signal, chunking_mode=chunking_mode)
                if not offsets:
                    return
