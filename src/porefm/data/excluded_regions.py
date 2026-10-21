"""Fast read exclusion from three- or four-column region files."""

from collections import defaultdict
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np


class ExcludedRegions:
    """Index ``chrom start end [strand]`` rows with one NCLS per contig/strand.

    Parameters
    ----------
    path : str or pathlib.Path
        UTF-8 file with three or four whitespace-separated columns, zero-based
        half-open coordinates, and optional ``+``/``-`` strand. Three-column
        rows apply to both strands. Blank lines and lines
        starting with ``#`` are ignored. Invalid rows raise ``ValueError``
        including the filename and line number. An empty file excludes nothing.

    Notes
    -----
    Coordinates remain signed 64-bit integers; separate chromosome/strand
    keys avoid coordinate offsets and floating-point arithmetic. Instances
    contain native NCLS objects and must be built inside each loader process,
    rather than attached to samples sent to spawned workers. ``sha256`` records
    the source contents for comparison run provenance.
    """

    def __init__(self, path):
        from ncls import NCLS

        content = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(content).hexdigest()
        intervals = defaultdict(list)
        for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            error = f"{path}:{line_number}: "
            if len(fields) not in (3, 4):
                raise ValueError(error + "expected three or four columns: chrom start end [strand]")
            chrom, start, end = fields[:3]
            strand = fields[3] if len(fields) == 4 else None
            try:
                start, end = int(start), int(end)
            except ValueError as exc:
                raise ValueError(error + "start and end must be integers") from exc
            if not 0 <= start < end <= np.iinfo(np.int64).max:
                raise ValueError(error + "require 0 <= start < end within signed 64-bit coordinates")
            if strand is not None and strand not in ("+", "-"):
                raise ValueError(error + "strand must be + or - in column 4")
            for reverse in ((False, True) if strand is None else (strand == "-",)):
                intervals[(chrom, reverse)].append((start, end))

        self._trees = {}
        for key, rows in intervals.items():
            coords = np.asarray(rows, dtype=np.int64)
            self._trees[key] = NCLS(
                coords[:, 0].copy(), coords[:, 1].copy(),
                np.arange(len(rows), dtype=np.int64),
            )

    def overlaps_read(self, bam_read):
        """Return whether a mapped read's full reference span hits its strand.

        Uses BAM alignment strand, including for RNA. The half-open span
        includes deletions and skipped reference regions (introns). Merely
        touching a boundary is not an overlap. Unmapped records and records
        without a nonempty reference span do not match. ``has_overlap`` stops
        at a hit without constructing a list of overlapping intervals.
        """
        if bam_read.is_unmapped:
            return False
        tree = self._trees.get((bam_read.reference_name, bam_read.is_reverse))
        if tree is None:
            return False
        start, end = bam_read.reference_start, bam_read.reference_end
        return end is not None and start >= 0 and end > start and tree.has_overlap(start, end)


def get_excluded_regions(path):
    """Return a process-cached index, or ``None`` when exclusion is disabled.

    Only a path needs to cross worker boundaries. Reuse indexes across BAMs
    and dataset iterators; a changed file size or modification time invalidates
    the cached entry. NCLS is imported only when a file is requested.
    """
    if not path:
        return None
    path = Path(path).resolve()
    stat = path.stat()
    return _cached_excluded_regions(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=8)
def _cached_excluded_regions(path, mtime_ns, size):
    """Keep native indexes local to this process; stat fields form the cache key."""
    return ExcludedRegions(path)
