"""Shared configuration and VCF masking helpers for reference-guided read polishing."""

from collections import defaultdict
from pathlib import Path

import pysam


DEFAULT_POLISH_INDEL_LIMIT = 5


def validate_polishing_options(polish, polish_indel_limit, polish_vcf_path):
    """Validate and normalize reference-polishing configuration values.

    Parameters
    ----------
    polish : bool
        Whether reference-guided read polishing is enabled.
    polish_indel_limit : int
        Largest insertion or deletion eligible for correction. Boolean values
        are rejected even though ``bool`` is an ``int`` subclass.
    polish_vcf_path : str or os.PathLike or None
        Optional VCF/BCF of biological variants that polishing must preserve.
        Blank strings are normalized to ``None`` and a path requires
        ``polish=True``.

    Returns
    -------
    tuple
        Normalized ``(polish, polish_indel_limit, polish_vcf_path)``.

    Raises
    ------
    TypeError
        If the enable flag or indel limit has the wrong type.
    ValueError
        If the limit is negative or a variant path is supplied while
        polishing is disabled.
    """
    if not isinstance(polish, bool):
        raise TypeError(f"polish must be true or false, got {polish!r}")

    if isinstance(polish_indel_limit, bool) or not isinstance(polish_indel_limit, int):
        raise TypeError(
            "polish_indel_limit must be a non-negative integer, "
            f"got {polish_indel_limit!r}"
        )
    if polish_indel_limit < 0:
        raise ValueError(
            "polish_indel_limit must be a non-negative integer, "
            f"got {polish_indel_limit}"
        )

    if polish_vcf_path is not None:
        polish_vcf_path = str(polish_vcf_path).strip()
        if not polish_vcf_path:
            polish_vcf_path = None

    if polish_vcf_path and not polish:
        raise ValueError("polish_vcf_path requires polish=true")

    return bool(polish), int(polish_indel_limit), polish_vcf_path


class VcfVariantMask:
    """Preserve VCF-listed SNP and indel alleles during read polishing.

    Contigs are loaded lazily and cached per DataLoader worker. Indexed VCF/BCF
    files use contig fetches; unindexed files fall back to a sequential scan.
    All coordinates exposed to the polishing code are zero-based.
    """

    def __init__(self, vcf_path):
        """Create a lazy mask backed by an existing VCF or BCF file."""
        self.vcf_path = str(vcf_path)
        if not Path(self.vcf_path).is_file():
            raise FileNotFoundError(f"Polishing VCF does not exist: {self.vcf_path}")
        self._loaded_contigs = set()
        self._snp_alts = defaultdict(dict)
        self._indel_positions = defaultdict(set)

    @staticmethod
    def _is_canonical_base(base):
        return isinstance(base, str) and len(base) == 1 and base.upper() in "ACGT"

    def _add_record(self, record):
        chrom = record.contig
        start = int(record.start)
        ref = (record.ref or "").upper()

        for alt_value in record.alts or ():
            alt = (alt_value or "").upper()
            if self._is_canonical_base(ref) and self._is_canonical_base(alt):
                self._snp_alts[chrom].setdefault(start, set()).add(alt)
                continue

            if not alt or alt == ".":
                continue

            # Include both the VCF anchor and the affected bases immediately
            # after it. This covers CIGAR insertions/deletions whose r_idx is
            # reported either at the anchor or at the first affected base.
            stop = max(int(record.stop or (start + len(ref))), start + 1)
            affected_stop = max(stop, start + 2)
            self._indel_positions[chrom].update(range(start, affected_stop))

    def _load_contig(self, chrom):
        if chrom in self._loaded_contigs:
            return

        with pysam.VariantFile(self.vcf_path) as variant_file:
            try:
                records = variant_file.fetch(chrom)
                for record in records:
                    self._add_record(record)
            except (KeyError, OSError, ValueError):
                # Plain or otherwise unindexed VCF: scan sequentially. Reopening
                # for a later contig is slower but keeps memory bounded by the
                # contigs actually encountered by this worker.
                for record in variant_file:
                    if record.contig == chrom:
                        self._add_record(record)

        self._loaded_contigs.add(chrom)

    def is_valid_snp(self, chrom, position, read_base):
        """Return whether ``read_base`` is a listed ALT SNP at a zero-based locus."""
        self._load_contig(chrom)
        return str(read_base).upper() in self._snp_alts[chrom].get(int(position), ())

    def has_indel(self, chrom, position):
        """Return whether an indel overlaps a zero-based CIGAR anchor position."""
        self._load_contig(chrom)
        position = int(position)
        positions = self._indel_positions[chrom]
        return position in positions or (position - 1) in positions


def get_polishing_variant_mask(sample, cache):
    """Return a worker-local cached VCF mask for a polishing-enabled sample.

    ``sample`` is expected to expose ``polish_vcf_path``. The mutable ``cache``
    is keyed by path so samples sharing a VCF also share its lazy contig cache.
    ``None`` is returned when the sample has no variant file.
    """
    vcf_path = getattr(sample, "polish_vcf_path", None)
    if not vcf_path:
        return None
    if vcf_path not in cache:
        cache[vcf_path] = VcfVariantMask(vcf_path)
    return cache[vcf_path]
