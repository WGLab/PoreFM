"""Annotate aligned BAM reads with pairwise anomaly calls from Parquet files.

Parquet calls are expressed in 0-based reference coordinates. This module maps
them into original read orientation and encodes anomaly significance in SAM
MM/ML tags, with optional byte-scaled coverage in a companion tag.
"""

import argparse
import array
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pysam

from porefm.cli._parser import PoreFMArgumentParser
from porefm.data.bam import revcomp
from porefm.data.bam_headers import add_pg_to_header


@dataclass(frozen=True)
class PairwiseCall:
    """One reference-coordinate anomaly call associated with a read name.

    ``strand`` uses the repository convention ``0`` for forward and ``1`` for
    reverse alignments; ``None`` allows the call to match either strand.
    ``coverage`` is optional because older pairwise tables may omit it.
    """

    refname: str
    refpos: int
    strand: int | None
    pvalue: float
    coverage: float | None


def expand_paths(paths, suffix):
    """Expand files and recursively discovered directory entries by suffix."""
    out = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            out.extend(sorted(path.rglob(f"*{suffix}")))
        else:
            out.append(path)
    return [str(path) for path in out]


def _resolve_column(columns, requested, candidates, required=True):
    """Resolve an explicit or first-known schema column name."""
    if requested:
        if requested not in columns:
            raise ValueError(f"Requested column '{requested}' not found. Available columns: {sorted(columns)}")
        return requested

    for candidate in candidates:
        if candidate in columns:
            return candidate

    if required:
        raise ValueError(f"None of {candidates} found. Available columns: {sorted(columns)}")
    return None


def _coverage_columns(columns, coverage_column=None, wt_coverage_column=None, control_coverage_column=None):
    """Identify a single coverage column or a test/control coverage pair."""
    if coverage_column:
        return ("single", _resolve_column(columns, coverage_column, [], required=True), None)

    if wt_coverage_column or control_coverage_column:
        if not (wt_coverage_column and control_coverage_column):
            raise ValueError("--wt_coverage_column and --control_coverage_column must be provided together.")
        return (
            "pair",
            _resolve_column(columns, wt_coverage_column, [], required=True),
            _resolve_column(columns, control_coverage_column, [], required=True),
        )

    for wt_col, control_col in (
        ("wt_coverage", "control_coverage"),
        ("wt_cov", "control_cov"),
        ("test_coverage", "control_coverage"),
        ("test_cov", "control_cov"),
        ("wt_count", "control_count"),
        ("test_count", "control_count"),
    ):
        if wt_col in columns and control_col in columns:
            return ("pair", wt_col, control_col)

    for col in ("min_coverage", "coverage", "cov", "min_cov"):
        if col in columns:
            return ("single", col, None)

    return None


def _optional_strand(value):
    """Convert a numeric or string strand value to ``int``, preserving missing values."""
    if value is None:
        return None
    try:
        if not np.isfinite(value):
            return None
    except TypeError:
        # Preserve support for string-encoded integer strands such as "0" and "1".
        pass
    return int(value)


def _call_sources(parquet_paths):
    """Yield column names, projected batch readers, and physical-name aliases.

    A split pair is read once even when both filenames are supplied. Either
    filename resolves its companion through the shared output reader, which
    validates lengths and restores identities using original row offsets.
    Flat files retain their original names for custom column overrides.
    """
    from porefm.outputs.dataset import ALIASES, open_output

    seen_pairs = set()
    for raw_path in parquet_paths:
        path = Path(raw_path)
        if path.name.endswith(('_data.parquet', '_meta.parquet')):
            data_path = (path.with_name(path.name.removesuffix('_meta.parquet') + '_data.parquet')
                         if path.name.endswith('_meta.parquet') else path)
            key = data_path.resolve()
            if key in seen_pairs:
                continue
            dataset = open_output(data_path, format='pairwise')
            seen_pairs.add(key)
            yield dataset.schema.names, dataset.iter_tables, ALIASES
        else:
            parquet = pq.ParquetFile(path)
            yield parquet.schema_arrow.names, parquet.iter_batches, {}


def load_pairwise_calls(
    parquet_paths,
    readname_column=None,
    refname_column=None,
    position_column=None,
    pvalue_column=None,
    strand_column=None,
    coverage_column=None,
    wt_coverage_column=None,
    control_coverage_column=None,
    require_coverage=False,
):
    """Load valid pairwise calls and group them by read name.

    Accept flat files or split metadata/data parts; either split filename is
    sufficient, and supplying both never duplicates calls. Incomplete pairs or
    mismatched metadata lengths raise before that pair is consumed. Split rows
    are reconstructed in bounded batches without a temporary expanded Parquet.
    The returned per-read mapping still retains all kept calls in memory.

    Column-name arguments override automatic schema discovery, accepting stored
    or shared logical names for split files. Coverage may be
    supplied as one column or as a test/control pair, in which case the stored
    value is their minimum. Rows with invalid positions, negative/non-finite
    p-values, or unparseable coordinates are excluded; invalid coverage becomes
    missing without excluding an otherwise valid call.

    Returns
    -------
    tuple
        ``(calls_by_read, stats)`` where the first item maps read names to
        ``PairwiseCall`` lists and ``stats`` reports total/retained rows and
        whether any input schema exposed coverage.
    """
    calls_by_read = defaultdict(list)
    total_rows = 0
    kept_rows = 0
    saw_coverage = False

    for names, read_batches, aliases in _call_sources(parquet_paths):
        columns = set(names)

        read_col = _resolve_column(columns, aliases.get(readname_column, readname_column), ("readname", "read_name"))
        ref_col = _resolve_column(columns, aliases.get(refname_column, refname_column), ("chrom", "refname", "chromosome"))
        pos_col = _resolve_column(columns, aliases.get(position_column, position_column), ("position", "refpos", "ref_pos"))
        p_col = _resolve_column(columns, aliases.get(pvalue_column, pvalue_column), ("p_val", "pvalue", "p_value"))
        strand_col = _resolve_column(columns, aliases.get(strand_column, strand_column), ("strand",), required=False)
        coverage_spec = _coverage_columns(
            columns,
            coverage_column=aliases.get(coverage_column, coverage_column),
            wt_coverage_column=aliases.get(wt_coverage_column, wt_coverage_column),
            control_coverage_column=aliases.get(control_coverage_column, control_coverage_column),
        )
        if coverage_spec is not None:
            saw_coverage = True

        read_columns = [read_col, ref_col, pos_col, p_col]
        if strand_col is not None:
            read_columns.append(strand_col)
        if coverage_spec is not None:
            _, cov_a, cov_b = coverage_spec
            read_columns.append(cov_a)
            if cov_b is not None:
                read_columns.append(cov_b)

        for table in read_batches(columns=list(dict.fromkeys(read_columns)), batch_size=65536):
            readnames = table[read_col].to_numpy(zero_copy_only=False)
            refnames = table[ref_col].to_numpy(zero_copy_only=False)
            positions = table[pos_col].to_numpy(zero_copy_only=False)
            pvalues = table[p_col].to_numpy(zero_copy_only=False)
            strands = (
                table[strand_col].to_numpy(zero_copy_only=False)
                if strand_col is not None
                else np.full(len(readnames), None, dtype=object)
            )

            coverage = None
            if coverage_spec is not None:
                kind, cov_a, cov_b = coverage_spec
                cov_values_a = table[cov_a].to_numpy(zero_copy_only=False).astype(np.float64)
                if kind == "pair":
                    cov_values_b = table[cov_b].to_numpy(zero_copy_only=False).astype(np.float64)
                    coverage = np.minimum(cov_values_a, cov_values_b)
                else:
                    coverage = cov_values_a

            for i, readname in enumerate(readnames):
                total_rows += 1
                try:
                    pvalue = float(pvalues[i])
                    refpos = int(positions[i])
                except (TypeError, ValueError):
                    continue

                if not np.isfinite(pvalue) or pvalue < 0:
                    continue

                cov_value = None
                if coverage is not None and np.isfinite(coverage[i]) and coverage[i] >= 0:
                    cov_value = float(coverage[i])

                strand_value = _optional_strand(strands[i]) if strand_col is not None else None

                calls_by_read[str(readname)].append(
                    PairwiseCall(
                        refname=str(refnames[i]),
                        refpos=refpos,
                        strand=strand_value,
                        pvalue=pvalue,
                        coverage=cov_value,
                    )
                )
                kept_rows += 1

    if require_coverage and not saw_coverage:
        raise ValueError("No coverage columns were found in the parquet inputs.")

    return calls_by_read, {"total_rows": total_rows, "kept_rows": kept_rows, "has_coverage": saw_coverage}


def pvalues_to_ml_bytes(pvalues, scale=75.0, phred_max=None):
    """Transform p-values into saturated uint8 anomaly scores for an ML tag.

    With ``phred_max`` supplied, round ``255 * clip(-10 log10(p) /
    phred_max, 0, 1)``; canonical-null annotation uses a maximum of 40.
    Otherwise retain the pairwise ``scale * sqrt(-log10(p))`` encoding.
    Zero p-values saturate and values above one map to zero.
    """
    pvalues = np.asarray(pvalues, dtype=np.float64)
    clipped = np.clip(pvalues, np.finfo(np.float64).tiny, 1.0)
    if phred_max is not None:
        if not np.isfinite(phred_max) or phred_max <= 0:
            raise ValueError("phred_max must be finite and greater than zero.")
        scores = 255.0 * np.clip(-10.0 * np.log10(clipped) / phred_max, 0, 1)
        return np.rint(scores).astype(np.uint8)
    scores = scale * np.sqrt(-np.log10(clipped))
    return np.minimum(scores.astype(np.int64), 255).astype(np.uint8)


def coverage_to_bytes(coverage):
    """Clip non-negative coverage values to the uint8 SAM-tag range."""
    coverage = np.asarray(coverage, dtype=np.float64)
    return np.minimum(np.clip(coverage, 0, None).astype(np.int64), 255).astype(np.uint8)


def reference_to_read_position_map(bam_read):
    """Map aligned reference positions to original-orientation read offsets.

    Insertions and deletions, for which either coordinate is absent, are
    omitted. Reverse-aligned query offsets are reflected around ``query_length``
    so the result indexes the restored 5'-to-3' sequence used for MM tags.
    """
    read_length = bam_read.query_length or 0
    mapping = {}
    for query_pos, ref_pos in bam_read.get_aligned_pairs(matches_only=False):
        if query_pos is None or ref_pos is None:
            continue
        read_pos = read_length - int(query_pos) - 1 if bam_read.is_reverse else int(query_pos)
        mapping.setdefault(int(ref_pos), read_pos)
    return mapping


def _base_match_settings(seq, base, seq_type):
    """Resolve the sequence character and SAM fundamental base for T/U calls."""
    base = base.upper()
    seq_type = seq_type.lower()

    if base not in {"A", "C", "G", "T", "U", "N"}:
        raise ValueError("base must be one of A,C,G,T,U,N")
    if seq_type not in {"dna", "rna"}:
        raise ValueError("seq_type must be 'dna' or 'rna'")

    if base == "U":
        return ("U" if "U" in seq else "T"), "U"
    if base == "T" and seq_type == "rna":
        return "T", "U"
    return base, base


def build_pairwise_mm_ml_tags(seq, base, read_positions, pvalues, coverage, seq_type, mod_code, skipped_flag="?", phred_max=None):
    """Encode read-coordinate pairwise calls as MM/ML and optional coverage.

    Candidate sites must be in range and match ``base`` in the original read
    sequence. Duplicate read coordinates retain the call with the greatest
    transformed ML score. RNA T calls are emitted under fundamental base U,
    and U calls may match T when the stored sequence contains no U.
    ``phred_max`` selects linear Phred-to-byte scaling when supplied; otherwise
    the existing square-root scaling is used.

    Returns
    -------
    tuple
        ``(mm, ml, coverage_bytes)`` in increasing read-coordinate order.
        ``mm`` and coverage are ``None`` when no valid base-matching call
        remains; ``ml`` is then empty.
    """
    seq = seq.upper()
    read_positions = np.asarray(read_positions, dtype=np.int64)
    pvalues = np.asarray(pvalues, dtype=np.float64)
    coverage = None if coverage is None else np.asarray(coverage, dtype=np.float64)

    valid = (
        (read_positions >= 0)
        & (read_positions < len(seq))
        & np.isfinite(pvalues)
        & (pvalues >= 0)
    )
    if coverage is not None:
        valid = valid & np.isfinite(coverage) & (coverage >= 0)

    read_positions = read_positions[valid]
    pvalues = pvalues[valid]
    if coverage is not None:
        coverage = coverage[valid]

    if len(read_positions) == 0:
        return None, [], None

    order = np.argsort(read_positions, kind="stable")
    read_positions = read_positions[order]
    pvalues = pvalues[order]
    if coverage is not None:
        coverage = coverage[order]

    seq_match_base, mm_base = _base_match_settings(seq, base, seq_type)
    if seq_match_base == "N":
        base_pos = np.arange(len(seq), dtype=np.int64)
        ranks = read_positions.copy()
        base_valid = np.ones(len(read_positions), dtype=bool)
    else:
        seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        base_pos = np.flatnonzero(seq_bytes == ord(seq_match_base))
        if len(base_pos) == 0:
            return None, [], None
        ranks_all = np.searchsorted(base_pos, read_positions)
        in_bounds = ranks_all < len(base_pos)
        base_valid = np.zeros(len(read_positions), dtype=bool)
        if np.any(in_bounds):
            base_valid[in_bounds] = base_pos[ranks_all[in_bounds]] == read_positions[in_bounds]
        ranks = ranks_all

    read_positions = read_positions[base_valid]
    pvalues = pvalues[base_valid]
    ranks = ranks[base_valid]
    if coverage is not None:
        coverage = coverage[base_valid]

    if len(read_positions) == 0:
        return None, [], None

    ml = pvalues_to_ml_bytes(pvalues, phred_max=phred_max)
    cov_bytes = coverage_to_bytes(coverage) if coverage is not None else None

    dedup = {}
    for idx, read_pos in enumerate(read_positions):
        read_pos = int(read_pos)
        if read_pos not in dedup or ml[idx] > dedup[read_pos][2]:
            dedup[read_pos] = (
                int(ranks[idx]),
                float(pvalues[idx]),
                int(ml[idx]),
                None if cov_bytes is None else int(cov_bytes[idx]),
            )

    sorted_items = sorted(dedup.items(), key=lambda item: item[0])
    ranks = np.asarray([item[1][0] for item in sorted_items], dtype=np.int64)
    ml = [item[1][2] for item in sorted_items]
    cov_out = None if cov_bytes is None else [item[1][3] for item in sorted_items]

    deltas = np.diff(np.concatenate(([-1], ranks))) - 1
    flag = "" if skipped_flag in (None, "") else skipped_flag
    mm = f"{mm_base}+{mod_code}{flag}," + ",".join(map(str, deltas.tolist())) + ";"
    return mm, ml, cov_out


def _clear_tags(bam_read, tags):
    """Remove the named optional fields from a mutable ``AlignedSegment``."""
    for tag in tags:
        if bam_read.has_tag(tag):
            bam_read.set_tag(tag, None)


def annotate_bam_from_pairwise_parquet(
    input_bams,
    parquet_paths,
    output_bam,
    base,
    mod_code,
    seq_type,
    threads=4,
    coverage_tag="cv",
    preserve_existing_mod_tags=False,
    canonical_null=False,
    drop_tags=(),
    chrom_list=None,
    **load_kwargs,
):
    """Stream input BAMs into one BAM annotated from pairwise Parquet calls.

    Directories in ``input_bams`` and ``parquet_paths`` are searched
    recursively. Calls must match read name, reference name, optional strand,
    and an aligned reference coordinate. By default existing MM/ML/MN and
    coverage tags are cleared from every output read. With
    ``preserve_existing_mod_tags=True``, they survive on reads with no new
    calls, but are replaced when new calls are written.

    Parameters
    ----------
    input_bams, parquet_paths : sequence of path-like
        BAM or Parquet files and/or directories to expand. Parquet inputs may
        be flat or paired metadata/data files; listing both members of a pair
        consumes its calls only once.
    output_bam : path-like
        Destination BAM, overwritten by ``pysam`` if it already exists.
    base, mod_code, seq_type : str
        SAM fundamental base, modification code, and ``"dna"``/``"rna"``
        sequence convention passed to tag construction.
    threads : int
        Threads used by the BAM writer.
    coverage_tag : str
        Optional-field name used for byte-scaled coverage. Callers are expected
        to provide a valid two-character SAM tag.
    preserve_existing_mod_tags : bool
        Preserve existing tags only on reads that receive no new annotation.
    canonical_null : bool
        Encode calibrated per-read p-values with Phred-40 scaling. Do not
        write coverage or interpret calibration sample counts as coverage.
        Remove existing ``CV``/``cv`` tags on every output record, including
        when ``preserve_existing_mod_tags`` is enabled. Coverage-column
        overrides and requirements are incompatible with this mode.
    drop_tags : sequence of str
        Additional auxiliary tags removed from all output records, including
        records without scores. Defaults to preserving other input tags.
    chrom_list : sequence of str or None
        Restrict output to mapped records on these references. Restricted
        inputs require BAM indexes and are fetched in header order to retain
        coordinate sorting. Omit to copy every source record.
    **load_kwargs
        Column overrides and coverage requirements forwarded to
        ``load_pairwise_calls``.

    Returns
    -------
    dict
        Input row counts, file/read/site counts, unaligned-site count, and the
        output path.
    """
    if canonical_null and any(load_kwargs.get(key) for key in (
        "require_coverage", "coverage_column", "wt_coverage_column", "control_coverage_column"
    )):
        raise ValueError("Canonical-null annotation does not use coverage columns.")

    input_bams = expand_paths(input_bams, ".bam")
    parquet_paths = expand_paths(parquet_paths, ".parquet")
    if not input_bams:
        raise ValueError("No input BAM files found.")
    if not parquet_paths:
        raise ValueError("No parquet files found.")

    calls_by_read, load_stats = load_pairwise_calls(parquet_paths, **load_kwargs)

    output_parent = os.path.dirname(output_bam)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    tags_to_clear = ["MM", "ML", "MN"]
    if not canonical_null:
        tags_to_clear.append(coverage_tag)
    total_reads = 0
    tagged_reads = 0
    tagged_sites = 0
    skipped_sites = 0

    with pysam.AlignmentFile(input_bams[0], "rb", check_sq=False) as first_bam:
        out_header = add_pg_to_header(first_bam.header)

    with pysam.AlignmentFile(output_bam, "wb", header=out_header, threads=threads) as out_bam:
        for bam_path in input_bams:
            with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as in_bam:
                if chrom_list:
                    selected = set(chrom_list)
                    reads = (
                        read for chrom in in_bam.references if chrom in selected
                        for read in in_bam.fetch(chrom) if not read.is_unmapped
                    )
                else:
                    reads = in_bam.fetch(until_eof=True)
                for bam_read in reads:
                    total_reads += 1
                    _clear_tags(bam_read, drop_tags)
                    if canonical_null:
                        # Coverage from an earlier pairwise run must not appear
                        # to describe independent canonical-null scores.
                        _clear_tags(bam_read, ("CV", "cv"))
                    if not preserve_existing_mod_tags:
                        _clear_tags(bam_read, tags_to_clear)

                    read_calls = calls_by_read.get(bam_read.query_name)
                    if not read_calls or bam_read.is_unmapped:
                        out_bam.write(bam_read)
                        continue

                    read_strand = 0 if bam_read.is_forward else 1
                    read_refname = bam_read.reference_name
                    matching_calls = [
                        call
                        for call in read_calls
                        if call.refname == read_refname
                        and (call.strand is None or call.strand == read_strand)
                    ]
                    if not matching_calls:
                        out_bam.write(bam_read)
                        continue

                    ref_to_read = reference_to_read_position_map(bam_read)
                    read_positions, pvalues, coverages = [], [], []
                    has_coverage = False
                    for call in matching_calls:
                        read_pos = ref_to_read.get(call.refpos)
                        if read_pos is None:
                            skipped_sites += 1
                            continue
                        read_positions.append(read_pos)
                        pvalues.append(call.pvalue)
                        coverages.append(call.coverage if call.coverage is not None else np.nan)
                        has_coverage = has_coverage or call.coverage is not None

                    if not read_positions:
                        out_bam.write(bam_read)
                        continue

                    seq = bam_read.query_sequence or ""
                    if bam_read.is_reverse:
                        seq = revcomp(seq)

                    mm, ml, cov = build_pairwise_mm_ml_tags(
                        seq,
                        base,
                        read_positions,
                        pvalues,
                        coverages if has_coverage and not canonical_null else None,
                        seq_type,
                        mod_code,
                        phred_max=40.0 if canonical_null else None,
                    )

                    if ml:
                        if preserve_existing_mod_tags:
                            _clear_tags(bam_read, tags_to_clear)
                        bam_read.set_tag("MM", mm, value_type="Z")
                        bam_read.set_tag("ML", array.array("B", ml))
                        bam_read.set_tag("MN", len(seq))
                        if cov is not None:
                            bam_read.set_tag(coverage_tag, array.array("B", cov))
                        tagged_reads += 1
                        tagged_sites += len(ml)

                    out_bam.write(bam_read)

    return {
        **load_stats,
        "input_bams": len(input_bams),
        "parquet_files": len(parquet_paths),
        "total_reads": total_reads,
        "tagged_reads": tagged_reads,
        "tagged_sites": tagged_sites,
        "skipped_unaligned_sites": skipped_sites,
        "output_bam": output_bam,
    }


def build_parser():
    """Build the command-line parser for pairwise BAM annotation."""
    parser = PoreFMArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Copy pairwise p-value parquet calls into a BAM as MM/ML tags.",
    )
    parser.add_argument("--bam", nargs="+", required=True, help="Input WT BAM file(s) or directories.")
    parser.add_argument("--parquet", nargs="+", required=True, help="Flat or split meta/data Parquet file(s) or directories; paired files are reconstructed once.")
    parser.add_argument("--output_bam", required=True, help="Annotated BAM output path.")
    parser.add_argument("--base", required=True, help="Canonical base for the MM tag.")
    parser.add_argument("--mod_code", required=True, help="Modification code for the MM tag.")
    parser.add_argument("--seq_type", choices=["dna", "rna"], required=True)
    parser.add_argument("--threads", type=int, default=4, help="Threads for BAM writing.")
    parser.add_argument("--canonical_null", action="store_true", help="Use Phred-40 for canonical-null p-values; omit coverage and remove existing CV/cv tags.")
    parser.add_argument("--drop_tags", nargs="+", default=(), help="Additional input tags to remove, for example SG mv.")
    parser.add_argument("--chrom_list", nargs="+", default=None, help="Only output mapped records on these chromosomes; requires indexed BAM inputs.")
    parser.add_argument("--coverage_tag", default="cv", help="Two-character byte-array tag for coverage values.")
    parser.add_argument("--preserve_existing_mod_tags", action="store_true", help="Keep existing MM/ML/MN tags on reads without new calls.")
    parser.add_argument("--require_coverage", action="store_true", help="Fail if parquet inputs do not contain coverage columns.")

    parser.add_argument("--readname_column", default=None)
    parser.add_argument("--refname_column", default=None)
    parser.add_argument("--position_column", default=None)
    parser.add_argument("--pvalue_column", default=None)
    parser.add_argument("--strand_column", default=None)
    parser.add_argument("--coverage_column", default=None, help="Single coverage column to copy into the coverage tag.")
    parser.add_argument("--wt_coverage_column", default=None, help="WT/test coverage column; used with --control_coverage_column.")
    parser.add_argument("--control_coverage_column", default=None, help="Control coverage column; used with --wt_coverage_column.")

    return parser


def main():
    """Parse command-line options, annotate the BAM, and print summary counts."""
    args = build_parser().parse_args()

    if len(args.coverage_tag) != 2:
        raise ValueError("--coverage_tag must be a two-character SAM tag.")

    stats = annotate_bam_from_pairwise_parquet(
        input_bams=args.bam,
        parquet_paths=args.parquet,
        output_bam=args.output_bam,
        base=args.base,
        mod_code=args.mod_code,
        seq_type=args.seq_type,
        threads=args.threads,
        coverage_tag=args.coverage_tag,
        preserve_existing_mod_tags=args.preserve_existing_mod_tags,
        canonical_null=args.canonical_null,
        drop_tags=args.drop_tags,
        chrom_list=args.chrom_list,
        readname_column=args.readname_column,
        refname_column=args.refname_column,
        position_column=args.position_column,
        pvalue_column=args.pvalue_column,
        strand_column=args.strand_column,
        coverage_column=args.coverage_column,
        wt_coverage_column=args.wt_coverage_column,
        control_coverage_column=args.control_coverage_column,
        require_coverage=args.require_coverage,
    )
    for key, value in stats.items():
        print(f"{key}: {value}", flush=True)


if __name__ == "__main__":
    main()
