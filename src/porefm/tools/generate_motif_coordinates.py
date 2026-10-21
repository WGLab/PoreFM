#!/usr/bin/env python
"""Generate 0-based motif base coordinates from a reference FASTA.

This is a cleaned, parameterized replacement for the motif-coordinate section
of mod_labels_ipynb_convert.py. It preserves the coordinate and strand
selection used by bam_utils.get_labelled_arrays, but reports every motif/index
match for each selected position instead of collapsing overlaps to one label.
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pysam

from porefm.cli._parser import PoreFMArgumentParser

NT_DICT = {
    "R": "GA",
    "Y": "CT",
    "K": "GT",
    "M": "AC",
    "S": "GC",
    "W": "AT",
    "B": "GTC",
    "D": "GAT",
    "H": "ACT",
    "V": "GCA",
    "N": "AGCT",
}
VALID_MOTIF_ALPHABET = set(NT_DICT).union({"A", "C", "G", "T"})
REVCOMP_TRANS = str.maketrans("ATCG[]atcg", "TAGC][tagc")


def revcomp(seq: str) -> str:
    """Reverse-complement a DNA sequence or expanded regex character class."""
    return seq.translate(REVCOMP_TRANS)[::-1]


def parse_chroms(chroms: str) -> list[str]:
    """Parse comma/whitespace contig names and inclusive numeric ranges.

    A range such as ``1-3`` is expanded as ``chr1`` through ``chr3``; an
    explicit prefix such as ``chr3-1`` is preserved and descending ranges are
    supported. Other tokens are returned unchanged and in input order.
    """
    parsed: list[str] = []
    for token in re.split(r"[,\s]+", chroms):
        token = token.strip()
        if not token:
            continue
        range_match = re.fullmatch(r"(chr)?(\d+)-(\d+)", token)
        if range_match:
            prefix = range_match.group(1) or "chr"
            start = int(range_match.group(2))
            end = int(range_match.group(3))
            step = 1 if start <= end else -1
            parsed.extend(f"{prefix}{i}" for i in range(start, end + step, step))
        else:
            parsed.append(token)
    if not parsed:
        raise ValueError("No chromosomes were parsed from --chroms")
    return parsed


def chroms_from_fai(ref: Path) -> list[str]:
    """Read ordered contig names from the ``<reference>.fai`` index."""
    fai_path = Path(f"{ref}.fai")
    if not fai_path.exists():
        raise FileNotFoundError(f"FASTA index not found: {fai_path}")

    chroms = []
    with fai_path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if fields and fields[0]:
                chroms.append(fields[0])
    if not chroms:
        raise ValueError(f"No contigs found in FASTA index: {fai_path}")
    return chroms


def motif_check(motif: str) -> tuple[str, str, list[int]]:
    """Validate one motif specification and expand IUPAC ambiguity codes.

    A specification contains a motif followed by one or more zero-based motif
    offsets, for example ``"DRACH 2"``. All selected offsets must identify the
    same unambiguous A/C/G/T base.

    Returns
    -------
    tuple
        Uppercase motif, equivalent regular-expression pattern, and selected
        offsets.
    """
    motif_parts = motif.split()
    if len(motif_parts) < 2:
        raise ValueError("Motif must include a sequence and at least one index")

    motif_seq = motif_parts[0].upper()
    if set(motif_seq) - VALID_MOTIF_ALPHABET:
        valid_letters = ",".join(sorted(VALID_MOTIF_ALPHABET))
        raise ValueError(f"Motif sequence contains invalid letters. Valid letters: {valid_letters}")

    if not all(index.isnumeric() for index in motif_parts[1:]):
        raise ValueError("Motif indices must be non-negative integers after the motif sequence")

    motif_indices = [int(index) for index in motif_parts[1:]]
    if any(index >= len(motif_seq) for index in motif_indices):
        raise ValueError(f"Motif index is outside motif length for {motif_seq}")

    indexed_bases = {motif_seq[index] for index in motif_indices}
    if len(indexed_bases) != 1 or indexed_bases - set("ACGT"):
        raise ValueError("Base of interest must be the same A/C/G/T base for all motif indices")

    exp_motif_seq = motif_seq
    for nt, bases in NT_DICT.items():
        exp_motif_seq = exp_motif_seq.replace(nt, f"[{bases}]")

    return motif_seq, exp_motif_seq, motif_indices


def parse_motifs(motif_arg: str) -> list[dict[str, object]]:
    """Parse semicolon-delimited motif specifications with stable input order."""
    parsed = []
    for motif_i, motif in enumerate(x.strip() for x in motif_arg.split(";")):
        if not motif:
            continue
        motif_seq, exp_motif_seq, motif_indices = motif_check(motif)
        parsed.append(
            {
                "motif": motif,
                "seq": motif_seq,
                "exp_seq": exp_motif_seq,
                "indices": tuple(int(i) for i in motif_indices),
                "order": motif_i,
            }
        )
    if not parsed:
        raise ValueError("No valid motifs were parsed from --motif")
    return parsed


def motif_metadata_string(parsed_motifs: list[dict[str, object]]) -> str:
    """Reconstruct the semicolon-delimited motif string stored in metadata."""
    return ";".join(str(m["motif"]) for m in parsed_motifs)


def _dictionary_string_array(values: list[str], length: int) -> pa.DictionaryArray:
    """Construct a dictionary-encoded Arrow string array in first-seen order."""
    if length == 0:
        return pa.DictionaryArray.from_arrays(
            pa.array([], type=pa.int32()),
            pa.array([], type=pa.string()),
        )

    unique_to_index: dict[str, int] = {}
    dictionary_values: list[str] = []
    indices = np.empty(length, dtype=np.int32)
    for i, value in enumerate(values):
        dict_i = unique_to_index.get(value)
        if dict_i is None:
            dict_i = len(dictionary_values)
            unique_to_index[value] = dict_i
            dictionary_values.append(value)
        indices[i] = dict_i

    return pa.DictionaryArray.from_arrays(
        pa.array(indices, type=pa.int32()),
        pa.array(dictionary_values, type=pa.string()),
    )


def _constant_dictionary_array(value: str, length: int) -> pa.DictionaryArray:
    """Construct a dictionary array containing one repeated string value."""
    return pa.DictionaryArray.from_arrays(
        pa.array(np.zeros(length, dtype=np.int32), type=pa.int32()),
        pa.array([value], type=pa.string()),
    )


def _table_from_position_matches(
    chrom: str,
    strand: int,
    position_matches: dict[int, set[tuple[int, str]]],
) -> pa.Table:
    """Build one strand's coordinate table from matches grouped by position.

    Multiple motif/index hits at one coordinate are retained in a deterministic,
    semicolon-delimited ``motif_matches`` value.
    """
    sorted_positions = np.array(sorted(position_matches), dtype=np.int32)
    n_rows = len(sorted_positions)

    if n_rows:
        match_strings = [
            ";".join(match for _, match in sorted(position_matches[int(pos)]))
            for pos in sorted_positions
        ]
        motif_array = _dictionary_string_array(match_strings, n_rows)
    else:
        motif_array = _dictionary_string_array([], 0)

    return pa.table(
        {
            "chrom": _constant_dictionary_array(chrom, n_rows),
            "position": pa.array(sorted_positions, type=pa.int32()),
            "strand": pa.array(np.full(n_rows, strand, dtype=np.int8), type=pa.int8()),
            "motif_matches": motif_array,
        }
    )


def scan_chromosome(
    chrom: str,
    ref: str,
    parsed_motifs: list[dict[str, object]],
) -> tuple[str, int, pa.Table]:
    """Scan both strands of one contig for all overlapping motif occurrences.

    Coordinates are 0-based reference offsets. Strand is encoded as ``0`` for
    forward motif matches and ``1`` for reverse-complement matches. Each
    selected motif index is reflected within reverse matches so it still names
    the same biological base of interest.

    Returns
    -------
    tuple
        ``(chrom, row_count, table)`` with forward rows followed by reverse
        rows, each sorted by coordinate.
    """
    ref_fasta = pysam.FastaFile(ref)
    seq = ref_fasta.fetch(chrom).upper()
    ref_fasta.close()

    fwd_matches_by_pos: dict[int, set[tuple[int, str]]] = {}
    rev_matches_by_pos: dict[int, set[tuple[int, str]]] = {}

    for motif in parsed_motifs:
        motif_seq = str(motif["seq"])
        exp_seq = str(motif["exp_seq"])
        motif_order = int(motif["order"])
        motif_indices = np.array(motif["indices"], dtype=np.int32)

        # A zero-width lookahead retains overlapping motif occurrences.
        fwd_pattern = re.compile(rf"(?={exp_seq})")
        rev_pattern = re.compile(rf"(?={revcomp(exp_seq)})")

        for match in fwd_pattern.finditer(seq):
            anchor = match.start(0)
            for motif_index in motif_indices:
                pos = anchor + int(motif_index)
                label = f"{motif_seq}:{int(motif_index)}"
                fwd_matches_by_pos.setdefault(pos, set()).add((motif_order, label))

        rev_offsets = len(motif_seq) - 1 - motif_indices
        for match in rev_pattern.finditer(seq):
            anchor = match.start(0)
            for motif_index, offset in zip(motif_indices, rev_offsets):
                pos = anchor + int(offset)
                label = f"{motif_seq}:{int(motif_index)}"
                rev_matches_by_pos.setdefault(pos, set()).add((motif_order, label))

    chrom_table = pa.concat_tables(
        [
            _table_from_position_matches(chrom, 0, fwd_matches_by_pos),
            _table_from_position_matches(chrom, 1, rev_matches_by_pos),
        ]
    )
    return chrom, chrom_table.num_rows, chrom_table


def resolve_output_path(output: Path, reference_name: str, chroms: list[str], motifs: str) -> Path:
    """Return an explicit Parquet path or derive a descriptive path in a directory."""
    if output.suffix == ".parquet":
        return output
    safe_motif = re.sub(r"[^A-Za-z0-9_.-]+", "_", motifs).strip("_")
    chrom_label = f"{chroms[0]}_{chroms[-1]}" if len(chroms) > 1 else chroms[0]
    return output / f"{reference_name}.{safe_motif}.{chrom_label}.parquet"


def write_motif_coordinates(
    ref: Path,
    chroms: list[str],
    parsed_motifs: list[dict[str, object]],
    output_path: Path,
    workers: int,
    reference_name: str,
) -> None:
    """Scan contigs in parallel and write one ordered Parquet dataset.

    Worker results may complete out of order, but tables are buffered and
    written in ``chroms`` order. The output schema records 0-based positions,
    integer strand, and all motif/index labels at each coordinate. Reference,
    contig, and motif provenance are stored as Parquet schema metadata.

    The destination parent directory is created as needed. Exceptions from a
    worker propagate after the writer is safely closed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [
            pa.field("chrom", pa.dictionary(pa.int32(), pa.string())),
            pa.field("position", pa.int32()),
            pa.field("strand", pa.int8()),
            pa.field("motif_matches", pa.dictionary(pa.int32(), pa.string())),
        ]
    )
    metadata = {
        "motif_string": motif_metadata_string(parsed_motifs),
        "chroms": ",".join(chroms),
        "reference": reference_name,
        "ref": str(ref),
        "motif_match_format": "semicolon-delimited motif:index entries",
    }
    writer = pq.ParquetWriter(
        output_path,
        schema.with_metadata({k: v.encode("utf-8") for k, v in metadata.items()}),
        compression="zstd",
        use_dictionary=True,
    )

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(scan_chromosome, chrom, str(ref), parsed_motifs): chrom
                for chrom in chroms
            }
            pending_tables: dict[str, pa.Table] = {}
            next_to_write = 0
            total_rows = 0

            for future in as_completed(futures):
                chrom, n_rows, table = future.result()
                print(f"{chrom}: {n_rows:,} rows", flush=True)
                pending_tables[chrom] = table
                while next_to_write < len(chroms) and chroms[next_to_write] in pending_tables:
                    table_to_write = pending_tables.pop(chroms[next_to_write])
                    writer.write_table(table_to_write.cast(schema))
                    total_rows += table_to_write.num_rows
                    next_to_write += 1

        print(f"Wrote {total_rows:,} rows to {output_path}", flush=True)
    finally:
        writer.close()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for reference motif scanning."""
    parser = PoreFMArgumentParser(
        description=(
            "Scan a reference FASTA for motif base coordinates. Motifs use the "
            "same syntax as bam_utils.get_labelled_arrays, e.g. 'DRACH 2' or "
            "'CTCCA 0;CNGG 0'."
        )
    )
    parser.add_argument(
        "--motif",
        required=True,
        help=(
            "Semicolon-delimited motif specs supplied as one shell argument. "
            "Quote values containing spaces, for example --motif 'DRACH 2;CG 0'."
        ),
    )
    parser.add_argument(
        "--chroms",
        default=None,
        help=(
            "Comma- or whitespace-delimited contigs/ranges in one shell argument. "
            "Quote whitespace-delimited input, for example --chroms 'chr1 chr2:0-1000'. "
            "Defaults to all contigs in <ref>.fai."
        ),
    )
    parser.add_argument("--ref", type=Path, required=True, help="Reference FASTA path.")
    parser.add_argument("--reference-name", default="GRCh38", help="Reference name for metadata/output.")
    parser.add_argument("--output", type=Path, required=True, help="Output parquet file or directory.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of chromosome worker processes.",
    )
    return parser


def main() -> None:
    """Parse command-line options and generate the motif-coordinate Parquet."""
    parser = build_arg_parser()
    args = parser.parse_args()

    chroms = parse_chroms(args.chroms) if args.chroms else chroms_from_fai(args.ref)
    parsed_motifs = parse_motifs(args.motif)
    output_path = resolve_output_path(
        args.output,
        args.reference_name,
        chroms,
        motif_metadata_string(parsed_motifs),
    )

    write_motif_coordinates(
        ref=args.ref,
        chroms=chroms,
        parsed_motifs=parsed_motifs,
        output_path=output_path,
        workers=args.workers,
        reference_name=args.reference_name,
    )


if __name__ == "__main__":
    main()
