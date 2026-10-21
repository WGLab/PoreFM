"""Copy POD5 signals into ``SG`` tags in one unsorted BAM output.

Reads are emitted in POD5 traversal order rather than genomic-coordinate
order. The completed BAM is installed atomically so an interrupted conversion
does not leave a partial file at the requested output path.
"""

from __future__ import annotations

import array
import datetime
import os
from pathlib import Path
import sys
import tempfile
import time

import blosc2
import pod5 as p5
import pysam

from porefm.cli._parser import PoreFMArgumentParser
from porefm.data.bam_headers import add_pg_to_header


def get_files(input_list):
    """Resolve a POD5 path, directory, or iterable into recursive POD5 files.

    Directories are searched recursively. Non-POD5 paths and ``None`` are
    ignored; the returned order determines read order in the output BAM.
    """
    file_type = "pod5"
    out_list = []
    if input_list is None:
        return out_list

    if isinstance(input_list, (str, os.PathLike)):
        input_list = [input_list]

    for item in input_list:
        item = os.fspath(item)
        if os.path.isdir(item):
            out_list.extend(Path(item).rglob(f"*.{file_type}"))
        elif item.endswith(f".{file_type}"):
            out_list.append(item)
    return out_list


def add_signal_to_bam(
    input_bam_path,
    signal_input,
    output_path,
    *,
    threads=1,
    clevel=1,
    remove_mod_tags=False,
):
    """Attach compressed POD5 signals to matching primary BAM alignments.

    Parameters
    ----------
    input_bam_path : str or pathlib.Path
        Input BAM whose records are matched to POD5 reads by query name. The
        complete read-name index is held in memory; a BAM index file is not
        required.
    signal_input : str, pathlib.Path, or iterable of paths
        POD5 file, recursively searched directory, or collection of either.
        POD5 traversal order determines output record order.
    output_path : str or pathlib.Path
        Destination for the single output BAM. Its parent directory is created
        when necessary. A temporary BAM is written beside it and atomically
        moved into place after successful completion.
    threads : int, default=1
        Number of htslib threads used to compress the output BAM.
    clevel : int, default=1
        Blosc2 Zstandard compression level for raw signal stored in ``SG``;
        valid values are 1 through 9.
    remove_mod_tags : bool, default=False
        Remove existing ``MM`` and ``ML`` modification tags from records that
        receive signal. By default those tags are preserved unchanged.

    Notes
    -----
    Every matching primary alignment receives the POD5 signal as a compressed
    byte-array ``SG`` tag. Supplementary and secondary alignments, records
    without an ``mv`` move table, POD5 reads absent from the BAM, and POD5
    files with an invalid signature are skipped. Other decoding and writing
    errors propagate, and the incomplete temporary BAM is removed.

    The output is generally unsorted because records follow POD5 order. Sort
    and index it separately before coordinate-based access.
    """
    input_bam_path = Path(input_bam_path)
    output_path = Path(output_path)
    if input_bam_path.resolve() == output_path.resolve():
        raise ValueError("input and output BAM paths must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".tmp.bam",
        dir=output_path.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)

    read_count = 0
    written_count = 0
    try:
        with pysam.AlignmentFile(input_bam_path, "rb", check_sq=False) as input_bam:
            output_header = add_pg_to_header(input_bam.header)

            print(
                f"{datetime.datetime.now()}: Building in-memory BAM read-name index.",
                flush=True,
            )
            bam_index = pysam.IndexedReads(input_bam)
            bam_index.build()
            print(
                f"{datetime.datetime.now()}: Finished building BAM read-name index.",
                flush=True,
            )

            signal_files = get_files(signal_input)
            print(f"Number of POD5 files={len(signal_files)}", flush=True)

            with pysam.AlignmentFile(
                temporary_path,
                "wb",
                header=output_header,
                threads=threads,
            ) as bam_out:
                for pod5_path in signal_files:
                    try:
                        with p5.Reader(pod5_path) as reader:
                            for read in reader.reads():
                                read_count += 1
                                try:
                                    read_iter = bam_index.find(str(read.read_id))
                                    for bam_read in read_iter:
                                        if bam_read.is_supplementary or bam_read.is_secondary:
                                            continue

                                        # Signal-BAM consumers require the move
                                        # table to align raw samples to bases.
                                        bam_read.get_tag("mv")
                                        signal = read.signal
                                        compressed_data = blosc2.compress2(
                                            signal,
                                            typesize=signal.itemsize,
                                            codec=blosc2.Codec.ZSTD,
                                            nthreads=1,
                                            clevel=clevel,
                                            filters=[
                                                blosc2.Filter.DELTA,
                                                blosc2.Filter.SHUFFLE,
                                            ],
                                        )
                                        bam_read.set_tag(
                                            "SG",
                                            array.array("B", compressed_data),
                                        )
                                        if remove_mod_tags:
                                            # Some downstream workflows need a
                                            # signal-only BAM without existing
                                            # modification calls.
                                            bam_read.set_tag("MM", None)
                                            bam_read.set_tag("ML", None)
                                        bam_out.write(bam_read)
                                        written_count += 1
                                except KeyError:
                                    # Missing read names and missing required
                                    # BAM tags retain the legacy skip behavior.
                                    continue
                    except RuntimeError as error:
                        if "Invalid signature" not in str(error):
                            raise
                        print(
                            f"Skipping corrupt POD5 file with invalid signature: {pod5_path}",
                            flush=True,
                        )

        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    print(
        f"{datetime.datetime.now()}: Processed {read_count} POD5 reads; "
        f"wrote {written_count} primary BAM records to {output_path}.",
        flush=True,
    )


def build_parser():
    """Build the command-line parser for single-BAM signal attachment."""
    parser = PoreFMArgumentParser(
        description=(
            "Attach POD5 signal records to BAM reads and write one unsorted BAM."
        )
    )
    parser.add_argument(
        "--bam",
        required=True,
        help="Input BAM whose records will receive SG signal tags.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "POD5 file or directory. Directories are searched recursively for "
            "files ending in .pod5."
        ),
    )
    parser.add_argument(
        "--output",
        help="Output directory. The current working directory is used when omitted.",
    )
    parser.add_argument(
        "--prefix",
        default="output",
        help="Output filename prefix; the file is written as <prefix>.bam.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of htslib threads used for BAM compression.",
    )
    parser.add_argument(
        "--clevel",
        type=int,
        choices=range(1, 10),
        default=1,
        metavar="{1..9}",
        help="Blosc2 Zstandard compression level for the SG signal payload.",
    )
    parser.add_argument(
        "--remove_mod_tags",
        action="store_true",
        help=(
            "Remove existing MM and ML modification tags from records receiving "
            "signal. By default they are preserved."
        ),
    )
    return parser


def main(argv=None):
    """Parse CLI arguments and write the requested single signal BAM."""
    start_time = time.time()
    args = build_parser().parse_args(argv)

    output_dir = Path(args.output) if args.output else Path.cwd()
    output_path = output_dir / f"{args.prefix}.bam"

    print(f"{datetime.datetime.now()}: Starting signal attachment.", flush=True)
    print(sys.argv, flush=True)
    add_signal_to_bam(
        args.bam,
        args.input,
        output_path,
        threads=args.threads,
        clevel=args.clevel,
        remove_mod_tags=args.remove_mod_tags,
    )
    print(
        f"{datetime.datetime.now()}: Time elapsed={time.time() - start_time:.4f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
