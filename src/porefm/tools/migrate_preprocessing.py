"""Add or explicitly correct preprocessing metadata in a new checkpoint file."""

import argparse
from pathlib import Path

from porefm.cli._parser import PoreFMArgumentParser


def migrate_file(input_path, output_path, **selection):
    """Copy a trusted checkpoint with resolved metadata; never overwrite inputs.

    Explicit chemistry/custom-table selection deliberately replaces incomplete
    or contradictory legacy metadata. Without selection, resolve the existing
    checkpoint. Tensor state, architecture, and optimizer payloads are retained.
    """
    import torch
    from porefm.preprocessing import attach_preprocessing, resolve_preprocessing

    source, target = Path(input_path).resolve(), Path(output_path).resolve()
    if source == target:
        raise ValueError("Input and output checkpoint paths must differ.")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    explicit = any(value is not None for value in selection.values())
    spec = resolve_preprocessing(checkpoint=None if explicit else checkpoint, **selection)
    attach_preprocessing(checkpoint, spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        torch.save(checkpoint, handle)
    return spec


def main(argv=None):
    """Resolve/correct preprocessing while writing a distinct checkpoint."""
    parser = PoreFMArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chemistry", help="Explicit built-in chemistry or custom; replaces old preprocessing metadata.")
    parser.add_argument("--seq_type", "--seq-type", choices=("dna", "rna"))
    parser.add_argument("--pore_type", "--pore-type")
    parser.add_argument("--kmer_dict_path", "--kmer-dict-path", help="Table for an explicit custom profile.")
    args = parser.parse_args(argv)
    spec = migrate_file(args.input, args.output, chemistry=args.chemistry, seq_type=args.seq_type,
                        pore_type=args.pore_type, kmer_dict_path=args.kmer_dict_path)
    print(f"Wrote {args.output}: {spec.chemistry}, {spec.kmer_size}-mer, sha256={spec.table_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
