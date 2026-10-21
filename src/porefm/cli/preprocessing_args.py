"""Shared lightweight CLI options and startup preprocessing resolution."""


def add_preprocessing_arguments(parser):
    """Add chemistry selection and make legacy options optional assertions.

    Call after existing options have been declared. Help construction does not
    import numpy, torch, or read packaged tables/checkpoints.
    """
    descriptions = {
        "chemistry": "Chemistry for standalone/scratch use: dna_r10.4.1_e8.2_400bps, rna004, rna002, or custom. Inferred from checkpoints.",
        "seq_type": "Optional sequence-type assertion; inferred from checkpoint or chemistry. Required for custom tables.",
        "pore_type": "Optional pore assertion (RNA002, RNA004, or R9 with RNA); inferred from checkpoint or chemistry.",
        "kmer_dict_path": "Optional custom table path; with a checkpoint/chemistry its levels must match exactly.",
    }
    actions = {action.dest: action for action in parser._actions}
    for name, description in descriptions.items():
        if name in actions:
            action = actions[name]
            action.required = False
            action.default = None
            action.help = description
        else:
            options = ["--" + name]
            if "_" in name:
                options.append("--" + name.replace("_", "-"))
            parser.add_argument(*options, dest=name, default=None, help=description)
    return parser


def resolve_cli_preprocessing(args, *, checkpoint_paths=None):
    """Resolve once before workflow side effects and write portable args metadata.

    The checkpoint fields used by PoreFM are checked together when present;
    resuming a derived model must agree with any separately supplied backbone.
    Subsequent calls reuse the serialized resolved contract, avoiding another
    checkpoint read. Legacy explicit options are assertions, never overrides.
    """
    from porefm.preprocessing import (
        assert_compatible_preprocessing, load_preprocessing, resolve_preprocessing,
    )
    from pathlib import Path

    if checkpoint_paths is None:
        checkpoint_paths = [getattr(args, key, None) for key in
                            ("model_load_path", "pretrain_load_path", "model_path", "null_model_path", "canonical_null_model_path")]
    paths = list(dict.fromkeys(str(Path(path).expanduser()) for path in checkpoint_paths if path))
    table_path = getattr(args, "kmer_dict_path", None)
    table_path = str(Path(table_path).expanduser()) if table_path else None
    existing = getattr(args, "preprocessing", None)
    if (existing is not None and getattr(args, "_preprocessing_resolved_paths", None) == paths
            and getattr(args, "_preprocessing_resolved_table_path", None) == table_path):
        return resolve_preprocessing(preprocessing=existing,
                                     chemistry=getattr(args, "chemistry", None),
                                     seq_type=getattr(args, "seq_type", None),
                                     pore_type=getattr(args, "pore_type", None))
    spec = resolve_preprocessing(preprocessing=existing) if existing is not None else None
    for path in paths:
        candidate = load_preprocessing(path)
        if spec is not None:
            assert_compatible_preprocessing(spec, candidate)
        spec = candidate
    spec = resolve_preprocessing(
        preprocessing=spec, chemistry=getattr(args, "chemistry", None),
        seq_type=getattr(args, "seq_type", None), pore_type=getattr(args, "pore_type", None),
        kmer_dict_path=table_path,
    )
    args.preprocessing = spec.to_dict()
    args._preprocessing_resolved_paths = paths
    args._preprocessing_resolved_table_path = table_path
    args.seq_type, args.pore_type = spec.seq_type, spec.pore_type
    return spec
