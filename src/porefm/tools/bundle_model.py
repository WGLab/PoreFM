"""Build a portable zero-shot or prototype inference bundle."""

from porefm.cli._parser import PoreFMArgumentParser
from porefm.common.model_bundle import SCORERS
from porefm.common.model_bundle_export import export_model_bundle


def parse_args(argv=None):
    """Validate the component set without importing the inference runtime."""
    parser = PoreFMArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=['zero-shot', 'prototype'])
    parser.add_argument('--output', required=True, help='New .pt bundle; never overwrites an existing file.')
    parser.add_argument('--canonical_null_model_path')
    parser.add_argument('--calibration_path')
    parser.add_argument('--whitening_artifact')
    parser.add_argument('--model_path', '--model-path', help='Pretrained checkpoint for prototype inference.')
    parser.add_argument('--effect_model')
    parser.add_argument('--prototype_embeddings', nargs='+')
    parser.add_argument('--class_labels', nargs='+')
    parser.add_argument('--scorer', choices=SCORERS, default=None)
    args = parser.parse_args(argv)
    if args.kind == 'zero-shot':
        if not args.canonical_null_model_path or not args.calibration_path:
            parser.error('zero-shot requires --canonical_null_model_path and --calibration_path')
        if any(getattr(args, name) is not None for name in
               ('model_path', 'effect_model', 'prototype_embeddings', 'class_labels', 'scorer')):
            parser.error('zero-shot does not accept prototype component/settings options')
        args.components = {name: getattr(args, name) for name in
                           ('canonical_null_model_path', 'calibration_path', 'whitening_artifact')
                           if getattr(args, name)}
        args.settings = {}
    else:
        if not args.model_path:
            parser.error('prototype requires --model_path')
        if any((args.canonical_null_model_path, args.calibration_path, args.whitening_artifact)):
            parser.error('prototype does not accept zero-shot components')
        args.scorer = args.scorer or ('residual-effects' if args.effect_model else 'per-kmer-lda')
        args.components = {'model_path': args.model_path}
        args.settings = {'scorer': args.scorer}
        if args.scorer == 'residual-effects':
            if not args.effect_model or args.prototype_embeddings or args.class_labels:
                parser.error('residual-effects requires --effect_model and no per-class prototypes')
            args.components['effect_model'] = args.effect_model
        else:
            if args.effect_model:
                parser.error('effect options require --scorer residual-effects')
            if not args.prototype_embeddings or not args.class_labels or len(args.prototype_embeddings) != len(args.class_labels):
                parser.error('prototype requires matching --prototype_embeddings and --class_labels')
            args.components.update({f'prototype_{i}': path for i, path in enumerate(args.prototype_embeddings)})
            args.settings['class_labels'] = args.class_labels
    return args


def main(argv=None):
    """Write a self-contained archive and report its destination."""
    args = parse_args(argv)
    export_model_bundle(args.output, kind=args.kind, components=args.components, settings=args.settings)
    print(f'Wrote {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
