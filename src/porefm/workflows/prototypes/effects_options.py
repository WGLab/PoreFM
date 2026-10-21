"""Named fit-time recipes for effect-prototype ablations (no runtime imports)."""

import math

CANONICAL_MODES = ('shrunken', 'empirical', 'regression')
EFFECT_MODES = ('regression', 'empirical', 'mixed')
NOISE_MODES = ('base', 'hierarchical-5mer', 'none')
DEFAULTS = dict(
    canonical_mean_mode='shrunken', canonical_unseen='regression', canonical_strength=32.0, canonical_ridge=10.0,
    effect_mode='regression', effect_shrinkage=0.5, effect_mix_weight=0.5, effect_unseen='no-call', effect_ridge=10.0,
    footprint_ridge=10.0, footprint_shrinkage=0.5,
    noise_mode='base', noise_shrinkage=0.1, noise_floor=1e-6,
    noise_local_strength=512.0, noise_local_shrinkage=None,
)
CHOICES = dict(canonical_mean_mode=CANONICAL_MODES, canonical_unseen=('regression', 'no-call'),
               effect_mode=EFFECT_MODES, effect_unseen=('no-call', 'global'), noise_mode=NOISE_MODES)
EFFECT_OVERRIDES = ('effect_mode', 'effect_shrinkage', 'effect_mix_weight', 'effect_unseen', 'effect_ridge')


def resolve_options(values):
    """Return defaults plus explicit settings, rejecting nonfinite/range errors.

    Unknown manifest fields are checked by the fitter, not this helper. The
    returned dictionary is recorded in the artifact so no defaults are inferred
    when scoring. Local covariance pooling uses a fixed weight when supplied;
    otherwise it uses strength/(strength + within-9-mer degrees of freedom).
    """
    result = {key: values.get(key, default) for key, default in DEFAULTS.items()}
    for key, options in CHOICES.items():
        if result[key] not in options:
            raise ValueError(f'{key} must be one of {options}')
    for key, value in result.items():
        if key in CHOICES or (key == 'noise_local_shrinkage' and value is None):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite and numeric')
        if key in ('effect_shrinkage', 'effect_mix_weight', 'footprint_shrinkage', 'noise_shrinkage', 'noise_local_shrinkage'):
            if not 0 <= value <= 1:
                raise ValueError(f'{key} must be in [0, 1]')
        elif key == 'canonical_strength':
            if value < 0:
                raise ValueError('canonical_strength must be nonnegative')
        elif value <= 0:
            raise ValueError(f'{key} must be positive')
    return result


def add_fit_options(parser):
    """Expose manifest overrides on fit-effect-prototypes; omitted flags inherit."""
    for key, options in CHOICES.items():
        parser.add_argument('--' + key, choices=options, default=None,
                            help=f'Override manifest {key}; fitting default: {DEFAULTS[key]}.')
    for key, default in DEFAULTS.items():
        if key not in CHOICES:
            parser.add_argument('--' + key, type=float, default=None,
                                help=f'Override manifest {key}; fitting default: {default}.')
