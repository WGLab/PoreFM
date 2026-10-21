"""Context-transferred effect prototypes and finite-range neighborhood inference.

All vectors are in the unchanged backbone's raw embedding coordinates. Canonical
noise transforms are applied to both observations and candidate effects at the
*receiving* base. Neighbor inference is an approximate discrete Gaussian model,
not calibrated modification probabilities or a conformal guarantee.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict, deque

import torch

from .effects_options import CANONICAL_MODES, EFFECT_MODES, NOISE_MODES
from .effects_noise import receiver_whitening

FORMAT_VERSION = 2
ARTIFACT_TYPE = 'context_conditioned_effect_prototypes'
BASES = 'ACGT'
DEFAULT_EFFECT_CACHE_MB = 256


def checkpoint_digest(path):
    """SHA256 of checkpoint bytes; independent of spelling/symlink of its path."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_k(k):
    """Require a centered, odd k-mer representable safely with two-bit IDs."""
    if not isinstance(k, int) or k < 1 or k > 15 or k % 2 != 1:
        raise ValueError('k-mer size must be an odd integer in [1, 15]')


def context_ids(sequence, k):
    """Return length-L two-bit IDs; incomplete/noncanonical contexts are -1.

    Sequence and all offsets follow read orientation (5-prime to 3-prime).
    RNA U is encoded as T, consistently with the existing prototype pipeline.
    """
    validate_k(k)
    seq = sequence.upper().replace('U', 'T')
    tokens = torch.tensor([BASES.find(c) for c in seq], dtype=torch.long)
    result = torch.full((len(seq),), -1, dtype=torch.long)
    if len(seq) >= k:
        windows = tokens.unfold(0, k, 1)
        weights = 4 ** torch.arange(k - 1, -1, -1)
        ids = (windows.clamp_min(0) * weights).sum(1)
        ids[(windows < 0).any(1)] = -1
        result[k // 2:len(seq) - k // 2] = ids
    return result


def sequence_features(ids, k):
    """Intercept and noncentral one-hot bases, shape (N, 1+4*(k-1)).

    Center base is implicit in the model family. Invalid IDs must be masked by
    the caller and are rejected here rather than silently interpreted as AAA.
    """
    validate_k(k)
    ids = torch.as_tensor(ids, dtype=torch.long, device='cpu').reshape(-1)
    if ((ids < 0) | (ids >= 4 ** k)).any():
        raise ValueError('sequence_features requires complete canonical k-mer IDs')
    offsets = torch.arange(k - 1, -1, -1)
    tokens = (ids[:, None] // (4 ** offsets)) % 4
    flank = tokens[:, torch.arange(k) != k // 2]
    onehot = torch.nn.functional.one_hot(flank, 4).reshape(len(ids), 4 * (k - 1)).double()
    return torch.cat([torch.ones(len(ids), 1, dtype=torch.double), onehot], 1)


def fit_effect_regression(ids, values, k, ridge=10.0, shrinkage=0.5):
    """Fit equal-context ridge effects, shrunk toward their shared mean.

    ``values`` is (contexts, embedding_dim), typically paired modified-control
    mean differences. Returns coefficients (1+4*(k-1), embedding_dim). No exact
    modified-target lookup is used at inference, including for unseen k-mers.
    """
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError('effect ridge must be finite and positive')
    if not math.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError('effect shrinkage must be in [0, 1]')
    x = sequence_features(ids, k)
    y = torch.as_tensor(values, dtype=torch.double, device='cpu')
    if y.ndim != 2 or len(y) != len(x) or len(y) == 0 or not torch.isfinite(y).all():
        raise ValueError('effect values must be nonempty finite context-by-feature vectors')
    penalty = torch.eye(x.shape[1], dtype=torch.double) * ridge
    penalty[0, 0] = 0
    coef = torch.linalg.solve(x.T @ x + penalty, x.T @ y)
    coef *= 1 - shrinkage
    coef[0] += shrinkage * y.mean(0)
    return coef.float()


def canonical_noise_metric(reservoir_groups, dim, shrinkage=0.1, floor=1e-6):
    """Fit pooled within-canonical-context covariance and row whitening.

    ``reservoir_groups`` yields (n_observations, dim) tensors, one per context.
    Each group is centered on its own sample mean. Covariance uses sum(n-1)
    degrees of freedom and isotropic shrinkage plus an absolute diagonal floor.
    No modified observations enter this estimator.
    """
    if not math.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError('noise shrinkage must be in [0, 1]')
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError('noise floor must be finite and positive')
    cross = torch.zeros(dim, dim, dtype=torch.double)
    degrees, observations, groups = 0, 0, 0
    for values in reservoir_groups:
        x = values.to(device='cpu', dtype=torch.double)
        if x.ndim != 2 or x.shape[1] != dim or not torch.isfinite(x).all():
            raise ValueError('canonical noise observations have invalid shape or nonfinite values')
        if len(x) < 2:
            continue
        x = x - x.mean(0)
        cross += x.T @ x
        degrees += len(x) - 1
        observations += len(x)
        groups += 1
    if not degrees:
        raise ValueError('canonical noise needs at least two observations in a context')
    empirical = cross / degrees
    covariance = ((1 - shrinkage) * empirical
                  + (shrinkage * empirical.diag().mean() + floor) * torch.eye(dim, dtype=torch.double))
    chol = torch.linalg.cholesky(covariance)
    whitening = torch.linalg.solve_triangular(chol, torch.eye(dim, dtype=torch.double), upper=False).T
    return dict(covariance=covariance.float(), whitening=whitening.float(),
                observations=observations, contexts=groups, degrees_of_freedom=degrees)


def canonical_expectation(ids, family):
    """Evaluate the selected canonical mean recipe, returning (N,D) raw vectors.

    ``empirical`` uses exact means; ``regression`` uses only the sequence
    regression; ``shrunken`` combines them using n/(n+canonical_strength).
    For empirical/shrunken modes, unseen complete contexts use the explicit
    ``canonical_unseen`` policy: regression or NaN (a downstream no-call).
    Regression mode always uses its fitted predictor. Incomplete IDs must be
    masked before calling. No parameters are estimated from query observations.
    """
    ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
    mode = family['canonical_mean_mode']
    coef = family['canonical_coefficients']
    prediction = (sequence_features(ids, family['context_k']).float() @ coef if coef is not None
                  else torch.full((len(ids), family['canonical_means'].shape[1]), float('nan')))
    if mode == 'regression':
        return prediction
    keys = family['canonical_ids']
    ix = torch.searchsorted(keys, ids)
    safe = ix.clamp_max(len(keys) - 1)
    known = (ix < len(keys)) & (keys[safe] == ids)
    if known.any():
        if mode == 'empirical':
            prediction[known] = family['canonical_means'][safe[known]]
        else:
            count = family['canonical_counts'][safe[known]].float()
            weight = count / (count + family['canonical_strength'])
            prediction[known] = (weight[:, None] * family['canonical_means'][safe[known]]
                                 + (1 - weight[:, None]) * prediction[known])
    if family['canonical_unseen'] == 'no-call':
        prediction[~known] = float('nan')
    return prediction


def fit_center_effect(ids, differences, counts, k, *, mode='regression', ridge=10.,
                      shrinkage=.5, mix_weight=.5, unseen='no-call'):
    """Fit a named center-effect recipe and retain its empirical support.

    ``ids`` are unique paired k-mers, differences is (N,D), and counts is (N,2)
    for modified/control observations. Counts are provenance, not a claim of
    independent broader-context replication. Regression coefficients already
    include global shrinkage, preserving the original scorer's arithmetic.
    Mixed mode uses a fixed empirical weight on supported contexts and zero
    empirical weight on unseen contexts. Empirical mode does not fit regression.
    """
    if mode not in EFFECT_MODES or unseen not in ('no-call', 'global'):
        raise ValueError('invalid center effect mode or unseen policy')
    if not math.isfinite(shrinkage) or not 0 <= shrinkage <= 1 or not math.isfinite(mix_weight) or not 0 <= mix_weight <= 1:
        raise ValueError('effect shrinkage and mix weight must be in [0, 1]')
    ids = torch.as_tensor(ids, dtype=torch.long)
    values = torch.as_tensor(differences, dtype=torch.float32)
    counts = torch.as_tensor(counts, dtype=torch.long)
    if (ids.ndim != 1 or not len(ids) or values.ndim != 2 or len(values) != len(ids)
            or counts.shape != (len(ids), 2) or (counts <= 0).any() or not torch.isfinite(values).all()
            or len(ids.unique()) != len(ids)):
        raise ValueError('invalid paired empirical effect arrays')
    sequence_features(ids, k)  # Validate the full context range even without regression.
    order = ids.argsort()
    coef = None if mode == 'empirical' else fit_effect_regression(ids, values, k, ridge, shrinkage)
    return dict(effect_mode=mode, effect_shrinkage=shrinkage, effect_mix_weight=mix_weight,
                effect_unseen=unseen, coefficients=coef, global_effect=values.mean(0),
                empirical_ids=ids[order], empirical_effects=values[order], empirical_counts=counts[order])


def effect_expectation(ids, modification, k):
    """Predict chosen center effects (N,D); NaN denotes unsupported hypotheses.

    With d=empirical difference, dbar=global average, r=globally shrunken ridge:
    empirical=(1-alpha)*d+alpha*dbar; regression=r; mixed=u*d+(1-u)*r.
    Alpha=0 disables global shrinkage. In mixed mode u is zero on unseen
    contexts. Empirical unseen contexts use dbar or a no-call, explicitly.
    """
    ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
    mode = modification['effect_mode']
    prediction = modification['global_effect'].expand(len(ids), -1).clone()
    if mode != 'empirical':
        prediction = sequence_features(ids, k).float() @ modification['coefficients']
    if mode == 'regression':
        return prediction
    keys = modification['empirical_ids']
    ix = torch.searchsorted(keys, ids)
    safe = ix.clamp_max(len(keys) - 1)
    known = (ix < len(keys)) & (keys[safe] == ids)
    direct = modification['empirical_effects'][safe[known]]
    if mode == 'mixed':
        weight = modification['effect_mix_weight']
        prediction[known] = weight * direct + (1 - weight) * prediction[known]
    else:
        alpha = modification['effect_shrinkage']
        prediction[known] = (1 - alpha) * direct + alpha * modification['global_effect']
        if modification['effect_unseen'] == 'no-call':
            prediction[~known] = float('nan')
    return prediction


def footprint_design(contexts, positions, strands, event_positions, k, radius, mode='full'):
    """Build an additive design for reference-position mean differences.

    ``contexts`` are oriented strings of length k+2*radius centered on each
    receiving position. ``event_positions`` is a list of candidate reference
    positions for that row's contig/strand. A forward offset is obs-event;
    reverse-strand offsets negate it. Central effects are excluded here and
    supplied by the separately fitted center-effect model. First four columns
    absorb receiver-base-specific sample shifts, never used as neighbor effects.
    """
    if (mode not in ('full', 'base') or any(len(v) != len(contexts) for v in (positions, strands, event_positions))
            or any(len(s) != k + 2 * radius or set(s) - set(BASES) for s in contexts)
            or any(int(s) not in (0, 1) for s in strands)):
        raise ValueError('invalid footprint contexts, coordinates, or design mode')
    width = 4 if mode == 'base' else 4 + 4 * (k - 1)
    x = torch.zeros(len(contexts), 4 + (2 * radius + 1) * width, dtype=torch.double)
    centers = []
    middle = (k + 2 * radius) // 2
    for row, (context, pos, strand, events) in enumerate(zip(contexts, positions, strands, event_positions)):
        receiver = BASES.find(context[middle])
        if receiver < 0:
            raise ValueError('noncanonical receiver in footprint statistics')
        x[row, receiver] = 1
        central_ids = []
        for event in events:
            offset = (int(pos) - int(event)) * (1 if int(strand) == 0 else -1)
            if abs(offset) > radius:
                continue
            center = middle - offset
            source = context[center - k // 2:center + k // 2 + 1]
            ids = context_ids(source, k)
            kid = int(ids[k // 2])
            if kid < 0:
                raise ValueError('incomplete source context around a labelled modification')
            if offset == 0:
                central_ids.append(kid)
                continue
            start = 4 + (offset + radius) * width
            x[row, start + receiver] += 1
            if mode == 'full':
                x[row, start + 4:start + width] += sequence_features([kid], k)[0, 1:]
        centers.append(central_ids)
    return x, centers


def fit_footprint(contexts, positions, strands, events, differences, counts,
                  center_effect, k, radius, ridge=10.0, shrinkage=0.5):
    """Fit off-center first-order effects with a nuisance background and shrinkage.

    All known same-type events are fitted simultaneously. Central effects are
    fixed to the selected center-effect recipe. Returns (2R+1,4+4*(k-1),D)
    coefficients; the center slice is zero because inference uses center effects.
    The discarded background is recorded separately for diagnostics.
    """
    if radius < 1 or radius > 12:
        raise ValueError('footprint radius must be in [1, 12]')
    if ridge <= 0 or not math.isfinite(ridge) or not 0 <= shrinkage <= 1:
        raise ValueError('invalid footprint ridge or shrinkage')
    y = torch.as_tensor(differences, dtype=torch.double).clone()
    weight = torch.as_tensor(counts, dtype=torch.double)
    if y.ndim != 2 or not len(y) or len(weight) != len(y) or len(contexts) != len(y) or not torch.isfinite(y).all():
        raise ValueError('invalid footprint mean differences')
    if not torch.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError('footprint counts must be positive and finite')
    x, central = footprint_design(contexts, positions, strands, events, k, radius)
    for row, ids in enumerate(central):
        if ids:
            center = effect_expectation(ids, center_effect, k)
            if not torch.isfinite(center).all():
                raise ValueError('footprint event has no supported center effect; provide paired prototypes or choose a fallback')
            y[row] -= center.double().sum(0)
    weight /= weight.mean()
    coefficients = []
    for mode in ('full', 'base'):
        design = x if mode == 'full' else footprint_design(contexts, positions, strands, events, k, radius, 'base')[0]
        penalty = torch.eye(design.shape[1], dtype=torch.double) * ridge
        penalty[:4, :4] *= 1e-6
        coef = torch.linalg.solve(design.T @ (design * weight[:, None]) + penalty,
                                  design.T @ (y * weight[:, None]))
        coefficients.append(coef)
    full, base = coefficients
    effect = full[4:].reshape(2 * radius + 1, -1, y.shape[1]) * (1 - shrinkage)
    effect[:, :4] += shrinkage * base[4:].reshape(2 * radius + 1, 4, y.shape[1])
    effect[radius] = 0
    return dict(coefficients=effect.float(), background=full[:4].float(),
                rows=len(y), radius=radius, ridge=ridge, shrinkage=shrinkage)


def validate_effect_model(model):
    """Validate the single versioned effect artifact contract before prediction."""
    if model.get('format_version') != FORMAT_VERSION or model.get('artifact_type') != ARTIFACT_TYPE:
        raise ValueError('unsupported context-conditioned effect artifact; refit with fit-effect-prototypes for format version 2')
    k, dim, radius = model['kmer_size'], model['embedding_dim'], model['footprint_radius']
    validate_k(k)
    if not isinstance(dim, int) or dim < 1 or not isinstance(radius, int) or not 0 <= radius <= 12:
        raise ValueError('invalid effect dimension or footprint radius')
    labels = model['class_labels']
    if not labels or len(set(labels)) != len(labels) or any(not isinstance(s, str) or not s.strip() for s in labels):
        raise ValueError('effect class labels must be nonempty and unique')
    if 'semantics' in labels:
        raise ValueError('class label semantics would collide with the score_semantics output column')
    if model.get('embedding_transform') != 'identity':
        raise ValueError('effect artifacts currently require raw, identity embedding coordinates')
    if not math.isfinite(model['modification_penalty']) or model['modification_penalty'] < 0:
        raise ValueError('modification penalty must be finite and nonnegative')
    seen = []
    if set(model['families']) != set(BASES):
        raise ValueError('all four receiving-base canonical families are required')
    for base, family in model['families'].items():
        if base not in BASES:
            raise ValueError('invalid canonical family')
        validate_k(family['context_k'])
        canonical = family['canonical_class_index']
        seen.append(canonical)
        ids, counts, means = family['canonical_ids'], family['canonical_counts'], family['canonical_means']
        if (ids.dtype != torch.long or counts.dtype != torch.long or ids.ndim != 1 or not len(ids)
                or counts.shape != ids.shape or means.shape != (len(ids), dim)):
            raise ValueError('invalid canonical context table shape')
        if (ids[1:] <= ids[:-1]).any() or ((ids < 0) | (ids >= 4 ** family['context_k'])).any() or (counts <= 0).any():
            raise ValueError('canonical IDs/counts are not sorted, unique and valid')
        center = (ids // 4 ** (family['context_k'] // 2)) % 4
        if (center != BASES.index(base)).any():
            raise ValueError('canonical contexts belong to the wrong center base')
        strength = family['canonical_strength']
        if not math.isfinite(strength) or strength < 0:
            raise ValueError('canonical strength must be finite and nonnegative')
        if family['canonical_mean_mode'] not in CANONICAL_MODES or family['canonical_unseen'] not in ('regression', 'no-call'):
            raise ValueError('invalid canonical mean mode or unseen policy')
        tensors = [means, family['whitening']]
        canonical_coef = family['canonical_coefficients']
        needs_regression = family['canonical_mean_mode'] != 'empirical' or family['canonical_unseen'] == 'regression'
        if needs_regression and canonical_coef is None:
            raise ValueError('canonical mode requires regression coefficients')
        if canonical_coef is not None:
            if canonical_coef.shape != (1 + 4 * (family['context_k'] - 1), dim):
                raise ValueError('invalid canonical regression shape')
            tensors.append(canonical_coef)
        if family['whitening'].shape != (dim, dim):
            raise ValueError('invalid canonical whitening shape')
        if (family['whitening'].diag() <= 0).any() or family['whitening'].tril(-1).abs().max() > 1e-5:
            raise ValueError('canonical whitening must be nonsingular upper triangular (row inverse-Cholesky)')
        if family['noise_mode'] not in NOISE_MODES:
            raise ValueError('invalid noise mode')
        if family['noise_mode'] == 'none' and not torch.equal(family['whitening'], torch.eye(dim)):
            raise ValueError('disabled whitening requires the identity transform')
        if family['noise_mode'] == 'hierarchical-5mer':
            noise_ids, transforms = family['noise_context_ids'], family['noise_context_whitening']
            if (noise_ids.dtype != torch.long or noise_ids.ndim != 1 or not len(noise_ids)
                    or transforms.shape != (len(noise_ids), dim, dim)
                    or (noise_ids < 0).any() or (noise_ids >= 4 ** 5).any()
                    or (noise_ids[1:] <= noise_ids[:-1]).any()
                    or (noise_ids // 4 ** 2 % 4 != BASES.index(base)).any()):
                raise ValueError('invalid hierarchical receiver-5-mer noise table')
            # Check one matrix at a time to avoid copying an entire GiB-scale
            # table just to inspect its lower triangles at artifact load time.
            for transform in transforms:
                if not torch.isfinite(transform).all() or (transform.diag() <= 0).any() or transform.tril(-1).abs().max() > 1e-5:
                    raise ValueError('invalid local noise whitening matrix')
        for modification in family['modifications']:
            seen.append(modification['class_index'])
            coef = modification['coefficients']
            if modification['effect_mode'] not in EFFECT_MODES or modification['effect_unseen'] not in ('global', 'no-call'):
                raise ValueError('invalid effect mode or unseen policy')
            for name in ('effect_shrinkage', 'effect_mix_weight'):
                value = modification[name]
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'{name} must be in [0, 1]')
            if modification['effect_mode'] != 'empirical' and coef is None:
                raise ValueError('selected effect mode requires regression coefficients')
            if coef is not None:
                if coef.shape != (1 + 4 * (k - 1), dim):
                    raise ValueError('invalid modification effect regression shape')
                tensors.append(coef)
            keys, empirical, counts = (modification['empirical_ids'], modification['empirical_effects'], modification['empirical_counts'])
            if (keys.dtype != torch.long or keys.ndim != 1 or not len(keys)
                    or counts.dtype != torch.long or counts.shape != (len(keys), 2) or (counts <= 0).any()
                    or empirical.shape != (len(keys), dim) or modification['global_effect'].shape != (dim,)
                    or (keys[1:] <= keys[:-1]).any() or (keys < 0).any() or (keys >= 4 ** k).any()
                    or (keys // 4 ** (k // 2) % 4 != BASES.index(base)).any()):
                raise ValueError('invalid empirical effect table')
            tensors.extend([empirical, modification['global_effect']])
            footprint = modification.get('footprint')
            if radius and footprint is None:
                raise ValueError('neighbor scoring requires a footprint for every modification')
            if footprint is not None:
                if footprint['coefficients'].shape != (2 * radius + 1, 4 + 4 * (k - 1), dim):
                    raise ValueError('invalid footprint coefficient shape')
                if footprint['coefficients'][radius].count_nonzero():
                    raise ValueError('footprint center must be zero; central effects are separate')
                tensors.append(footprint['coefficients'])
        if any(not torch.isfinite(t).all() for t in tensors):
            raise ValueError('nonfinite effect model parameters')
    if sorted(seen) != list(range(len(labels))):
        raise ValueError('classes must occur exactly once across canonical families')


def load_effect_model(path):
    """Load a trusted project artifact on CPU and validate its full contract."""
    model = torch.load(path, map_location='cpu', weights_only=False)
    validate_effect_model(model)
    return model


class EffectInferenceCache:
    """Bounded, process-local effect cache for one immutable fitted model.

    Pass the same instance to successive ``score_effect_read`` calls to reuse
    source features/center effects, raw neighbor effects, and whitened effects.
    No query embeddings, validity masks, penalties, or inferred states are
    stored. Hierarchical whitened entries include the receiving 5-mer; base
    noise entries need only the receiving base.

    ``max_bytes`` bounds retained tensor payload, excluding Python bookkeeping
    and the referenced model. Least recently used entries are evicted; zero
    disables retention. Construction is lazy and does not modify or serialize
    the model. The cache is intended for serial inference. After changing any
    fitted model parameters, call ``clear()`` or construct a new cache.
    """

    def __init__(self, model, max_bytes=DEFAULT_EFFECT_CACHE_MB * 1024 ** 2):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError('effect cache max_bytes must be a nonnegative integer')
        self.model = model
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self.size_bytes = 0

    def clear(self):
        """Release all retained effects before reusing a mutated model."""
        self._entries.clear()
        self.size_bytes = 0

    def _get(self, key):
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry[0]

    def _put(self, key, value, size_bytes):
        """Retain a value only if its tensor payload fits the shared budget."""
        if not self.max_bytes or size_bytes > self.max_bytes:
            return
        while self.size_bytes + size_bytes > self.max_bytes:
            _, (_, removed_bytes) = self._entries.popitem(last=False)
            self.size_bytes -= removed_bytes
        self._entries[key] = (value, size_bytes)
        self.size_bytes += size_bytes

    def source_context(self, source_id):
        """Return a feature vector, ordered center effects, and support flag.

        Vectors use the original one-context regression arithmetic. NaN effects
        remain cached as unsupported hypotheses, preserving no-call masking.
        Returned tensors are shared cache entries and must not be modified.
        """
        key = ('source', source_id)
        value = self._get(key)
        if value is None:
            k = self.model['kmer_size']
            base = BASES[source_id // 4 ** (k // 2) % 4]
            feature = sequence_features([source_id], k).float()[0]
            effects = tuple(effect_expectation([source_id], mod, k)[0].detach()
                            for mod in self.model['families'][base]['modifications'])
            supported = all(bool(torch.isfinite(effect).all()) for effect in effects)
            value = (feature, effects, supported)
            size = sum(t.numel() * t.element_size() for t in (feature, *effects))
            self._put(key, value, size)
        return value

    def whitened_effect(self, source_id, modification_index, offset, receiver,
                        receiver_id, transform, source_context):
        """Return a shared length-D effect in the receiving noise metric.

        Source ID and modification index identify the source-base hypothesis.
        ``receiver_id`` is its receiving position's 5-mer (not the source's).
        ``transform`` must be that receiving position's fitted whitening view.
        Raw effects are cached separately so different receiving noise contexts
        can reuse the same source/offset/base regression. Offset zero uses the
        center effect, never the fitted off-center coefficients.
        """
        family = self.model['families'][receiver]
        noise_id = receiver_id if family['noise_mode'] == 'hierarchical-5mer' else None
        effect_key = (source_id, modification_index, offset, receiver)
        white_key = ('white', *effect_key, noise_id)
        whitened = self._get(white_key)
        if whitened is not None:
            return whitened
        feature, centers, _ = source_context
        if offset == 0:
            effect = centers[modification_index]
        else:
            raw_key = ('raw', *effect_key)
            effect = self._get(raw_key)
            if effect is None:
                k = self.model['kmer_size']
                base = BASES[source_id // 4 ** (k // 2) % 4]
                modification = self.model['families'][base]['modifications'][modification_index]
                coef = modification['footprint']['coefficients'][offset + self.model['footprint_radius']]
                effect = (coef[BASES.index(receiver)] + feature[1:] @ coef[4:]).detach()
                self._put(raw_key, effect, effect.numel() * effect.element_size())
        whitened = (effect @ transform).detach()
        self._put(white_key, whitened, whitened.numel() * whitened.element_size())
        return whitened


def build_potentials(sequence, embeddings, valid, model, neighbors=True, penalty=None, *, cache=None):
    """Build Gaussian unary/pair potentials for a read, without label information.

    Observations are (L,D); validity is (L,). Every receiving position has its
    own canonical mean and receiving-base/5-mer noise transform. Pair interactions are
    nonzero only for footprints overlapping within 2R positions. Templates are
    retained only for the current 2R frontier, bounding temporary memory.

    Returns lists of unary score vectors, global class IDs, sparse earlier-pair
    matrices, and whitened observations/availability. Missing observations have
    only a placeholder no-call state, never a canonical call.

    ``cache`` optionally reuses model-only effects across reads. It must belong
    to this exact, unchanged model. Omission creates a cache for this read only;
    passing a zero-budget cache disables effect retention.
    """
    if cache is None:
        cache = EffectInferenceCache(model)
    elif cache.model is not model:
        raise ValueError('effect cache belongs to a different model')
    seq = sequence.upper().replace('U', 'T')
    h = torch.as_tensor(embeddings, device='cpu', dtype=torch.float32)
    valid = torch.as_tensor(valid, device='cpu', dtype=torch.bool).clone()
    length, dim = len(seq), model['embedding_dim']
    if h.shape != (length, dim) or valid.shape != (length,):
        raise ValueError('read embedding/validity dimensions do not match sequence')
    valid &= torch.isfinite(h).all(1)
    radius = model['footprint_radius'] if neighbors else 0
    cost = model['modification_penalty'] if penalty is None else penalty
    if not math.isfinite(cost) or cost < 0:
        raise ValueError('modification penalty must be finite and nonnegative')
    source_ids = context_ids(seq, model['kmer_size'])
    receiver_ids = context_ids(seq, 5)
    observed = torch.zeros_like(h)
    usable = torch.zeros(length, dtype=torch.bool)
    mean = torch.zeros_like(h)
    ids_by_k = {}
    # Shared references to fitted matrix views, never an L*D*D tensor.
    whiteners = [None] * length
    for base, family in model['families'].items():
        context_k = family['context_k']
        if context_k not in ids_by_k:
            ids_by_k[context_k] = context_ids(seq, context_k)
        ids = ids_by_k[context_k]
        mask = torch.tensor([c == base for c in seq], dtype=torch.bool) & valid & (ids >= 0)
        if mask.any():
            mean[mask] = canonical_expectation(ids[mask], family)
            mask &= torch.isfinite(mean).all(1)
            grouping = receiver_ids if family['noise_mode'] == 'hierarchical-5mer' else torch.zeros(length, dtype=torch.long)
            for kid in grouping[mask].unique().tolist():
                transform = receiver_whitening(family, kid)
                if transform is None:
                    continue
                group = mask & (grouping == kid)
                observed[group] = (h[group] - mean[group]) @ transform
                usable[group] = True
                for index in torch.where(group)[0].tolist():
                    whiteners[index] = transform
    source_contexts = {}
    unsupported_effect = torch.zeros(length, dtype=torch.bool)
    # A missing empirical hypothesis must not be treated as a canonical
    # neighbor. Mask observations that this unknown event could perturb; the
    # complete-window rule below then conservatively masks overlapping calls.
    for kid in source_ids[source_ids >= 0].unique().tolist():
        context = cache.source_context(kid)
        source_contexts[kid] = context
        if not context[2]:
            unsupported_effect |= source_ids == kid
    if unsupported_effect.any():
        affected = torch.nn.functional.max_pool1d(unsupported_effect.float()[None, None],
                                                  kernel_size=2 * radius + 1, stride=1, padding=radius)[0, 0].bool()
        usable &= ~affected
        observed[affected] = 0
    unaries, classes, interactions = [], [], []
    frontier = deque()
    for pos, base in enumerate(seq):
        family = model['families'].get(base)
        supported = family is not None and usable[pos] and source_ids[pos] >= 0
        # Require a complete observed footprint. Never reinterpret a truncated
        # or unaligned neighborhood as zero evidence for a modification.
        start, end = pos - radius, pos + radius + 1
        supported = supported and start >= 0 and end <= length and bool(usable[start:end].all())
        if not supported:
            unaries.append(torch.zeros(1))
            classes.append([-1])
            interactions.append({})
            continue
        source_id = int(source_ids[pos])
        context = source_contexts[source_id]
        mods = family['modifications']
        template = torch.zeros(1 + len(mods), 2 * radius + 1, dim)
        for m in range(len(mods)):
            for offset in range(-radius, radius + 1):
                receiver = seq[pos + offset]
                template[m + 1, offset + radius] = cache.whitened_effect(
                    source_id, m, offset, receiver, int(receiver_ids[pos + offset]),
                    whiteners[pos + offset], context)
        unary = (template * observed[start:end]).sum((1, 2)) - .5 * template.square().sum((1, 2))
        unary[1:] -= cost
        unaries.append(unary)
        classes.append([family['canonical_class_index']] + [m['class_index'] for m in mods])
        while frontier and pos - frontier[0][0] > 2 * radius:
            frontier.popleft()
        pairs = {}
        for previous, old_template in frontier:
            distance = pos - previous
            overlap = 2 * radius + 1 - distance
            pairs[previous] = template[:, :overlap].flatten(1) @ old_template[:, distance:].flatten(1).T
        interactions.append(pairs)
        frontier.append((pos, template))
    return dict(unaries=unaries, classes=classes, interactions=interactions,
                whitened_observations=observed, usable=usable, radius=radius, canonical_mean=mean)


def _decode_potentials_python(unaries, interactions, radius, beam_width=32, max_sweeps=8):
    """Approximate finite-range MAP using beam DP, then monotone coordinate polish.

    Score = sum(unary(state)) - sum(overlap inner products). Thus neither
    overlapping evidence nor modification costs are counted twice. Multiple
    adjacent modifications are allowed. Beam states merge on their live 2R
    suffix; history uses linked backpointers instead of quadratic-length copies.
    Returns selected local states, conditional score vectors, convergence flag,
    and objective relative to the entirely canonical configuration.

    Inputs and returned score vectors are CPU tensors. Decoding uses NumPy
    views internally to avoid tensor dispatch for every scalar beam transition.
    Conditional updates retain the input dtype and subtraction order; beam
    accumulation uses Python floats, with the original stable tie ordering.
    Inputs are never modified.
    """
    if beam_width < 1 or max_sweeps < 1:
        raise ValueError('beam width and coordinate sweeps must be positive')
    unaries = [unary.detach().numpy() for unary in unaries]
    interactions = [{previous: matrix.detach().numpy() for previous, matrix in pairs.items()}
                    for pairs in interactions]
    beams = [(0.0, (), None)]
    for pos, unary in enumerate(unaries):
        merged = {}
        for previous_score, suffix, node in beams:
            for state in range(len(unary)):
                score = previous_score + float(unary[state])
                for previous, matrix in interactions[pos].items():
                    score -= float(matrix[state, suffix[previous - pos]])
                key = (suffix + (state,))[-2 * radius:] if radius else ()
                if key not in merged or score > merged[key][0]:
                    merged[key] = (score, key, (node, state))
        beams = sorted(merged.values(), key=lambda item: item[0], reverse=True)[:beam_width]
    states = []
    node = beams[0][2]
    while node is not None:
        node, state = node
        states.append(state)
    states.reverse()
    future = [[] for _ in states]
    for pos, pairs in enumerate(interactions):
        for previous in pairs:
            future[previous].append(pos)

    def conditional(pos):
        score = unaries[pos].copy()
        for previous, matrix in interactions[pos].items():
            score -= matrix[:, states[previous]]
        for later in future[pos]:
            score -= interactions[later][pos][states[later]]
        return score

    converged = False
    for sweep in range(max_sweeps):
        changed = False
        order = range(len(states)) if sweep % 2 == 0 else range(len(states) - 1, -1, -1)
        for pos in order:
            score = conditional(pos)
            best = int(score.argmax())
            if float(score[best] - score[states[pos]]) > 1e-6:
                states[pos] = best
                changed = True
        if not changed:
            converged = True
            break
    scores = [torch.from_numpy(conditional(pos)) for pos in range(len(states))]
    objective = sum(float(unaries[p][state]) for p, state in enumerate(states))
    objective -= sum(float(matrix[states[p], states[q]]) for p, pairs in enumerate(interactions) for q, matrix in pairs.items())
    return states, scores, converged, objective


def decode_potentials(unaries, interactions, radius, beam_width=32, max_sweeps=8, *, decoder='auto'):
    """Decode with the optional compiled kernel or the unchanged Python path.

    Auto selects Numba when installed. Both paths preserve beam width, stable
    suffix ties, input-dtype conditional arithmetic, and coordinate sweeps.
    Nonfinite/unusual-dtype inputs retain reference behavior in auto mode;
    explicit numba mode validates and rejects unsupported inputs.
    """
    from .effects_decode import decoder_backend, decode_arrays, pack_potentials
    if decoder_backend(decoder) == 'python':
        return _decode_potentials_python(unaries, interactions, radius, beam_width, max_sweeps)
    if decoder == 'auto' and any(row.dtype not in (torch.float32, torch.float64)
                                or not torch.isfinite(row).all() for row in
                                [*unaries, *(matrix for pairs in interactions for matrix in pairs.values())]):
        return _decode_potentials_python(unaries, interactions, radius, beam_width, max_sweeps)
    values, counts, pairs, order = pack_potentials(unaries, interactions, radius)
    states, scores, converged, objective = decode_arrays(values, counts, pairs, order, beam_width, max_sweeps)
    return states.tolist(), [torch.from_numpy(row[:n]) for row, n in zip(scores, counts)], converged, objective


def score_effect_read(sequence, embeddings, valid, model, *, neighbors=True,
                      penalty=None, beam_width=32, max_sweeps=8, cache=None, decoder='auto'):
    """Classify a whole stitched read, jointly across canonical base families.

    Returns global (L,C) conditional scores and selected global indices. Scores
    condition on the chosen neighbor configuration and are uncalibrated.
    Unscorable rows are NaN/-1.
    ``labels`` intentionally are not an input: output selection must happen
    after all neighboring bases have participated in inference.

    Pass one ``EffectInferenceCache`` across reads to retain model-only work.
    The cache requires an unchanged model, but is independent of query signal,
    validity, penalty, neighbor mode, beam width, and refinement sweep limit.
    """
    potentials = build_potentials(sequence, embeddings, valid, model, neighbors, penalty, cache=cache)
    return score_effect_potentials(potentials, model['class_labels'], beam_width=beam_width,
                                   max_sweeps=max_sweeps, decoder=decoder)


def score_effect_potentials(potentials, class_labels, *, beam_width=32, max_sweeps=8, decoder='auto'):
    """Decode CPU potentials and return the standard per-read call tensors.

    ``potentials`` contains CPU unary/pair tensors, per-position class lists,
    radius, and either a length-L ``canonical_energy`` tensor or the CPU
    builder's ``whitened_observations``. Auto uses optional Numba for decoding
    and batches result assembly by local class count. Python selects the
    reference path. No embeddings are transferred to the decoder.
    """
    import numpy as np
    from .effects_decode import decoder_backend, pack_potentials
    if decoder_backend(decoder) == 'numba':
        arrays = pack_potentials(potentials['unaries'], potentials['interactions'], potentials['radius'])
        values, counts, pairs, order = arrays
        if decoder != 'auto' or (values.dtype in (np.dtype('float32'), np.dtype('float64'))
                                 and np.isfinite(values).all() and np.isfinite(pairs).all()):
            classes = torch.full(values.shape, -1, dtype=torch.long)
            for pos, ids in enumerate(potentials['classes']):
                classes[pos, :len(ids)] = torch.tensor(ids)
            energy = (potentials['canonical_energy'] if 'canonical_energy' in potentials
                      else potentials['whitened_observations'].square().sum(1))
            return _score_effect_arrays(arrays, classes, class_labels, energy, potentials['radius'],
                                        beam_width, max_sweeps)
    states, conditional, converged, objective = decode_potentials(
        potentials['unaries'], potentials['interactions'], potentials['radius'], beam_width, max_sweeps, decoder='python')
    length, count = len(potentials['unaries']), len(class_labels)
    scores = torch.full((length, count), float('nan'))
    predicted = torch.full((length,), -1, dtype=torch.long)
    isolated = torch.full((length,), -1, dtype=torch.long)
    neighbors_called = torch.zeros(length, dtype=torch.long)
    for pos, (local, global_ids) in enumerate(zip(conditional, potentials['classes'])):
        if global_ids == [-1]:
            continue
        scores[pos, global_ids] = local
        predicted[pos] = global_ids[states[pos]]
        isolated[pos] = global_ids[int(potentials['unaries'][pos].argmax())]
        radius = potentials['radius']
        neighbors_called[pos] = sum(states[j] > 0 for j in range(max(0, pos - radius), min(length, pos + radius + 1)) if j != pos)
    return dict(scores=scores, predicted_class_index=predicted,
                isolated_class_index=isolated, neighbor_count=neighbors_called,
                converged=converged, objective=objective,
                canonical_energy=(potentials['canonical_energy'] if 'canonical_energy' in potentials
                                  else potentials['whitened_observations'].square().sum(1)))


def score_effect_packet(packet, class_labels, *, beam_width=32, max_sweeps=8, decoder='auto'):
    """Decode a complete CPU score packet without constructing per-edge tensors.

    GPU scoring has already transferred all packet tensors to CPU. Numba reads
    their NumPy views directly, avoiding sparse dictionaries and per-base tensor
    slices. Python intentionally retains the reference packet adapter for
    controlled comparisons. Statistical settings and output schemas are shared.
    """
    from .effects_decode import decoder_backend, packet_order
    if decoder_backend(decoder) == 'python':
        return score_effect_potentials(packet.potentials(), class_labels, beam_width=beam_width,
                                        max_sweeps=max_sweeps, decoder='python')
    arrays = (packet.unaries.numpy(), packet.counts.numpy(), packet.pairs.numpy(), packet_order(packet))
    return _score_effect_arrays(arrays, packet.classes, class_labels, packet.canonical_energy,
                                packet.radius, beam_width, max_sweeps)


def _score_effect_arrays(arrays, classes, class_labels, energy, radius, beam_width, max_sweeps):
    """Batch local score scatter by state count, never across padding.

    Arrays use local class indices; classes[L,S] maps them to global labels.
    The existing output contract is float32 scores with NaN for
    unavailable/off-family values and -1 for unavailable class indices.
    """
    from .effects_decode import decode_arrays
    values, counts, _, _ = arrays
    states, local, converged, objective = decode_arrays(*arrays, beam_width, max_sweeps)
    states, local, unaries = torch.from_numpy(states), torch.from_numpy(local), torch.from_numpy(values)
    length, nclasses = len(values), len(class_labels)
    scores = torch.full((length, nclasses), float('nan'))
    predicted = torch.full((length,), -1, dtype=torch.long)
    isolated = torch.full_like(predicted, -1)
    available = classes[:, 0] >= 0
    for count in set(counts.tolist()):
        positions = torch.where(available & torch.from_numpy(counts == count))[0]
        if not len(positions):
            continue
        ids = classes[positions, :count]
        conditional = local[positions, :count]
        scores[positions[:, None], ids] = conditional.to(scores.dtype)
        predicted[positions] = ids.gather(1, states[positions, None])[:, 0]
        isolated[positions] = ids.gather(1, unaries[positions, :count].argmax(1, keepdim=True))[:, 0]
    modified = (states > 0).long()
    prefix = torch.cat([torch.zeros(1, dtype=torch.long), modified.cumsum(0)])
    positions = torch.arange(length)
    neighbors = prefix[(positions + radius + 1).clamp_max(length)] - prefix[(positions - radius).clamp_min(0)] - modified
    neighbors[~available] = 0
    return dict(scores=scores, predicted_class_index=predicted,
                isolated_class_index=isolated, neighbor_count=neighbors, converged=converged,
                objective=objective, canonical_energy=energy)
