"""Fit context-conditioned effect artifacts from explicitly paired data sources.

The manifest identifies canonical-only reservoirs separately from modified
centroids. Optional canonical 9-mer tables replace the 5-mer canonical baseline;
modified effects remain learned from matched 5-mer contrasts. Paths are resolved
relative to the manifest, and checkpoint provenance is checked before fitting.
"""

from __future__ import annotations

from porefm.preprocessing import validate_artifact_preprocessing
import json
from pathlib import Path

from porefm.cli._parser import PoreFMArgumentParser
from .effects_options import DEFAULTS, EFFECT_OVERRIDES, add_fit_options, resolve_options


def read_events(path):
    """Read single-base BED6 events as {(contig, 0/1 strand): set(0-based pos)}.

    BED name/score are ignored. Multi-base intervals and duplicate coordinates
    are rejected to avoid silently changing modification multiplicity.
    """
    events = {}
    with open(path) as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith(('#', 'track ', 'browser ')):
                continue
            fields = line.split()
            if len(fields) < 6 or fields[5] not in ('+', '-'):
                raise ValueError(f'{path}:{number}: expected stranded BED6')
            start, end = int(fields[1]), int(fields[2])
            if start < 0 or end != start + 1:
                raise ValueError(f'{path}:{number}: expected a single-base interval')
            key = (fields[0], int(fields[5] == '-'))
            if start in events.setdefault(key, set()):
                raise ValueError(f'{path}:{number}: duplicate event')
            events[key].add(start)
    if not events:
        raise ValueError(f'{path}: empty event file')
    return events


def load_statistics(path, *, checkpoint_sha256, embedding_key, dim, k, radius):
    """Validate the versioned, read-deduplicated neighborhood mean contract."""
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('artifact_type') != 'effect_neighborhood_statistics' or payload.get('format_version') != 1:
        raise ValueError(f'{path}: unsupported neighborhood statistics')
    expected = dict(embedding_key=embedding_key, embedding_dim=dim, kmer_size=k, radius=radius)
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{path}: neighborhood representation/dimensions do not match model')
    if payload['metadata'].get('checkpoint_sha256') != checkpoint_sha256:
        raise ValueError(f'{path}: neighborhood checkpoint mismatch')
    n = len(payload['contexts'])
    if (not n or payload['means'].shape != (n, dim)
            or any(len(payload[key]) != n for key in ('chroms', 'positions', 'strands', 'counts'))
            or not torch.isfinite(payload['means']).all() or (payload['counts'] <= 0).any()):
        raise ValueError(f'{path}: invalid neighborhood arrays')
    if ((payload['positions'] < 0).any() or not torch.isin(payload['strands'], torch.tensor([0, 1])).all()
            or any(len(s) != k + 2 * radius or set(s) - set('ACGT') for s in payload['contexts'])):
        raise ValueError(f'{path}: invalid neighborhood coordinates or contexts')
    keys = list(zip(payload['chroms'], payload['positions'].tolist(), payload['strands'].tolist()))
    if len(set(keys)) != n:
        raise ValueError(f'{path}: duplicate neighborhood rows')
    return payload, {key: i for i, key in enumerate(keys)}


def fit_manifest(manifest, root):
    """Fit a CPU artifact from a manifest dictionary and its path-resolution root.

    Parameters
    ----------
    manifest : dict
        See ``docs/prototype_effects.md`` for the canonical/modification schema.
        Each canonical family supplies context tables and noise reservoirs;
        each modification supplies a matched control and optional BED-linked
        footprint pairs. No query labels or evaluation reads are consumed.
    root : path-like
        Directory against which relative input paths are resolved.

    Returns
    -------
    dict
        A validated, CPU-only model with source metadata and numerical fit
        diagnostics. Saving the result is the caller's responsibility.
    """
    import torch
    from . import effects as e
    from .embedding_selection import EMBEDDING_KEY_CHOICES

    allowed = set(DEFAULTS) | {'backbone_path', 'allow_unverified_prototypes', 'allow_unverified_statistics', 'kmer_size',
               'footprint_radius', 'effect_ridge', 'effect_shrinkage', 'min_count', 'noise_samples_per_context',
               'noise_shrinkage', 'noise_floor', 'canonical_strength', 'seed', 'modification_penalty',
               'canonical', 'modifications'}
    if set(manifest) - allowed:
        raise ValueError(f'unknown effect manifest options: {sorted(set(manifest) - allowed)}')
    settings = resolve_options(manifest)
    root = Path(root)
    path = lambda value: str((root / value).resolve())
    k, radius = manifest.get('kmer_size', 5), manifest.get('footprint_radius', 4)
    e.validate_k(k)
    if not isinstance(radius, int) or not 0 <= radius <= 12:
        raise ValueError('footprint_radius must be an integer in [0, 12]')
    min_count = manifest.get('min_count', 2)
    if not isinstance(min_count, int) or min_count < 1:
        raise ValueError('min_count must be a positive integer')
    noise_cap = manifest.get('noise_samples_per_context', 256)
    if not isinstance(noise_cap, int) or noise_cap < 2:
        raise ValueError('noise_samples_per_context must be an integer >= 2')
    backbone_path = path(manifest['backbone_path'])
    fingerprint = e.checkpoint_digest(backbone_path)
    sources, unverified, cache = {}, [], {}
    unverified_statistics = []
    representation = [None, None]

    def prototype(value):
        """Memory-map trusted reservoirs; validate without copying their samples."""
        source = path(value)
        if source in cache:
            return cache[source]
        p = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
        meta = p['metadata']
        if meta.get('dry_run_no_model', False):
            raise ValueError(f'{source}: dry-run prototypes are not backbone embeddings')
        key = meta.get('embedding_key')
        if key not in EMBEDDING_KEY_CHOICES:
            raise ValueError(f'{source}: missing/unknown embedding_key')
        ids, counts, means = p['kmer_ids'], p['counts'], p['mean_embeddings']
        size = meta['kmer_size']
        e.validate_k(size)
        if (ids.dtype != torch.long or counts.dtype != torch.long
                or ids.ndim != 1 or counts.shape != ids.shape or means.ndim != 2 or means.shape[1] < 1
                or len(means) != len(ids) or not len(ids) or not torch.isfinite(means).all()
                or (counts <= 0).any() or ((ids < 0) | (ids >= 4 ** size)).any()
                or len(ids.unique()) != len(ids)):
            raise ValueError(f'{source}: invalid prototype arrays')
        if representation[0] is None:
            representation[:] = [key, means.shape[1]]
        if representation != [key, means.shape[1]] or meta.get('embedding_transform', 'identity') != 'identity':
            raise ValueError(f'{source}: incompatible raw embedding representation')
        actual = meta.get('checkpoint_sha256')
        # An acknowledged legacy union may still contain a known mismatched
        # checkpoint. Missing top-level provenance must not hide that evidence.
        pending = [meta]
        while pending:
            recorded = pending.pop()
            known = recorded.get('checkpoint_sha256')
            if known is not None and known != fingerprint:
                raise ValueError(f'{source}: checkpoint_sha256 mismatch in prototype or union source')
            pending.extend(item['metadata'] for item in recorded.get('source_summaries', []) if 'metadata' in item)
        if actual is None and manifest.get('allow_unverified_prototypes', False):
            unverified.append(source)
        elif actual != fingerprint:
            raise ValueError(f'{source}: missing/mismatched checkpoint_sha256; legacy sources require explicit acknowledgement')
        sources[source] = dict(kind='prototype', metadata=meta)
        cache[source] = p
        return p

    def table(values, base):
        """Pool canonical means by exact context, retaining count information."""
        if not values or not isinstance(values, list):
            raise ValueError('canonical prototype sources must be a nonempty list')
        if len(set(map(path, values))) != len(values):
            raise ValueError('duplicate canonical prototype source')
        grouped, sizes = {}, set()
        for value in values:
            p = prototype(value)
            size = p['metadata']['kmer_size']
            sizes.add(size)
            for idx, kid in enumerate(p['kmer_ids'].tolist()):
                if kid // 4 ** (size // 2) % 4 != e.BASES.index(base):
                    continue
                n = int(p['counts'][idx])
                total, count = grouped.get(kid, (torch.zeros(representation[1], dtype=torch.double), 0))
                grouped[kid] = (total + n * p['mean_embeddings'][idx].double(), count + n)
        if len(sizes) != 1:
            raise ValueError('canonical context tables must all use the same k-mer size')
        grouped = {kid: item for kid, item in grouped.items() if item[1] >= min_count}
        ids = sorted(grouped)
        if not ids:
            raise ValueError(f'no canonical contexts for {base}')
        return (torch.tensor(ids), torch.tensor([grouped[x][1] for x in ids]),
                torch.stack([grouped[x][0] / grouped[x][1] for x in ids]).float(), sizes.pop())

    def noise_groups(values, base):
        """Yield (context ID, context k, samples) without combining context means."""
        if not isinstance(values, list) or not values or len(set(map(path, values))) != len(values):
            raise ValueError('noise sources must be a nonempty list without duplicates')
        for value in values:
            p = prototype(value)
            size = p['metadata']['kmer_size']
            offsets, counts, values_ = p['reservoir_offsets'], p['reservoir_counts'], p['reservoir_embeddings']
            if (offsets.dtype != torch.long or counts.dtype != torch.long
                    or offsets.shape != (len(p['kmer_ids']) + 1,) or counts.shape != p['kmer_ids'].shape
                    or int(offsets[0]) != 0 or int(offsets[-1]) != len(values_)
                    or (offsets[1:] - offsets[:-1] != counts).any() or (counts < 0).any()
                    or (counts > p['counts']).any() or values_.shape != (len(values_), representation[1])):
                raise ValueError(f'{value}: invalid canonical reservoir layout')
            for idx, kid in enumerate(p['kmer_ids'].tolist()):
                if kid // 4 ** (size // 2) % 4 == e.BASES.index(base):
                    group = values_[int(offsets[idx]):int(offsets[idx + 1])]
                    if len(group) > noise_cap:
                        generator = torch.Generator().manual_seed(kid + manifest.get('seed', 0))
                        group = group[torch.randperm(len(group), generator=generator)[:noise_cap]]
                    yield kid, size, group

    canonical = {base.replace('U', 'T'): spec for base, spec in manifest['canonical'].items()}
    if len(canonical) != len(manifest['canonical']) or set(canonical) != set(e.BASES):
        raise ValueError('provide canonical A, C, G, and T (or U) families for receiving-base noise metrics')
    for spec in canonical.values():
        if set(spec) - {'label', 'prototypes', 'noise_prototypes'}:
            raise ValueError('unknown canonical options; use prototypes for context means and noise_prototypes for noise')
    labels, families = [], {}
    for base in e.BASES:
        spec = canonical[base]
        ids, counts, means, size = table(spec['prototypes'], base)
        mode = settings['noise_mode']
        if mode == 'none':
            noise = dict(covariance=torch.eye(representation[1]), whitening=torch.eye(representation[1]),
                         observations=0, contexts=0, degrees_of_freedom=0)
        else:
            groups = noise_groups(spec.get('noise_prototypes', spec['prototypes']), base)
            if mode == 'hierarchical-5mer':
                from .effects_noise import fit_hierarchical_noise
                noise = fit_hierarchical_noise(groups, representation[1], shrinkage=settings['noise_shrinkage'],
                                               floor=settings['noise_floor'], strength=settings['noise_local_strength'],
                                               local_shrinkage=settings['noise_local_shrinkage'])
            else:
                noise = e.canonical_noise_metric((x for _, _, x in groups), representation[1],
                                                 settings['noise_shrinkage'], settings['noise_floor'])
        labels.append(spec.get('label', base))
        needs_regression = settings['canonical_mean_mode'] != 'empirical' or settings['canonical_unseen'] == 'regression'
        coefficients = e.fit_effect_regression(ids, means, size, settings['canonical_ridge'], 0.) if needs_regression else None
        families[base] = dict(context_k=size, canonical_class_index=len(labels) - 1,
                              canonical_ids=ids, canonical_counts=counts, canonical_means=means,
                              canonical_coefficients=coefficients, canonical_mean_mode=settings['canonical_mean_mode'],
                              canonical_unseen=settings['canonical_unseen'], canonical_strength=settings['canonical_strength'],
                              noise_mode=mode,
                              whitening=noise.pop('whitening'), noise_diagnostics=noise, modifications=[])
        if mode == 'hierarchical-5mer':
            families[base]['noise_context_ids'] = noise.pop('context_ids')
            families[base]['noise_context_whitening'] = noise.pop('context_whitening')
        print(f'Canonical {base}: {len(ids)} {size}-mers; mean={settings["canonical_mean_mode"]}; '
              f'noise={mode}; noise observations={noise["observations"]}', flush=True)

    modifications = manifest['modifications']
    if not modifications:
        raise ValueError('at least one modification is required')
    modified_paths = {path(s['prototype']) for s in modifications}
    canonical_paths = {path(v) for s in canonical.values() for key in ('prototypes', 'noise_prototypes') for v in s.get(key, [])}
    canonical_paths.update(path(s['control_prototype']) for s in modifications)
    if modified_paths & canonical_paths:
        raise ValueError('modified sources cannot also be canonical/noise/control sources')
    for spec in modifications:
        if set(spec) - ({'label', 'canonical', 'prototype', 'control_prototype', 'footprint_pairs'} | set(EFFECT_OVERRIDES)):
            raise ValueError('unknown modification manifest options')
        mod_settings = resolve_options({**settings, **{key: spec[key] for key in EFFECT_OVERRIDES if key in spec}})
        base = spec['canonical'].replace('U', 'T')
        if base not in families:
            raise ValueError(f'invalid modification canonical base: {base}')
        mod, ctrl = prototype(spec['prototype']), prototype(spec['control_prototype'])
        if mod['metadata']['kmer_size'] != k or ctrl['metadata']['kmer_size'] != k:
            raise ValueError('matched effect prototypes must use the configured effect k-mer size')
        ctrl_index = {kid: idx for idx, kid in enumerate(ctrl['kmer_ids'].tolist())}
        matched = [(i, ctrl_index[kid], kid) for i, kid in enumerate(mod['kmer_ids'].tolist())
                   if kid in ctrl_index and kid // 4 ** (k // 2) % 4 == e.BASES.index(base)
                   and min(int(mod['counts'][i]), int(ctrl['counts'][ctrl_index[kid]])) >= min_count]
        if not matched:
            raise ValueError(f'{spec["label"]}: no matched control contexts')
        ids = torch.tensor([kid for _, _, kid in matched])
        difference = torch.stack([mod['mean_embeddings'][i] - ctrl['mean_embeddings'][j] for i, j, _ in matched])
        paired_counts = torch.tensor([[int(mod['counts'][i]), int(ctrl['counts'][j])] for i, j, _ in matched])
        center_effect = e.fit_center_effect(ids, difference, paired_counts, k, mode=mod_settings['effect_mode'],
                                            ridge=mod_settings['effect_ridge'], shrinkage=mod_settings['effect_shrinkage'],
                                            mix_weight=mod_settings['effect_mix_weight'], unseen=mod_settings['effect_unseen'])
        footprint = None
        pairs = spec.get('footprint_pairs', [])
        if any(set(pair) != {'modified', 'control', 'positions'} for pair in pairs):
            raise ValueError('each footprint pair requires exactly modified, control, and positions')
        pair_keys = [tuple(path(pair[key]) for key in ('modified', 'control', 'positions')) for pair in pairs]
        if len(set(pair_keys)) != len(pair_keys):
            raise ValueError('duplicate footprint pair')
        if radius and not pairs:
            raise ValueError(f'{spec["label"]}: neighbor scoring requires matched footprint_pairs')
        contexts, positions, strands, events, differences, weights = [], [], [], [], [], []
        for pair in pairs if radius else []:
            statistics = []
            for name in ('modified', 'control'):
                source = path(pair[name])
                p, index = load_statistics(source, checkpoint_sha256=fingerprint, embedding_key=representation[0],
                                            dim=representation[1], k=k, radius=radius)
                if p['metadata'].get('provenance_status') == 'legacy_path_size_assumption':
                    if not manifest.get('allow_unverified_statistics', False):
                        raise ValueError('converted pilot statistics require allow_unverified_statistics acknowledgement')
                    unverified_statistics.append(source)
                sources[source] = dict(kind='neighborhood', metadata=p['metadata'])
                statistics.append((p, index))
            if path(pair['modified']) == path(pair['control']):
                raise ValueError('footprint modified and control sources must differ')
            (p, pi), (q, qi) = statistics
            if not p['metadata'].get('qc') or p['metadata']['qc'] != q['metadata'].get('qc'):
                raise ValueError('paired footprint statistics must have identical recorded QC settings')
            event_path = path(pair['positions'])
            event_map = read_events(event_path)
            sources[event_path] = dict(kind='events', sha256=e.checkpoint_digest(event_path))
            central_events = set()
            for key in sorted(pi.keys() & qi.keys()):
                chrom, pos, strand = key
                i, j = pi[key], qi[key]
                if p['contexts'][i] != q['contexts'][j]:
                    raise ValueError('paired footprint references/contexts differ')
                if min(int(p['counts'][i]), int(q['counts'][j])) < min_count:
                    continue
                context = p['contexts'][i]
                these_events = sorted(event_map.get((chrom, strand), ()))
                for event in these_events:
                    offset = (pos - event) * (1 if strand == 0 else -1)
                    if abs(offset) <= radius and context[len(context) // 2 - offset] != base:
                        raise ValueError('BED neighbor event does not match modification canonical base')
                if pos in these_events:
                    if context[len(context) // 2] != base:
                        raise ValueError('BED event center does not match modification canonical base')
                    central_events.add((chrom, strand, pos))
                contexts.append(context)
                positions.append(pos)
                strands.append(strand)
                events.append(these_events)
                differences.append(p['means'][i] - q['means'][j])
                # Harmonic effective sample size for a difference of means.
                weights.append(1 / (1 / int(p['counts'][i]) + 1 / int(q['counts'][j])))
            if not central_events:
                raise ValueError('no matched, adequately covered BED event centers in footprint pair')
        if radius:
            footprint = e.fit_footprint(contexts, positions, strands, events, torch.stack(differences),
                                        torch.tensor(weights), center_effect, k, radius,
                                        settings['footprint_ridge'], settings['footprint_shrinkage'])
        labels.append(spec['label'])
        families[base]['modifications'].append(dict(**center_effect, class_index=len(labels) - 1,
                                                    footprint=footprint, matched_contexts=len(ids), fit_settings=mod_settings))
        print(f'Effect {spec["label"]}: {len(ids)} paired contexts; mode={mod_settings["effect_mode"]}; '
              f'global shrinkage={mod_settings["effect_shrinkage"]}; footprint rows={len(contexts)}', flush=True)
    model = dict(format_version=e.FORMAT_VERSION, artifact_type=e.ARTIFACT_TYPE, kmer_size=k,
                 embedding_dim=representation[1], embedding_key=representation[0], embedding_transform='identity',
                 footprint_radius=radius, modification_penalty=manifest.get('modification_penalty', 4.0),
                 class_labels=labels, families=families,
                 metadata=dict(checkpoint_sha256=fingerprint, backbone_path=backbone_path,
                               fit_settings=settings,
                               manifest=manifest, manifest_root=str(root.resolve()), sources=sources,
                               unverified_prototypes=unverified, unverified_statistics=unverified_statistics,
                               calibration='none'))
    preprocessing = validate_artifact_preprocessing(sources)
    if preprocessing is not None:
        model["metadata"]["preprocessing"] = preprocessing.to_dict()
    e.validate_effect_model(model)
    return model


def parse_args(argv=None):
    """Parse fit inputs and explicit recipe overrides without loading tensors."""
    from porefm.common.run_outputs import add_output_safety_argument
    parser = PoreFMArgumentParser(description=__doc__)
    add_output_safety_argument(parser)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=4, help='CPU tensor threads.')
    add_fit_options(parser)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    overrides = {key: getattr(args, key) for key in DEFAULTS if getattr(args, key) is not None}
    try:
        resolve_options(overrides)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    """Fit an artifact, with CLI globals overriding manifest global settings.

    Per-modification settings remain more specific than globals. Changing the
    center recipe refits its footprints; it is not an inference-only switch.
    All resolved settings are recorded in the fitted artifact.
    """
    from porefm.common.run_outputs import prepare_run_outputs
    args = parse_args(argv)
    prepare_run_outputs(args, command="prototype fit-effect-prototypes", files={"output": args.output})
    import torch
    torch.set_num_threads(args.threads)
    target = Path(args.output)
    source = Path(args.manifest).resolve()
    with source.open() as handle:
        manifest = json.load(handle)
    manifest.update({key: getattr(args, key) for key in DEFAULTS if getattr(args, key) is not None})
    model = fit_manifest(manifest, source.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, target)
    print(f'Wrote {target}; conditional scores are not calibrated probabilities', flush=True)


if __name__ == '__main__':
    main()
