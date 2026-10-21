"""Batched device-side effect scores with compact CPU decoder packets.

The backbone and its forward contract are not involved here. All statistical
operations use the fitted canonical means, receiving-position whitening, and
additive modification footprints. Coordinate/context bookkeeping stays on CPU;
embedding arithmetic, template whitening, and overlap products run on device.
"""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.nn import functional as F

from .effects import BASES


def _sequence_tokens(sequence):
    """Encode normalized ASCII bases once; ambiguous symbols map to -1."""
    lookup = np.full(256, -1, dtype=np.int64)
    lookup[np.frombuffer(b'ACGT', dtype=np.uint8)] = np.arange(4)
    return lookup[np.frombuffer(sequence.encode('ascii', errors='replace'), dtype=np.uint8)]


def _context_ids(tokens, k):
    """Return centered two-bit IDs, with -1 at incomplete/ambiguous windows."""
    result = np.full(len(tokens), -1, dtype=np.int64)
    if len(tokens) >= k:
        windows = np.lib.stride_tricks.sliding_window_view(tokens, k)
        ids = (np.maximum(windows, 0) * (4 ** np.arange(k-1, -1, -1))).sum(1)
        ids[(windows < 0).any(1)] = -1
        result[k//2:len(tokens)-k//2] = ids
    return result


@dataclass
class EffectScorePacket:
    """Owned CPU scores for one complete read, with no embeddings/device refs.

    Shapes are unaries (L,S), pairs (L,2R,S,S), classes (L,S), counts (L,), and
    canonical_energy (L,). S is the model's maximum states per base, including
    canonical. ``pairs[i,d-1]`` couples i to i-d. Uncallable positions have one
    placeholder state with class -1. Padding is never decoded as a real class.
    """

    unaries: torch.Tensor
    pairs: torch.Tensor
    classes: torch.Tensor
    counts: torch.Tensor
    canonical_energy: torch.Tensor
    radius: int

    @property
    def nbytes(self):
        """Tensor payload bytes; excludes Python objects and read metadata."""
        return sum(t.numel() * t.element_size() for t in
                   (self.unaries, self.pairs, self.classes, self.counts, self.canonical_energy))

    def potentials(self):
        """Expose CPU views in the decoder contract, preserving pair order."""
        sizes = self.counts.tolist()
        called = self.classes[:, 0].numpy() >= 0
        return dict(
            unaries=[self.unaries[i, :n] for i, n in enumerate(sizes)],
            classes=[self.classes[i, :n].tolist() for i, n in enumerate(sizes)],
            interactions=[{j: self.pairs[i, i-j-1, :n, :sizes[j]]
                           for j in range(max(0, i-2*self.radius), i) if called[j]}
                          if called[i] else {} for i, n in enumerate(sizes)],
            radius=self.radius, canonical_energy=self.canonical_energy)


@contextmanager
def _full_precision(device):
    """Use FP32 scoring without TF32, restoring the caller's matmul setting.

    Only downstream scoring executes inside this scope, on the producer
    thread. No backbone invocation occurs while the setting is overridden.
    """
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        if device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        if device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = previous


class DeviceEffectScorer:
    """Keep fitted tables on device and score independent reads in bounded tiles.

    Parameters
    ----------
    model : dict
        Validated, immutable CPU effect artifact. Only scoring tables are copied.
    device : str or torch.device
        Usually CUDA; CPU is supported for deterministic unit comparisons.
    cache_bytes : int
        Maximum retained raw-effect tensor bytes. Entries include every local
        modification, offset and receiving base for a source context. The LRU
        cache is populated lazily; zero disables retention. Model tables and
        working tiles are separate from this budget.
    block_size : int
        Maximum centers per scoring tile, plus a 2R left halo for overlap terms.
        Input embeddings and compact output scores remain resident until the
        read/group is prepared. No read is split into independent decoding jobs.

    GPU and CPU reductions may round differently. The objective, missing-
    context policy, and class ordering are preserved. Recreate this scorer if
    fitted parameters change. It is owned by a single producer thread.
    """

    def __init__(self, model, device, *, cache_bytes=256 * 1024 ** 2, block_size=4096):
        if not isinstance(cache_bytes, int) or cache_bytes < 0 or block_size < 1:
            raise ValueError('device effect cache budget must be nonnegative and block size positive')
        self.model = model
        self.device = torch.device(device)
        self.block_size = block_size
        self.dim = model['embedding_dim']
        self.k = model['kmer_size']
        self.radius = model['footprint_radius']
        self.max_mods = max(1, max(len(f['modifications']) for f in model['families'].values()))
        self.states = self.max_mods + 1
        self.families = {}
        transforms = []
        self.noise_lookup = np.full(1024, -1, dtype=np.int64)
        self.base_noise = {}
        for base, family in model['families'].items():
            copied = dict(family)
            for key in ('canonical_means', 'canonical_counts', 'canonical_coefficients'):
                if copied[key] is not None:
                    copied[key] = copied[key].to(self.device)
            copied['modifications'] = []
            for mod in family['modifications']:
                item = dict(mod)
                for key in ('coefficients', 'empirical_effects', 'global_effect'):
                    if item[key] is not None:
                        item[key] = item[key].to(self.device)
                item['footprint'] = (dict(coefficients=mod['footprint']['coefficients'].to(self.device))
                                     if mod.get('footprint') is not None else None)
                copied['modifications'].append(item)
            self.families[base] = copied
            self.base_noise[base] = len(transforms)
            ids = np.arange(1024)
            self.noise_lookup[ids // 16 % 4 == BASES.index(base)] = len(transforms)
            transforms.append(family['whitening'])
            if family['noise_mode'] == 'hierarchical-5mer':
                for kid, matrix in zip(family['noise_context_ids'].tolist(), family['noise_context_whitening']):
                    self.noise_lookup[kid] = len(transforms)
                    transforms.append(matrix)
        self.whiteners = torch.stack(transforms).to(self.device)
        self.entry_shape = (self.max_mods, 2*self.radius+1, 4, self.dim)
        entry_bytes = math.prod(self.entry_shape) * 4
        self.capacity = min(4 ** self.k, cache_bytes // entry_bytes)
        self.cache_tensor_bytes = self.capacity * entry_bytes
        self._bank = None
        self._slots = OrderedDict()

    def _index(self, values):
        return torch.as_tensor(values, dtype=torch.long, device=self.device)

    def _features(self, ids, k):
        """Build exact float32 one-hot context features without CPU tensor dispatch."""
        offsets = np.arange(k-1, -1, -1)
        offsets = offsets[offsets != k//2]
        flank = ids[:, None] // (4 ** offsets) % 4
        features = np.zeros((len(ids), 1+4*(k-1)), dtype=np.float32)
        features[:, 0] = 1
        features[np.arange(len(ids))[:, None], 1+4*np.arange(k-1)+flank] = 1
        return torch.from_numpy(features).to(self.device)

    def _canonical(self, ids, family):
        """Evaluate the existing mean recipe; IDs and support lookup are CPU."""
        keys = family['canonical_ids'].numpy()
        ix = np.searchsorted(keys, ids)
        safe = np.minimum(ix, len(keys)-1)
        known = (ix < len(keys)) & (keys[safe] == ids)
        mode = family['canonical_mean_mode']
        prediction = torch.full((len(ids), self.dim), float('nan'), device=self.device)
        use_regression = np.ones(len(ids), dtype=bool) if mode != 'empirical' else ~known
        if family['canonical_unseen'] == 'no-call' and mode == 'empirical':
            use_regression[:] = False
        if use_regression.any() and family['canonical_coefficients'] is not None:
            rows = np.flatnonzero(use_regression)
            features = self._features(ids[rows], family['context_k'])
            prediction[self._index(rows)] = features @ family['canonical_coefficients']
        if mode == 'regression':
            return prediction
        rows = np.flatnonzero(known)
        target = self._index(rows)
        matched = self._index(safe[rows])
        if mode == 'empirical':
            prediction[target] = family['canonical_means'][matched]
        else:
            count = family['canonical_counts'][matched].float()
            weight = count / (count + family['canonical_strength'])
            prediction[target] = (weight[:, None] * family['canonical_means'][matched]
                                  + (1-weight[:, None]) * prediction[target])
        if family['canonical_unseen'] == 'no-call':
            prediction[self._index(np.flatnonzero(~known))] = float('nan')
        return prediction

    def _center(self, ids, features, mod):
        """Device evaluation of regression/empirical/mixed center effects."""
        mode = mod['effect_mode']
        prediction = (features @ mod['coefficients'] if mode != 'empirical'
                      else mod['global_effect'].expand(len(ids), -1).clone())
        if mode == 'regression':
            return prediction
        keys = mod['empirical_ids'].numpy()
        ix = np.searchsorted(keys, ids)
        safe = np.minimum(ix, len(keys)-1)
        known = (ix < len(keys)) & (keys[safe] == ids)
        rows = np.flatnonzero(known)
        target = self._index(rows)
        direct = mod['empirical_effects'][self._index(safe[rows])]
        if mode == 'mixed':
            weight = mod['effect_mix_weight']
            prediction[target] = weight * direct + (1-weight) * prediction[target]
        else:
            alpha = mod['effect_shrinkage']
            prediction[target] = (1-alpha) * direct + alpha * mod['global_effect']
            if mod['effect_unseen'] == 'no-call':
                prediction[self._index(np.flatnonzero(~known))] = float('nan')
        return prediction

    def _source_support(self, unique_ids):
        """Check finite center hypotheses without retaining per-read vectors."""
        supported = torch.ones(len(unique_ids), dtype=torch.bool, device=self.device)
        for start in range(0, len(unique_ids), self.block_size):
            ids = unique_ids[start:start+self.block_size]
            for base, family in self.families.items():
                rows = np.flatnonzero(ids // 4 ** (self.k//2) % 4 == BASES.index(base))
                if not len(rows) or not family['modifications']:
                    continue
                features = self._features(ids[rows], self.k)
                for mod in family['modifications']:
                    supported[self._index(rows+start)] &= torch.isfinite(self._center(ids[rows], features, mod)).all(1)
        return supported

    def _raw_effects(self, ids):
        """Compute source-context tables in batches; no receiving whitening."""
        result = torch.zeros((len(ids), *self.entry_shape), device=self.device)
        for base, family in self.families.items():
            rows = np.flatnonzero(ids // 4 ** (self.k//2) % 4 == BASES.index(base))
            if not len(rows) or not family['modifications']:
                continue
            features = self._features(ids[rows], self.k)
            target = self._index(rows)
            for m, mod in enumerate(family['modifications']):
                if self.radius:
                    coef = mod['footprint']['coefficients']
                    dependent = features[:, 1:] @ coef[:, 4:].permute(1, 0, 2).reshape(
                        features.shape[1]-1, (2*self.radius+1)*self.dim)
                    effect = dependent.reshape(len(rows), 2*self.radius+1, self.dim)[:, :, None] + coef[None, :, :4]
                    result[target, m] = effect
                result[target, m, self.radius] = self._center(ids[rows], features, mod)[:, None, :]
        return result

    def _raw_table(self, source_ids):
        """Return a table and slot map; a tile larger than the cache bypasses it."""
        unique, inverse = np.unique(source_ids, return_inverse=True)
        if len(unique) > self.capacity:
            return self._raw_effects(unique), inverse
        if self._bank is None:
            self._bank = torch.empty((self.capacity, *self.entry_shape), device=self.device)
        requested = set(unique.tolist())
        missing = [kid for kid in unique.tolist() if kid not in self._slots]
        for kid in unique.tolist():
            if kid in self._slots:
                self._slots.move_to_end(kid)
        new_slots = []
        for kid in missing:
            if len(self._slots) < self.capacity:
                slot = len(self._slots)
            else:
                # Protect every source requested by this tile until it is read.
                victim = next(key for key in self._slots if key not in requested)
                slot = self._slots.pop(victim)
            self._slots[kid] = slot
            new_slots.append(slot)
        if missing:
            values = self._raw_effects(np.asarray(missing, dtype=np.int64))
            self._bank.index_copy_(0, self._index(new_slots), values)
        slots = np.asarray([self._slots[kid] for kid in unique.tolist()], dtype=np.int64)
        return self._bank, slots[inverse]

    def _whiten(self, vectors, noise_ids):
        """Group matrix products by metric and bucket by row count.

        Each bucket pads groups to a power-of-two row count (under 2x total row
        padding). At most 32 distinct D-by-D matrices are gathered at once;
        matrices are never expanded to one copy per observation/effect vector.
        """
        result = torch.zeros_like(vectors)
        positions = np.flatnonzero(noise_ids >= 0)
        if not len(positions):
            return result
        order = positions[np.argsort(noise_ids[positions], kind='stable')]
        keys, starts, counts = np.unique(noise_ids[order], return_index=True, return_counts=True)
        buckets = {}
        for key, start, count in zip(keys.tolist(), starts.tolist(), counts.tolist()):
            width = 1 << (count-1).bit_length()
            buckets.setdefault(width, []).append((key, order[start:start+count]))
        for width, groups in buckets.items():
            for offset in range(0, len(groups), 32):
                subset = groups[offset:offset+32]
                indices = np.zeros((len(subset), width), dtype=np.int64)
                keep = np.zeros_like(indices, dtype=bool)
                for i, (_, rows) in enumerate(subset):
                    indices[i, :len(rows)] = rows
                    keep[i, :len(rows)] = True
                idx = self._index(indices)
                transformed = torch.bmm(vectors[idx], self.whiteners[self._index([g[0] for g in subset])])
                # Padding locations are already known on CPU. Integer gathers
                # avoid CUDA nonzero/host synchronization from boolean indexing.
                take = np.flatnonzero(keep.reshape(-1))
                target = indices.reshape(-1)[take]
                result.index_copy_(0, self._index(target),
                                   transformed.flatten(0, 1).index_select(0, self._index(take)))
        return result

    def prepare(self, sequence, embeddings, valid, *, neighbors=True, penalty=None):
        """Return compact, ready-to-read CPU scores for a complete read.

        Embeddings are (L,D) on this scorer's device; a CPU input is accepted
        and copied for standalone validation. The only device-to-host transfers
        are unary/pair scores, callability, and per-base canonical energy.
        """
        with _full_precision(self.device):
            return self._prepare(sequence, embeddings, valid, neighbors, penalty)

    def prepare_many(self, reads, *, neighbors=True, penalty=None):
        """Score a bounded group jointly and return one owned CPU packet per read.

        Each input mapping contains ``sequence``, ``embeddings[L,D]`` and
        ``valid[L]``. Invalid N guards isolate every context, receiving footprint
        and 2R interaction across read boundaries. Reads are decoded separately;
        no artificial boundary is introduced inside a read. The caller bounds
        total read bases retained on device. Dense working tensors remain tiled
        by ``block_size``. Empty reads are supported and retain input ordering.
        """
        if not reads:
            return []
        if len(reads) == 1:
            read = reads[0]
            return [self.prepare(read['sequence'], read['embeddings'], read['valid'],
                                 neighbors=neighbors, penalty=penalty)]
        guard = max(self.k, 5, 2*self.radius+1,
                    *(family['context_k'] for family in self.families.values()))
        lengths = [len(read['sequence']) for read in reads]
        for read, length in zip(reads, lengths):
            if read['embeddings'].shape != (length, self.dim) or read['valid'].shape != (length,):
                raise ValueError('read embedding/validity dimensions do not match sequence')
        starts = np.cumsum([0] + [length+guard for length in lengths[:-1]]).tolist()
        total = sum(lengths) + guard*(len(reads)-1)
        h = torch.zeros(total, self.dim, device=self.device)
        valid = torch.zeros(total, dtype=torch.bool, device=self.device)
        for read, start, length in zip(reads, starts, lengths):
            h[start:start+length] = read['embeddings'].to(self.device)
            valid[start:start+length] = read['valid'].to(self.device)
        sequence = ('N'*guard).join(read['sequence'] for read in reads)
        packet = self.prepare(sequence, h, valid, neighbors=neighbors, penalty=penalty)
        # Clone the small CPU slices so queue byte accounting covers owned
        # storage, not views retaining the entire grouped packet and its guards.
        fields = ('unaries', 'pairs', 'classes', 'counts', 'canonical_energy')
        return [EffectScorePacket(*(getattr(packet, key)[start:start+length].clone() for key in fields),
                                  packet.radius) for start, length in zip(starts, lengths)]

    def _prepare(self, sequence, embeddings, valid, neighbors, penalty):
        seq = sequence.upper().replace('U', 'T')
        length = len(seq)
        h = torch.as_tensor(embeddings, dtype=torch.float32, device=self.device)
        valid = torch.as_tensor(valid, dtype=torch.bool, device=self.device)
        if h.shape != (length, self.dim) or valid.shape != (length,):
            raise ValueError('read embedding/validity dimensions do not match sequence')
        cost = self.model['modification_penalty'] if penalty is None else penalty
        if not math.isfinite(cost) or cost < 0:
            raise ValueError('modification penalty must be finite and nonnegative')
        radius = self.radius if neighbors else 0
        unaries = torch.zeros(length, self.states, device=self.device)
        pairs = torch.zeros(length, 2*radius, self.states, self.states, device=self.device)
        classes = torch.full((length, self.states), -1, dtype=torch.long)
        counts = torch.ones(length, dtype=torch.long)
        if not length:
            return EffectScorePacket(unaries.cpu(), pairs.cpu(), classes, counts, torch.empty(0), radius)
        bases = _sequence_tokens(seq)
        contexts = {k: _context_ids(bases, k) for k in
                    {self.k, 5, *(family['context_k'] for family in self.families.values())}}
        source_ids = contexts[self.k]
        receiver_ids = contexts[5]
        noise_ids = np.full(length, -1, dtype=np.int64)
        mean = torch.zeros_like(h)
        complete_mean = np.zeros(length, dtype=bool)
        for base, family in self.families.items():
            positions = np.flatnonzero(bases == BASES.index(base))
            ids = contexts[family['context_k']]
            available = positions[ids[positions] >= 0]
            complete_mean[available] = True
            mean[self._index(available)] = self._canonical(ids[available], family)
            if family['noise_mode'] == 'hierarchical-5mer':
                eligible = positions[receiver_ids[positions] >= 0]
                noise_ids[eligible] = self.noise_lookup[receiver_ids[eligible]]
            else:
                noise_ids[positions] = self.base_noise[base]
            labels = [family['canonical_class_index']] + [m['class_index'] for m in family['modifications']]
            classes[positions, :len(labels)] = torch.tensor(labels)
            counts[positions] = len(labels)
        usable = (valid & torch.isfinite(h).all(1) & torch.isfinite(mean).all(1)
                  & torch.as_tensor(complete_mean & (noise_ids >= 0), device=self.device))
        residual = torch.where(usable[:, None], h-mean, 0.)
        observed = self._whiten(residual, noise_ids)
        del residual, mean
        sources, inverse = np.unique(np.maximum(source_ids, 0), return_inverse=True)
        support = self._source_support(sources)[self._index(inverse)]
        source_complete = torch.as_tensor(source_ids >= 0, device=self.device)
        unsupported = source_complete & ~support
        affected = F.max_pool1d(unsupported.float()[None, None], 2*radius+1, stride=1, padding=radius)[0, 0].bool()
        usable &= ~affected
        observed[affected] = 0
        bad = F.max_pool1d((~usable).float()[None, None], 2*radius+1, stride=1, padding=radius)[0, 0].bool()
        positions = torch.arange(length, device=self.device)
        callable_mask = usable & source_complete & ~bad & (positions >= radius) & (positions < length-radius)
        for start in range(0, length, self.block_size):
            stop = min(length, start+self.block_size)
            begin = max(0, start-2*radius)
            table, slots = self._raw_table(np.maximum(source_ids[begin:stop], 0))
            receivers = np.arange(begin, stop)[:, None] + np.arange(-radius, radius+1)[None]
            clipped = np.clip(receivers, 0, length-1)
            receiver_bases = np.maximum(bases[clipped], 0)
            raw = table[self._index(slots)[:, None, None],
                        torch.arange(self.max_mods, device=self.device)[None, :, None],
                        torch.arange(self.radius-radius, self.radius+radius+1, device=self.device)[None, None],
                        self._index(receiver_bases)[:, None]]
            metrics = np.broadcast_to(noise_ids[clipped][:, None], raw.shape[:-1]).reshape(-1)
            white = self._whiten(raw.reshape(-1, self.dim), metrics).reshape(raw.shape)
            template = torch.zeros(stop-begin, self.states, 2*radius+1, self.dim, device=self.device)
            template[:, 1:] = torch.where(callable_mask[begin:stop, None, None, None], white, 0.)
            center = template[start-begin:]
            observations = observed[self._index(clipped[start-begin:])]
            unaries[start:stop] = (center*observations[:, None]).sum((2, 3)) - .5*center.square().sum((2, 3))
            unaries[start:stop, 1:] -= cost
            for distance in range(1, 2*radius+1):
                lo = max(start, distance)
                if lo >= stop:
                    continue
                overlap = 2*radius+1-distance
                current = template[lo-begin:stop-begin, :, :overlap].flatten(2)
                previous = template[lo-distance-begin:stop-distance-begin, :, distance:].flatten(2)
                pairs[lo:stop, distance-1] = torch.bmm(current, previous.transpose(1, 2))
        called = callable_mask.cpu()
        classes[~called] = -1
        counts[~called] = 1
        # Blocking copies complete before the packet is submitted to CPU work.
        return EffectScorePacket(unaries.cpu(), pairs.cpu(), classes, counts, observed.square().sum(1).cpu(), radius)
