"""Encode conditional residual-effect scores as source-read MM/ML BAM tags."""

from __future__ import annotations

import array
import math
from pathlib import Path

import numpy as np
import pysam

from porefm.data.bam import revcomp
from porefm.data.bam_headers import add_pg_to_header
from porefm.inference.predictions import make_mm_ml_tags


# RNA codes follow Dorado, including T (not U) for uridine modifications.
# Accept the alternative methyl naming used by existing prototype artifacts.
RNA_SAM_CODES = {
    'm5c': ('C', 'm'), '5mc': ('C', 'm'),
    'm6a': ('A', 'a'), '6ma': ('A', 'a'),
    'inosine': ('A', '17596'),
    'pseu': ('T', '17802'),
    '2omec': ('C', '19228'),
    '2omea': ('A', '69426'),
    '2omeg': ('G', '19229'),
    '2omeu': ('T', '19227'),
}
BAM_TAGS_TO_DROP = ('MM', 'ML', 'MN', 'Mm', 'Ml', 'SG', 'mv')


def bam_families(model):
    """Resolve artifact class indices to canonical-first families and SAM codes.

    Returns ``(base, indices, codes)`` tuples in A/C/G/T order. Names are
    case-insensitive. Unknown labels, wrong canonical families, and duplicate
    codes within a family fail before inference rather than losing classes.
    """
    families = []
    for base in 'ACGT':
        family = model['families'][base]
        indices = [family['canonical_class_index']]
        codes = []
        for modification in family['modifications']:
            index = modification['class_index']
            label = model['class_labels'][index]
            mapping = RNA_SAM_CODES.get(label.lower())
            if mapping is None:
                raise ValueError(f'No RNA SAM code is defined for effect class {label!r}')
            if mapping[0] != base:
                raise ValueError(f'RNA SAM code for {label!r} requires family {mapping[0]}, not {base}')
            if mapping[1] in codes:
                raise ValueError(f'Duplicate RNA SAM code {base}+{mapping[1]} in effect classes')
            indices.append(index)
            codes.append(mapping[1])
        families.append((base, indices, codes))
    return families


def effect_bam_header(paths):
    """Union source BAM headers by reference name and RG/PG ID.

    Records are reconstructed by reference name, so input SQ order may differ.
    Conflicting definitions fail instead of mislabelling references/read groups.
    Output is explicitly unsorted because loader workers may interleave reads.
    Unmapped-only BAM headers without SQ entries are supported.
    """
    merged = {'HD': {'VN': '1.6', 'SO': 'unsorted'}, 'CO': []}
    entries = {key: {} for key in ('SQ', 'RG', 'PG')}
    for path in paths:
        with pysam.AlignmentFile(str(path), 'rb', check_sq=False) as source:
            header = source.header.to_dict()
        for kind, records in entries.items():
            key = 'SN' if kind == 'SQ' else 'ID'
            for record in header.get(kind, []):
                name = record[key]
                if name in records and records[name] != record:
                    raise ValueError(f'Conflicting BAM {kind} definition for {name!r} in {path}')
                records[name] = record
        for comment in header.get('CO', []):
            if comment not in merged['CO']:
                merged['CO'].append(comment)
    merged.update({kind: list(records.values()) for kind, records in entries.items() if records})
    merged['CO'].append('PoreFM residual-effects: ML contains uncalibrated conditional softmax '
                        'probabilities given inferred neighbors; skipped MM sites are unknown.')
    return add_pg_to_header(merged, program_name='porefm', program_id='porefm-residual-effects')


def effect_mm_ml_tags(read, result, selected, families, *, seq_type, temperature=1.0):
    """Convert selected, callable family scores into MM and group-major ML bytes.

    ``result['scores']`` has shape (L, global classes); ``selected`` is (L,).
    Each canonical-plus-modification family is normalized independently with
    softmax(scores / temperature). Canonical mass participates in normalization
    but is implicit in BAM. All modeled modifications are emitted, including
    low-probability alternatives and probabilities at canonical-winning sites.

    Coordinates use the full source read's as-sequenced orientation, including
    clipped/unobserved bases when counting MM deltas. No-call/unselected sites
    are omitted with the '?' skipped-site flag. Reverse alignments retain '+'
    in MM, as required by SAM. This does not marginalize neighboring states.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('effect probability temperature must be finite and positive')
    record = read['pysam_record']
    sequence = record['seq'].upper().replace('U', 'T')
    if int(record['flag']) & 16:
        sequence = revcomp(sequence)
    scores = np.asarray(result['scores'], dtype=np.float64)
    predicted = np.asarray(result['predicted_class_index'])
    selected = np.asarray(selected, dtype=bool)
    length = len(read['sequence'])
    if scores.ndim != 2 or len(scores) != length or selected.shape != (length,) or predicted.shape != (length,):
        raise ValueError('effect BAM scores/selection do not match stitched read length')
    bases = np.frombuffer(read['sequence'].upper().replace('U', 'T').encode('ascii'), dtype=np.uint8)
    source_bases = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
    mm_parts, ml_all = [], []
    for base, indices, codes in families:
        if not codes:
            continue
        positions = np.flatnonzero(selected & (predicted >= 0) & (bases == ord(base)))
        local = scores[np.ix_(positions, indices)]
        finite = np.isfinite(local).all(axis=1)
        positions, local = positions[finite], local[finite]
        if not len(positions):
            continue
        if positions[-1] >= len(sequence) or not np.all(source_bases[positions] == ord(base)):
            raise ValueError(f'Effect BAM coordinates disagree with source sequence for {read["readname"]!r}')
        # Shift before dividing: avoid overflow and retain relative evidence.
        with np.errstate(over='ignore', under='ignore'):
            weights = np.exp((local - local.max(axis=1, keepdims=True)) / temperature)
        probabilities = weights / weights.sum(axis=1, keepdims=True)
        mm, ml = make_mm_ml_tags(sequence, base, positions, probabilities, seq_type,
                                 mod_codes=codes, strand='+', skipped_flag='?')
        mm_parts.append(mm)
        ml_all.extend(ml)
    return ''.join(mm_parts), ml_all


class EffectBamWriter:
    """Stream source records with replacement MM/ML/MN tags to an output BAM.

    Writes one record per processed read, even when it has no callable selected
    sites. Original sequence, alignment, qualities and unrelated tags survive;
    old modification tags and large signal/move tags are removed, matching
    supervised output. Successful context exit publishes ``output.bam``;
    failure leaves only the partial file from this attempt. Existing paths
    require ``force_overwrite``. No sort or index is performed.
    """

    def __init__(self, output_dir, header, families, *, seq_type, temperature=1.0,
                 threads=4, force_overwrite=False):
        if not math.isfinite(temperature) or temperature <= 0 or threads < 1:
            raise ValueError('invalid effect BAM temperature or thread count')
        self.path = Path(output_dir) / 'output.bam'
        self.partial = self.path.with_name('output.bam.partial')
        for path in (self.path, self.partial):
            if path.is_symlink() or (path.exists() and not force_overwrite):
                raise FileExistsError(f'refusing to overwrite {path}')
        self.header, self.families = header, families
        self.seq_type, self.temperature, self.threads = seq_type, temperature, threads
        self.total_reads = 0
        self.tagged_reads = 0

    def __enter__(self):
        self.handle = pysam.AlignmentFile(str(self.partial), 'wb', header=self.header, threads=self.threads)
        return self

    def append(self, read, result, selected):
        """Write one stitched read using its retained source SAM dictionary."""
        mm, ml = effect_mm_ml_tags(read, result, selected, self.families,
                                   seq_type=self.seq_type, temperature=self.temperature)
        record = dict(read['pysam_record'])
        record['tags'] = [tag for tag in record.get('tags', []) if tag.split(':', 1)[0] not in BAM_TAGS_TO_DROP]
        segment = pysam.AlignedSegment.from_dict(record, self.header)
        if ml:
            segment.set_tag('MM', mm, value_type='Z')
            segment.set_tag('ML', array.array('B', ml))
            segment.set_tag('MN', len(segment.query_sequence))
            self.tagged_reads += 1
        self.handle.write(segment)
        self.total_reads += 1

    def __exit__(self, exc_type, exc, traceback):
        self.handle.close()
        if exc_type is None:
            self.partial.replace(self.path)
