# Chemistry and signal preprocessing

Checkpoint-based commands infer sequence type, pore chemistry, and expected
k-mer current from the checkpoint. Do not supply `--seq_type`, `--pore_type`,
or `--kmer_dict_path` in ordinary inference, fine-tuning, or embedding commands.
If supplied, these options assert compatibility and fail on disagreement.
Pretraining from scratch requires `--chemistry` or an explicit custom profile.

## Built-in profiles

| Chemistry | Sequence | Expected-current table |
| --- | --- | --- |
| `dna_r10.4.1_e8.2_400bps` | DNA | `dna_r10.4.1_e8.2_400bps/9mer_levels_v1.txt` |
| `rna004` | RNA | `rna004/9mer_levels_v1.txt` |
| `rna002` | RNA | `rna_r9.4_180mv_70bps/5mer_levels_v1.txt` |

These files ship with PoreFM, pinned to
[`nanoporetech/kmer_models` commit `4e56daed7fbb79b538f58e41262d5c54b07356ea`](https://github.com/nanoporetech/kmer_models/tree/4e56daed7fbb79b538f58e41262d5c54b07356ea).
Runtime use needs neither a separate checkout nor network access. Package
metadata records upstream paths, file SHA-256 hashes, and hashes of the parsed
float32 lookup values. Upstream license and copyright notices are included.

Identifiers and pore aliases are case-insensitive. `RNA002` implies RNA;
`R9` plus `seq_type="rna"` also selects RNA002. Bare `R9` is ambiguous and
does not identify a table. Bare `rna` cannot distinguish RNA002 from RNA004.
`R10` plus DNA selects the registered 400-bps DNA profile.

RNA002 uses move-table decoding. Signal-based chunk inference remains
restricted to 9-mer tables: provide BAM move tags and use `mv`, or `auto` with
moves, for RNA002. Registering a chemistry does not make a pretrained backbone
compatible with different signal widths or move strides.

## Python readers and datasets

Custom readers can use these exports without adopting PoreFM's dataset classes:

```python
from porefm.data.bam import resolve_preprocessing, build_lookup_table, get_read_info_lite

prep = resolve_preprocessing(chemistry="rna002")
table = build_lookup_table(chemistry="rna002")  # Cached, read-only float32 array.
read_fields = get_read_info_lite(bam_record, preprocessing=prep)
```

Alternatively resolve from a trusted checkpoint without constructing a model:

```python
from porefm.preprocessing import load_preprocessing
prep = load_preprocessing("model.pt")
```

All signal iterable datasets accept `preprocessing=prep` or one
`chemistry="rna004"` argument. Model loading also exposes the resolved contract:

```python
from porefm.models import load_model
from porefm.data.reader import BamIterableDataset

model = load_model("model.pt", model_type="backbone")
dataset = BamIterableDataset(
    bam_list=bam_list, size=2048, step=1792, move_downsample_factor=1,
    preprocessing=model.preprocessing,
)
```

Resolve once before the read loop. Collators that batch decoded samples need
no table or chemistry arguments. Built-in tables retain upstream
sequence orientation; BAM decoding reverses RNA signal into read orientation.

## Custom profiles and checkpoints

For an experiment with a custom table, provide its sequence type explicitly:

```python
prep = resolve_preprocessing(
    chemistry="custom", seq_type="rna", pore_type="experimental",
    kmer_dict_path="my_levels.txt",
)
```

The headerless text format is `KMER LEVEL`, using a consistent odd k and ACGT
encoding and finite current levels. `build_lookup_table(path)` returns
a writable array; profile arrays are immutable. Custom checkpoints embed their
exact table, so subsequent users do not need the original file.

When saving a custom model, include its resolved preprocessing profile and
table so it can process signal on another machine. PoreFM checkpoint helpers
use `model.preprocessing`. For a custom checkpoint writer, use
`attach_preprocessing(payload, prep)` from `porefm.preprocessing`.
