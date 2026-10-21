# Export per-base embeddings

Use PoreFM embeddings for your own exploratory analyses or downstream models.
This recipe exports representations for **two aligned populations** using the
same collector as pairwise comparison. It needs the sorted, indexed signal BAMs
and reference described in [input preparation](inputs.md#prepare-signal-bams-for-pairwise-comparison).

## Export an RNA region

```bash
porefm pairwise-comparison \
  --model WGLab/PoreFM-RNA@v0.1.0 \
  --control-bam signal/control.signal.bam \
  --test-bam signal/test.signal.bam \
  --ref reference.fa --chrom-list chr3 \
  --embedding-key contextual_base_seq \
  --store-embeddings --block-size 1000 \
  --forward-batch-size 16 \
  --output outputs/embeddings
```

Replace `chr3` with a contig in your inputs. `contextual_base_seq` is the
signal-contextualized representation at each base. Its width is 512 for the
released RNA backbone and 384 for the DNA backbone. Select a chemistry-matched
checkpoint from [Models](../models.md).

`--store-embeddings` can run alone, as above, or accompany the scoring flags in
the [pairwise guide](pairwise.md). Export alone does not produce anomaly p-values.
Eligible observations are assigned to one overlapping chunk, using the greatest
distance from its sequence edges, so a read position contributes at most once.
Input filters still apply; the export is not an unfiltered copy of BAM depth.

## Inspect a saved block

Files are written under `outputs/embeddings/embeddings/`. Each `.pt` file covers
a genomic block and stores `control`, `test`, `control_summary`, and
`test_summary`, together with the chromosome and representation name.

```python
from pathlib import Path
import torch
import pandas as pd

path = next(Path("outputs/embeddings/embeddings").glob("*.pt"))
# These locally generated blocks contain NumPy metadata as well as tensors.
block = torch.load(path, map_location="cpu", weights_only=False)
rows = block["test"]
print(block["chrom"], block["embedding_key"])
print(rows["embeddings"].shape)  # (observations, embedding_dimension)

preview = pd.DataFrame({
    "readname": rows["readnames"][:5],
    "position": rows["positions"][:5].numpy(),
    "read_pos": rows["read_positions"][:5].numpy(),
    "strand": rows["strands"][:5].numpy(),
})
print(preview)
features = rows["embeddings"][:5].float().numpy()
print(features.shape)
```

Load only trusted `.pt` files. Observation embeddings are stored as bfloat16;
convert selected rows to float32 before passing them to NumPy-based tools.
Site summaries store float32 `means`, integer `counts`, `positions`, and
`strands`. Reference positions are zero-based, strand is 0 forward/1 reverse,
and `read_positions` follow the collector's oriented read sequence.

Process one block at a time to limit memory use. Keep biological sources or
reads separate when constructing training and evaluation sets. Distances and
clusters in this representation do not themselves establish modification identity.
For loading a checkpoint in custom Python code, see the [model API](../api/models.md).
