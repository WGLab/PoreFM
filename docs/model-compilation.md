# Model compilation defaults

Model-executing CLI commands use
`torch.compile(model, dynamic=True)`. The first run can take longer while
kernels compile; use the opt-out below to diagnose compilation failures.

| CLI | Default | Opt-out / notes |
|---|---|---|
| `pairwise-comparison` | Compiled | `--disable-compile` |
| `prototype extract-kmer-prototypes` | Compiled | `--disable_compile`; union and model-free dry runs execute no model |
| `pretrain` | Compiled | `--disable_compile` |
| `finetune-modification` | Compiled | `--disable_compile` |
| `detect-modification` | Compiled | `--disable_compile` |
| `zero-shot train`, `train-scale`, `fit-whitening`, `calibrate` | Compiled | `--disable_compile` |
| `zero-shot pipeline` | Compiled | Passes `--disable_compile` to its model stages |
| `zero-shot detect` | Compiled | `--disable_compile`; optional `--deterministic` fixes numerically sensitive kernel choices |
| `prototype classify` | Compiled backbone | `--disable_compile` |

Commands that process stored embeddings, plots, BAM tags without a model, or
configuration files have no model to compile.

`zero-shot detect --deterministic` retains compilation and the command
precision settings, but opts into deterministic PyTorch kernels. It can change
runtime speed and numerical rounding relative to the default execution policy.
The option applies to detection. See the
[detection guide](canonical-null-inference.md#optional-deterministic-execution)
for reproducibility limits and calibration considerations.
