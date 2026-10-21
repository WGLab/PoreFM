# Installation

Use Linux with an NVIDIA GPU supported by
[FlashAttention 2](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3#nvidia-cuda-support),
such as an A100, H100, RTX 3090, or RTX 4090. Python 3.12 is recommended.
All four model workflows require CUDA-enabled PyTorch and FlashAttention.

## Set up GPU dependencies

In your Python environment, install PyTorch before FlashAttention. This setup
uses PyTorch 2.10.0 with CUDA 12.8 and FlashAttention 2.8.3. Skip these steps
if you already have these versions working.

```bash
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

The CUDA wheel must be compatible with your NVIDIA driver; see the
[official PyTorch installation options](https://pytorch.org/get-started/previous-versions/#v2100).
Install FlashAttention after selecting PyTorch, because its compiled extension
must match that environment. `--no-build-isolation` lets its installer use the
PyTorch you just installed.

FlashAttention attempts to use a matching prebuilt wheel when available.
Plan for a source build with the **PyTorch 2.10 / CUDA 12.8** combination above,
unless a compatible cached or separately supplied wheel is available.
The upstream release includes PyTorch 2.10 wheels labelled `cu13`; those target
CUDA 13 and do not match this CUDA 12.8 setup. Compilation can take a long time. See the
[official wheel list](https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.8.3)
and the build guidance below.

## Install PoreFM

```bash
git clone https://github.com/WGLab/PoreFM.git
cd PoreFM
python -m pip install .
```

Pip installs the remaining Python dependencies, including those used to read,
summarize, and plot outputs. For an editable
installation, use `python -m pip install -e .` instead.

Input preparation also uses the external `samtools` command. Install it through
your system or cluster package manager and check that `samtools --version`
works. Model files are supplied separately.

## Check the installation

Run this on a machine with an allocated GPU:

```bash
porefm --help
python - <<'PY'
import torch
import flash_attn

print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("FlashAttention:", flash_attn.__version__)
assert torch.cuda.is_available(), "No CUDA GPU is visible to PyTorch"
print("GPU:", torch.cuda.get_device_name(0))
PY
```

## If FlashAttention needs to compile

Source builds require a CUDA toolkit with `nvcc` and a C++ compiler. Use a
toolkit matching the CUDA version of PyTorch (12.8 for the command above).
The CUDA runtime in a PyTorch wheel does not provide the `nvcc` compiler.
On a cluster, load the appropriate CUDA module for your system before building.

To limit build memory use, cap the number of compiler jobs:

```bash
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
```

Leave `FLASH_ATTENTION_FORCE_BUILD` unset for normal installation: setting it
to `TRUE` forces compilation even if a matching wheel exists. Keep pip's cache
enabled so subsequent installs can reuse a built wheel. A prebuilt wheel must
match Python, PyTorch, CUDA, the platform, and the C++ ABI; a wheel for another
PyTorch version is not interchangeable. See the
[FlashAttention installation guide](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3#installation-and-features).

Continue with the [HeLa RNA example](../examples.md) for a first result using
downloadable data, or [prepare your own inputs](../guides/inputs.md).
