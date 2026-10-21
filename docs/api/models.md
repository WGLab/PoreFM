# Models API

Use `load_model` to load a local backbone, pretraining, or classifier checkpoint
in Python. For complete analysis workflows and automatic model downloads, use
the [analysis guides](../index.md#choose-an-analysis).

```python
from porefm.models import load_model

model = load_model("backbone.pt", model_type="backbone", device="cpu")
print(model.preprocessing)
```

By default the model is in evaluation mode with gradients disabled. For
training, pass `for_training=True` when loading; calling `model.train()` alone
does not re-enable gradients. Datasets can use `model.preprocessing` to match
the checkpoint's signal processing; see [preprocessing](../preprocessing.md).

Single-file zero-shot and prototype bundles are loaded through their respective
CLI commands. See [model selection](../models.md#choosing-a-checkpoint).

::: porefm.models.load_model
    options:
      show_root_heading: true
      show_source: false
