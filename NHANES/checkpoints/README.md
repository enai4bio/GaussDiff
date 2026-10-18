# NHANES pretrained checkpoint

This directory contains the trained GaussDiff denoising network used for the
NHANES CVD results reported in the paper.

| File | Description |
|------|-------------|
| `diffusion.pt` | Denoising MLP weights, saved at the best-validation step (`step_009031`). 1.3 MB. |
| `train_config.toml` | The configuration recorded by the training run that produced `diffusion.pt`. |
| `load_checkpoint.py` | Helper that loads `diffusion.pt` (handles the legacy module path, see below). |

## Checkpoint details

- Architecture: `d_layers = [256, 256, 256]`, `dim_t = 128`
- Diffusion: `n_timesteps = 1000`, cosine scheduler, `parametrization = "x0"`
- Training: 11000 steps, `init_lr = 0.00013`, full-batch (2954), seed 0
- Selected step: 9031 (lowest validation loss)
- MD5: `e6572c7bcee1f0bdf5da1e6c8751c934`

## Loading

The checkpoint is a pickled `torch.nn.Module` (saved with `torch.save` on the
module itself, not a `state_dict`), and the pickle records the module path the
network had at training time, `tab_ddpm.modules`. That path was later renamed
to `nhanes_pipeline.models.tabular_diffusion.modules`, so a plain
`torch.load` fails with:

```
ModuleNotFoundError: No module named 'tab_ddpm'
```

Use the helper in this directory, which registers the alias and then loads:

```python
from load_checkpoint import load_denoise_mlp

denoise_mlp = load_denoise_mlp(device="cuda:0")
```

Or, equivalently, by hand:

```python
import sys, types, torch
sys.path.insert(0, "NHANES/src")
from nhanes_pipeline.models.tabular_diffusion import modules
pkg = types.ModuleType("tab_ddpm"); pkg.modules = modules
sys.modules["tab_ddpm"] = pkg
sys.modules["tab_ddpm.modules"] = modules

denoise_mlp = torch.load(
    "NHANES/checkpoints/diffusion.pt", weights_only=False, map_location="cuda:0"
)
```

The class definition itself is unchanged, so the weights load exactly as
saved. Verified: loads to `MLPDiffusion` with 342,040 parameters and produces
a correct forward pass.

To sample from it, wrap the loaded network in `GaussianMultinomialDiffusion`
as `src/nhanes_pipeline/cli.py` does, or point the `best_model_path` key of
your run configuration at this file.

## Notes on `train_config.toml`

This file is kept as a record of the original run, so its values are
reproduced as they were written rather than rewritten to match the current
repository layout. Two differences are worth knowing:

- Absolute paths from the original machine have been replaced with relative
  ones; they are provenance, not working paths.
- `label_column` reads `CVDs`, whereas `configs/nhanes.toml` in this
  repository uses `CVD`. This is a naming change made after training; it
  refers to the same outcome variable and does not affect the weights.

For running the pipeline, use `configs/nhanes.toml`, not this file.
