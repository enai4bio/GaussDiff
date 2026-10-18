"""Load the released NHANES GaussDiff checkpoint.

The checkpoint was serialised with ``torch.save`` on the module object itself
(not a ``state_dict``) at a time when the denoising network lived in a module
named ``tab_ddpm.modules``. That module path is recorded inside the pickle, so
a plain ``torch.load`` raises ``ModuleNotFoundError: No module named
'tab_ddpm'`` against the current package layout.

This helper registers the old module path as an alias of the current one
(``nhanes_pipeline.models.tabular_diffusion.modules``) before unpickling. The
class definition is unchanged, so the weights load exactly as saved.

Usage:
    from load_checkpoint import load_denoise_mlp
    net = load_denoise_mlp(device="cuda:0")
"""

import sys
import types
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_DEFAULT_CKPT = _HERE / "diffusion.pt"
_SRC = _HERE.parent / "src"


def _load_modules_source():
    """Import the denoising-network module file directly.

    The module is loaded from its file rather than as
    ``nhanes_pipeline.models.tabular_diffusion.modules``, because importing it
    through the package would execute the package __init__ chain, which pulls
    in the sampling code and hits a circular import. Only the layer
    definitions are needed to unpickle the checkpoint.
    """
    import importlib.util

    path = (
        _SRC / "nhanes_pipeline" / "models" / "tabular_diffusion" / "modules.py"
    )
    spec = importlib.util.spec_from_file_location("tab_ddpm.modules", path)
    modules = importlib.util.module_from_spec(spec)
    sys.modules["tab_ddpm.modules"] = modules
    spec.loader.exec_module(modules)
    return modules


def _register_legacy_module_alias():
    """Alias the checkpoint's recorded module path to the current one."""
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    if "tab_ddpm" not in sys.modules:
        sys.modules["tab_ddpm"] = types.ModuleType("tab_ddpm")
    if "tab_ddpm.modules" not in sys.modules:
        sys.modules["tab_ddpm"].modules = _load_modules_source()


def load_denoise_mlp(path=_DEFAULT_CKPT, device="cpu"):
    """Return the trained denoising MLP, ready for sampling."""
    _register_legacy_module_alias()
    net = torch.load(path, weights_only=False, map_location=device)
    net.eval()
    return net


if __name__ == "__main__":
    net = load_denoise_mlp()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Loaded {type(net).__name__} with {n_params:,} parameters")
