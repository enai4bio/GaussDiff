import os
from .core import lib

import zero
import torch

from .models.tabular_diffusion import GaussianMultinomialDiffusion

def sample(dataset, diffusion, raw_config):

    zero.improve_reproducibility(raw_config['sample']['main']['sample_seed'] )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    diffusion.eval()
    with torch.no_grad():
        diffusion.random_sample(dataset, raw_config)
