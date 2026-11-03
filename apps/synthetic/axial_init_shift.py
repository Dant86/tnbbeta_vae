"""Run ``axial_recovery`` (tnbbeta only) with a constant raw-p offset added.

The offset (for a given p target) is added inside ``heads.tnbbeta_posterior``.
Usage: python -m apps.synthetic.axial_init_shift P_TARGET(or none) -- <axial args>
"""

import math
import sys

import torch

from apps.synthetic import axial_recovery
from tnbbeta_vae.models import heads
import tnbbeta_vae.models.mlp_vae as mv

pt = sys.argv[1]
rest = sys.argv[3:]  # argv[2] is the literal "--"
orig = heads.tnbbeta_posterior
if pt != "none":
    off = math.log(float(pt) / (1 - float(pt)))

    def patched(raw, latent_dim, fixed_epsilon=None, fixed_mean_direction=False):
        s = torch.zeros(raw.shape[-1], dtype=raw.dtype, device=raw.device)
        s[(0 if fixed_mean_direction else latent_dim)] = off
        return orig(raw + s, latent_dim, fixed_epsilon, fixed_mean_direction)

    heads.tnbbeta_posterior = patched
    mv.tnbbeta_posterior = patched

axial_recovery.main(rest)
