"""Regression test for a circular import between training and models.

``tnbbeta_vae.training.checkpoint`` imports ``tnbbeta_vae.models`` (for its
``@register_model`` side effects) and ``tnbbeta_vae.models`` imports every
model module, including ``sphere_diffusion``. If any model module imports
from ``tnbbeta_vae.training`` (or one of its submodules) at module level,
that creates an import cycle: importing ``tnbbeta_vae.training`` first pulls
in ``tnbbeta_vae.models``, which pulls the model module, which tries to
import back from the partially-initialized ``tnbbeta_vae.training`` package.

This only shows up when ``tnbbeta_vae.training`` is the *first* of the two
packages imported in a fresh interpreter -- once ``tnbbeta_vae.models`` (or
anything that imports it, e.g. another test module) has already run in the
same process, ``sys.modules`` is pre-populated and the cycle is invisible.
Pytest's own test collection order can accidentally provide that priming,
so this test spawns a genuinely fresh subprocess rather than relying on
whatever other test modules happened to run first in-process.
"""

from __future__ import annotations

import subprocess
import sys


def test_importing_training_package_alone_does_not_raise() -> None:
    """``import tnbbeta_vae.training`` must succeed with no other imports first."""
    result = subprocess.run(
        [sys.executable, "-c", "import tnbbeta_vae.training"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
