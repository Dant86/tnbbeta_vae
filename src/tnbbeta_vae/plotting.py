"""Shared plotly styling matching this project's paper template (schein-lab/alptex).

alptex's active font configuration (its ``preamble.tex``, option 2 -- the default)
pairs Bitstream Charter (serif body text, via the ``mathdesign`` package) with PT
Sans (sans-serif supporting text). Kaleido renders through a headless browser using
whatever fonts are installed on the machine producing the image, so this maps to
that pairing's closest broadly-available equivalent behind a CSS-style font stack,
rather than assuming Charter itself is installed: a plot generated on a bare
cluster node still renders, just falling back to Georgia (or a generic serif) if
Charter isn't available there.

Importing this module registers the ``"alptex"`` template and makes it plotly's
default for every figure built afterward, in this process -- every plotting
script in this project (``apps.eval.svae_latitude``, ``apps.eval.hammer_projection``)
imports it for exactly that reason, so the theme only needs to change here.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

__all__ = ["FONT_FAMILY", "TEMPLATE_NAME"]

TEMPLATE_NAME = "alptex"
FONT_FAMILY = "Charter, Georgia, serif"
_FONT_COLOR = "#1a1a1a"
_BASE_SIZE = 14


def _register_template() -> None:
    """Registers the ``alptex`` template and makes it plotly's default."""
    pio.templates[TEMPLATE_NAME] = go.layout.Template(
        layout=go.Layout(
            font={"family": FONT_FAMILY, "size": _BASE_SIZE, "color": _FONT_COLOR},
            title={"font": {"family": FONT_FAMILY, "size": _BASE_SIZE + 4}},
            legend={"font": {"family": FONT_FAMILY, "size": _BASE_SIZE - 2}},
        )
    )
    pio.templates.default = TEMPLATE_NAME


_register_template()
