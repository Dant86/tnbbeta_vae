"""Tests for tnbbeta_vae.plotting."""

from __future__ import annotations

import plotly.io as pio

from tnbbeta_vae import plotting


def test_importing_registers_and_activates_the_template() -> None:
    assert plotting.TEMPLATE_NAME in pio.templates
    assert pio.templates.default == plotting.TEMPLATE_NAME


def test_template_sets_the_alptex_font_family() -> None:
    template = pio.templates[plotting.TEMPLATE_NAME]

    layout = template.layout
    assert layout.font.family == plotting.FONT_FAMILY  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
    assert layout.title.font.family == plotting.FONT_FAMILY  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
    assert layout.legend.font.family == plotting.FONT_FAMILY  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
