"""One genuine kaleido render, so a broken Chrome/kaleido setup is still caught.

Skipped by default (the stub in conftest.py covers every other CLI test); run with
``uv run pytest -m real_render``.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import pytest


@pytest.mark.real_render
def test_kaleido_renders_a_png(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    if "real_render" not in (request.config.getoption("-m") or ""):
        pytest.skip("slow; run with -m real_render")
    output = tmp_path / "figure.png"

    go.Figure(go.Scatter(x=[0, 1], y=[0, 1])).write_image(output)

    assert output.read_bytes()[:4] == b"\x89PNG"
    assert output.stat().st_size > 100
