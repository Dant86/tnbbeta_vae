"""Shared fixtures for the CLI tests.

Plotly's ``write_image`` launches headless Chrome through kaleido on every call
(~20 s each, and it can hang on a busy node), yet the CLI tests only assert that an
output file appears. The figure-building code still runs in full; only the final
rasterization is stubbed. Mark a test ``@pytest.mark.real_render`` to opt back in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plotly.basedatatypes import BaseFigure
import pytest

# The 8-byte PNG signature: enough for "a non-empty file with the right extension".
_PNG_STUB = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def _stub_plotly_write_image(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.node.get_closest_marker("real_render") is not None:
        return

    def fake_write_image(
        self: BaseFigure, file: Any, *args: Any, **kwargs: Any
    ) -> None:
        Path(file).write_bytes(_PNG_STUB)

    monkeypatch.setattr(BaseFigure, "write_image", fake_write_image)
