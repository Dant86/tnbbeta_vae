import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

_PATH = Path(__file__).parents[2] / "notebooks" / "plot_sphere_3d.py"
_spec = importlib.util.spec_from_file_location("plot_sphere_3d", _PATH)
assert _spec is not None and _spec.loader is not None
plot_sphere_3d: Any = importlib.util.module_from_spec(_spec)
sys.modules["plot_sphere_3d"] = plot_sphere_3d
_spec.loader.exec_module(plot_sphere_3d)

_N = 200


def _arrays(latent_dim: int = 3) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    unit = rng.normal(size=(_N, latent_dim))
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    return {
        "model_name": np.array("conv_tnbbeta_spherical_vae"),
        "labels": np.arange(_N) % 10,
        "direction": unit,
        "mode_direction": unit,
        "z": unit,
        "kl": rng.uniform(1, 9, _N),
        "p": np.where(np.arange(_N) < 20, 0.52, 0.9),
        "q": rng.uniform(0, 0.1, _N),
        "epsilon": rng.uniform(1, 3, _N),
    }


def _run(monkeypatch: pytest.MonkeyPatch, npz: Path, out_dir: Path) -> None:
    argv = ["plot_sphere_3d.py", "--npz", str(npz), "--out-dir", str(out_dir)]
    monkeypatch.setattr(sys, "argv", argv)
    plot_sphere_3d.main()


def test_writes_html_named_after_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    npz = tmp_path / "run_a" / "latents_final_test.npz"
    npz.parent.mkdir()
    np.savez(npz, **_arrays())

    _run(monkeypatch, npz, tmp_path / "out")

    html = (tmp_path / "out" / "run_a_sphere.html").read_text()
    for name in plot_sphere_3d.CLASSES:
        assert f'"name":"{name}"' in html
        assert f'"name":"{name} (z)"' in html
    assert "color: p" in html


def test_figure_layers_and_ambiguous_points() -> None:
    arrays = _arrays()
    fig = plot_sphere_3d.build_figure(arrays, "t", max_points=100)

    assert [t.type for t in fig.data] == ["surface"] + ["scatter3d"] * 20
    assert [t.visible for t in fig.data[11:]] == ["legendonly"] * 10
    assert sum(len(t.x) for t in fig.data[1:11]) == 100
    symbols = np.concatenate([np.asarray(t.marker.symbol) for t in fig.data[1:11]])
    assert set(symbols) == {"circle", "diamond"}


def test_rejects_other_dimensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    npz = tmp_path / "d4.npz"
    np.savez(npz, **_arrays(latent_dim=4))
    with pytest.raises(SystemExit):
        _run(monkeypatch, npz, tmp_path)
