"""Tests for tnbbeta_vae.registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel
import pytest

from tnbbeta_vae import registry

if TYPE_CHECKING:
    from collections.abc import Iterator


class _DummyConfig(BaseModel):
    hidden_dim: int = 16


class _DummyModel:
    def __init__(self, config: _DummyConfig) -> None:
        self.config = config


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Isolates each test's registrations from the module-level registry."""
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)


def test_register_and_build_model() -> None:
    registry.register_model("dummy", config_cls=_DummyConfig)(_DummyModel)

    model = registry.build_model("dummy", hidden_dim=32)

    assert isinstance(model, _DummyModel)
    assert model.config.hidden_dim == 32


def test_register_duplicate_name_raises() -> None:
    registry.register_model("dummy", config_cls=_DummyConfig)(_DummyModel)

    with pytest.raises(ValueError, match="already registered"):
        registry.register_model("dummy", config_cls=_DummyConfig)(_DummyModel)


def test_get_unknown_model_raises_key_error() -> None:
    with pytest.raises(KeyError, match="No model registered"):
        registry.get_registered_model("does_not_exist")


def test_list_registered_models_sorted() -> None:
    registry.register_model("zeta", config_cls=_DummyConfig)(_DummyModel)
    registry.register_model("alpha", config_cls=_DummyConfig)(_DummyModel)

    assert registry.list_registered_models() == ["alpha", "zeta"]
