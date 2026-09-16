"""Model registry mapping string names to (config, model) pairs.

Models register themselves with :func:`register_model`, associating a name
with a Pydantic config class and the model class it configures. This lets
experiments be specified declaratively (e.g. from a YAML/JSON config file
or a CLI flag) and looked up by name at train time, and lets tooling list
every known model without importing each one by hand.

Example:
    ```python
    from pydantic import BaseModel
    from torch import nn

    from tnbbeta_vae.registry import register_model


    class MyVaeConfig(BaseModel):
        latent_dim: int = 64


    @register_model("my_vae", config_cls=MyVaeConfig)
    class MyVae(nn.Module):
        def __init__(self, config: MyVaeConfig) -> None:
            super().__init__()
            self.latent_dim = config.latent_dim
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "RegisteredModel",
    "build_model",
    "get_registered_model",
    "list_registered_models",
    "register_model",
]


@dataclass(frozen=True)
class RegisteredModel:
    """A registry entry pairing a model class with its config class.

    Attributes:
        name: Unique registry key.
        config_cls: Pydantic model describing the model's hyperparameters.
        model_cls: The model class, whose ``__init__`` accepts a single
            ``config`` argument of type ``config_cls``.
    """

    name: str
    config_cls: type[BaseModel]
    model_cls: type


_REGISTRY: dict[str, RegisteredModel] = {}


def register_model[ConfigT: BaseModel, ModelT](
    name: str, *, config_cls: type[ConfigT]
) -> Callable[[type[ModelT]], type[ModelT]]:
    """Class decorator that registers a model under ``name``.

    Args:
        name: Unique registry key, e.g. ``"tnbbeta_vae_v1"``.
        config_cls: Pydantic config class for the model.

    Returns:
        A decorator that registers and returns the model class unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(model_cls: type[ModelT]) -> type[ModelT]:
        if name in _REGISTRY:
            raise ValueError(f"Model {name!r} is already registered.")
        _REGISTRY[name] = RegisteredModel(
            name=name, config_cls=config_cls, model_cls=model_cls
        )
        return model_cls

    return decorator


def get_registered_model(name: str) -> RegisteredModel:
    """Looks up a registry entry by name.

    Args:
        name: Registry key to look up.

    Returns:
        The matching :class:`RegisteredModel`.

    Raises:
        KeyError: If no model is registered under ``name``.
    """
    try:
        return _REGISTRY[name]
    except KeyError as e:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(
            f"No model registered under {name!r}. Known models: {known}."
        ) from e


def list_registered_models() -> list[str]:
    """Returns the names of all registered models, sorted alphabetically."""
    return sorted(_REGISTRY)


def build_model(name: str, **config_overrides: object) -> object:
    """Builds a registered model by name from keyword config overrides.

    Args:
        name: Registry key of the model to build.
        **config_overrides: Fields to pass to the model's config class.

    Returns:
        An instance of the registered model class.
    """
    entry = get_registered_model(name)
    config = entry.config_cls(**config_overrides)
    return entry.model_cls(config)
