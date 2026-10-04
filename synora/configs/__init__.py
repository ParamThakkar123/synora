"""Lazy config exports.

Configuration modules can have optional training dependencies, so the package
initializer avoids importing every config eagerly.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DreamerConfig": "synora.configs.dreamer_config",
    "JEPAConfig": "synora.configs.jepa_config",
    "DiTConfig": "synora.configs.dit_config",
    "get_dit_config": "synora.configs.dit_config",
    "dit_preset_config": "synora.configs.dit_config",
    "list_dit_presets": "synora.configs.dit_config",
    "DIT_PRESETS": "synora.configs.dit_config",
    "DiamondConfig": "synora.configs.diamond_config",
    "IRISConfig": "synora.configs.iris_config",
    "ATARI_100K_GAMES": "synora.configs.diamond_config",
    "HUMAN_SCORES": "synora.configs.diamond_config",
    "RANDOM_SCORES": "synora.configs.diamond_config",
    "GenieConfig": "synora.configs.genie_config",
    "GenieSmallConfig": "synora.configs.genie_config",
    "STTransformerConfig": "synora.configs.genie_config",
    "VideoTokenizerConfig": "synora.configs.genie_config",
    "LatentActionModelConfig": "synora.configs.genie_config",
    "DynamicsModelConfig": "synora.configs.genie_config",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = list(_EXPORTS)
