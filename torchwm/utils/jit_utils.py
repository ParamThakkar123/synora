"""Deprecated TorchScript helpers.

TorchScript is in maintenance mode upstream. For in-process speed use
:func:`torchwm.utils.memory_utils.maybe_compile` (``torch.compile``); for
deployment artifacts use ``torchwm.export`` with ``format="exported_program"``
or ``"aoti"`` (``torch.export``).
"""

import warnings
from typing import Callable

import torch


def _warn(name: str) -> None:
    warnings.warn(
        f"{name}() is deprecated: TorchScript is in maintenance mode upstream. "
        "Use torchwm.maybe_compile() for speed, or torchwm.export_model(..., "
        "format='exported_program') for deployment.",
        DeprecationWarning,
        stacklevel=3,
    )


def jit_compile_function(func: Callable) -> Callable:
    """JIT compile a function for performance. Deprecated."""
    _warn("jit_compile_function")
    try:
        return torch.jit.script(func)
    except Exception as e:
        warnings.warn(f"JIT compilation failed: {e}", RuntimeWarning, stacklevel=2)
        return func


def jit_compile_module(module: torch.nn.Module) -> torch.nn.Module:
    """JIT compile a PyTorch module. Deprecated."""
    _warn("jit_compile_module")
    try:
        return torch.jit.script(module)
    except Exception as e:
        warnings.warn(f"JIT compilation failed: {e}", RuntimeWarning, stacklevel=2)
        return module
