"""Efficient inference and deployment of world models.

The runtime API (importing it does not pull in OpenCV or the play scripts):

* :func:`optimize_for_inference` / :class:`InferenceModel` - precision,
  ``torch.compile`` + CUDA graphs, channels-last, weight casting.
* :func:`inference_context` / :func:`resolve_precision` - one precision policy.
* Steppers (:class:`DreamerStepper`, :class:`IRISStepper`,
  :func:`make_stepper`) - a uniform, stateful step interface, plus
  :class:`DreamerStepModule` for export.
* :func:`benchmark_step` / :func:`rollout_drift` - speed and fidelity.
* :func:`quantize_weights` - weight-only int8.
* :func:`save_bundle` / :func:`load_bundle` - deployment bundles.

The evaluation and interactive-play entry points used by ``torchwm eval`` /
``torchwm play`` also live in this package (``eval_diamond``, ``play_*``). They
are imported lazily by the CLI because they pull in OpenCV and the evaluation
networks.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "InferenceModel": "torchwm.inference.runtime",
    "optimize_for_inference": "torchwm.inference.runtime",
    "inference_context": "torchwm.inference.precision",
    "precision_dtype": "torchwm.inference.precision",
    "resolve_precision": "torchwm.inference.precision",
    "DreamerStepModule": "torchwm.inference.steppers",
    "DreamerStepper": "torchwm.inference.steppers",
    "IRISStepper": "torchwm.inference.steppers",
    "ImagineOutput": "torchwm.inference.steppers",
    "WorldModelStepper": "torchwm.inference.steppers",
    "make_stepper": "torchwm.inference.steppers",
    "DriftReport": "torchwm.inference.benchmark",
    "LatencyReport": "torchwm.inference.benchmark",
    "benchmark_step": "torchwm.inference.benchmark",
    "rollout_drift": "torchwm.inference.benchmark",
    "Int8WeightOnlyLinear": "torchwm.inference.quantize",
    "quantize_weights": "torchwm.inference.quantize",
    "weight_memory_bytes": "torchwm.inference.quantize",
    "Bundle": "torchwm.inference.bundle",
    "load_bundle": "torchwm.inference.bundle",
    "save_bundle": "torchwm.inference.bundle",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = list(_EXPORTS)
