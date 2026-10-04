"""Synora public API.

This package keeps imports lightweight while still exposing a friendly top-level
surface.  Common workflows can use the small factory helpers::

    import synora

    cfg = synora.create_config("dreamer", env="walker-walk")
    agent = synora.create_model("dreamer", cfg)
    env = synora.make_env("CartPole-v1", backend="gym")

Lower-level research components remain available as lazy top-level exports, for
example ``from synora import DreamerAgent, ConvEncoder, ReplayBuffer``, and
every implementation submodule is reachable directly::

    from synora.models import Dreamer
    from synora.training.eval_jepa import jepa_linear_probe
    import synora.envs
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from synora._version import __version__  # noqa: F401


try:
    from synora.export import install_export_method as _install_export_method

    _install_export_method()
except ModuleNotFoundError as exc:  # pragma: no cover - torch-free metadata imports
    if exc.name != "torch":
        raise

_API_EXPORTS = {
    "EnvBackendSpec": "synora.api",
    "ModelSpec": "synora.api",
    "MODEL_SPECS": "synora.api",
    "ENV_BACKEND_SPECS": "synora.api",
    "create_config": "synora.api",
    "create_model": "synora.api",
    "get_env_backend_spec": "synora.api",
    "get_model_spec": "synora.api",
    "list_env_backends": "synora.api",
    "list_envs": "synora.api",
    "list_models": "synora.api",
    "make_env": "synora.api",
    "export_any": "synora.export",
    "export_model": "synora.export",
    "load_exported": "synora.export",
    "verify_export": "synora.export",
    "ExportableAgentMixin": "synora.export",
}

_LAZY_EXPORTS: dict[str, str] = {
    # Agents and high-level models.
    "Dreamer": "synora.models",
    "DreamerV1": "synora.models",
    "DreamerV2": "synora.models",
    "DreamerV3": "synora.models",
    "DreamerAgent": "synora.models",
    "Planet": "synora.models",
    "JEPAAgent": "synora.models",
    "IRISAgent": "synora.models",
    "IRISTransformer": "synora.models",
    "IRISWorldModel": "synora.models",
    "LPIPSPerceptualLoss": "synora.vision",
    "build_perceptual_loss": "synora.vision",
    "compute_lambda_return": "synora.models",
    "VisionTransformer": "synora.models",
    "ModularRSSM": "synora.models",
    "create_modular_rssm": "synora.models",
    "Genie": "synora.models",
    "LatentActionModel": "synora.models",
    "DynamicsModel": "synora.models",
    "create_genie": "synora.models",
    "create_genie_small": "synora.models",
    "create_genie_large": "synora.models",
    "create_latent_action_model": "synora.models",
    "create_dynamics_model": "synora.models",
    # State-space models.
    "RSSM": "synora.models",
    "RecurrentStateSpaceModel": "synora.models",
    # ``dreamer_rssm`` defines the class as ``RSSM``; ``synora.models``
    # is what exposes it under the ``DreamerRSSM`` alias.
    "DreamerRSSM": "synora.models",
    # Vision components.
    "ConvEncoder": "synora.vision",
    "CNNEncoder": "synora.vision",
    "ConvDecoder": "synora.vision",
    "CNNDecoder": "synora.vision",
    "DenseDecoder": "synora.vision",
    "ActionDecoder": "synora.vision",
    "TanhBijector": "synora.vision",
    "SampleDist": "synora.vision",
    "IRISEncoder": "synora.vision",
    "IRISDecoder": "synora.vision",
    "VideoTokenizer": "synora.vision",
    "create_video_tokenizer": "synora.vision",
    "VectorQuantizer": "synora.vision",
    "VectorQuantizerEMA": "synora.vision",
    # Memory.
    "ReplayBuffer": "synora.memory",
    "Memory": "synora.memory",
    "Episode": "synora.memory",
    "IRISReplayBuffer": "synora.memory",
    "IRISOnPolicyBuffer": "synora.memory",
    # Diffusion models.
    # ``DiT`` and ``DDPM`` name both a class and a sibling module. Importing the
    # module binds it over the class on the package, so which one
    # ``synora.models.diffusion.DiT`` returns depends on import order -
    # point at the defining modules to make it deterministic.
    "DiT": "synora.models.diffusion.DiT",
    "create_dit": "synora.models.diffusion",
    "PatchEmbed": "synora.models.diffusion",
    "PatchUnEmbed": "synora.models.diffusion",
    "DDPM": "synora.models.diffusion.DDPM",
    "ActorCriticNetwork": "synora.models.diffusion",
    "RewardTerminationModel": "synora.models.diffusion",
    "sinusoidal_time_embedding": "synora.models.diffusion",
    # Transformer blocks and layers.
    "STTransformer": "synora.blocks",
    "MultiHeadSelfAttention": "synora.blocks",
    "MultiHeadAttention": "synora.blocks",
    "AdaLNNormalization": "synora.blocks",
    "RMSNorm": "synora.blocks",
    # Controllers and policies.
    "RSSMPolicy": "synora.controller",
    "RolloutGenerator": "synora.controller",
    "IRISActor": "synora.controller",
    "IRISCritic": "synora.controller",
    "IRISPolicy": "synora.controller",
    "CNNFeatureExtractor": "synora.controller",
    # Configs.
    "DreamerConfig": "synora.configs",
    "JEPAConfig": "synora.configs",
    "DiTConfig": "synora.configs",
    "dit_preset_config": "synora.configs",
    "list_dit_presets": "synora.configs",
    "get_dit_config": "synora.configs",
    "DiamondConfig": "synora.configs",
    "IRISConfig": "synora.configs",
    "GenieConfig": "synora.configs",
    "GenieSmallConfig": "synora.configs",
    "STTransformerConfig": "synora.configs",
    "VideoTokenizerConfig": "synora.configs",
    "LatentActionModelConfig": "synora.configs",
    "DynamicsModelConfig": "synora.configs",
    "ATARI_100K_GAMES": "synora.configs",
    "HUMAN_SCORES": "synora.configs",
    "RANDOM_SCORES": "synora.configs",
    # Environments and wrappers.
    "BSuiteImageEnv": "synora.envs",
    "make_bsuite_env": "synora.envs",
    "list_available_bsuite_ids": "synora.envs",
    "make_atari_env": "synora.envs",
    "list_available_atari_envs": "synora.envs",
    "make_atari_vector_env": "synora.envs",
    "make_diamond_atari_env": "synora.envs.diamond_atari",
    "MuJoCoImageEnv": "synora.envs",
    "make_mujoco_env": "synora.envs",
    "make_mujoco_env_from_config": "synora.envs",
    "list_gymnasium_robotics_envs": "synora.envs",
    "make_robotics_env": "synora.envs",
    "register_gymnasium_robotics_envs": "synora.envs",
    "GymImageEnv": "synora.envs",
    "make_gym_env": "synora.envs",
    "WorldModelEnv": "synora.envs",
    "make_world_model_env": "synora.envs",
    "BraxImageEnv": "synora.envs",
    "make_brax_env": "synora.envs",
    "DeepMindControlEnv": "synora.envs",
    "DMLabEnv": "synora.envs",
    "make_dmlab_env": "synora.envs",
    "DMLAB_LEVELS": "synora.envs",
    "UnityMLAgentsEnv": "synora.envs",
    "make_unity_mlagents_env": "synora.envs",
    "MujocoEnv": "synora.envs",
    "TimeLimit": "synora.envs",
    "ActionRepeat": "synora.envs",
    "NormalizeActions": "synora.envs",
    "ObsDict": "synora.envs",
    "OneHotAction": "synora.envs",
    "RewardObs": "synora.envs",
    "ResizeImage": "synora.envs",
    "RenderImage": "synora.envs",
    "SelectAction": "synora.envs",
    # I-JEPA evaluation (paper Appendix A.2).
    "jepa_linear_probe": "synora.training.eval_jepa",
    "load_jepa_encoder": "synora.training.eval_jepa",
    # Reward/value models.
    "RewardModel": "synora.reward",
    "ValueModel": "synora.reward",
    "DreamerRewardModel": "synora.reward",
    "DreamerValueModel": "synora.reward",
    # Registry / plugin system.
    "register_world_model": "synora.registry",
    "deregister_world_model": "synora.registry",
    "get_registered_model_spec": "synora.registry",
    "list_registered_models": "synora.registry",
    "register_env_backend": "synora.registry",
    "deregister_env_backend": "synora.registry",
    "list_registered_env_backends": "synora.registry",
    # Deprecation helpers.
    "deprecated": "synora.utils.deprecation",
    "deprecated_class": "synora.utils.deprecation",
    "deprecated_function": "synora.utils.deprecation",
    # Utilities.
    "Logger": "synora.utils",
    "FreezeParameters": "synora.utils",
    "compute_return": "synora.utils",
    "preprocess_obs": "synora.utils",
    # Performance measurement and tuning.
    "ThroughputMeter": "synora.utils.throughput",
    "measure_steps": "synora.utils.throughput",
    "tensor_nbytes": "synora.utils.throughput",
    "enable_performance_defaults": "synora.utils.memory_utils",
    "maybe_compile": "synora.utils.memory_utils",
    "to_channels_last": "synora.utils.memory_utils",
    # Efficient inference and deployment.
    "InferenceModel": "synora.inference",
    "optimize_for_inference": "synora.inference",
    "inference_context": "synora.inference",
    "make_stepper": "synora.inference",
    "DreamerStepper": "synora.inference",
    "DreamerStepModule": "synora.inference",
    "IRISStepper": "synora.inference",
    "benchmark_step": "synora.inference",
    "rollout_drift": "synora.inference",
    "quantize_weights": "synora.inference",
    "save_bundle": "synora.inference",
    "load_bundle": "synora.inference",
    "STKVCache": "synora.blocks.st_transformer",
}

_EXPORTS = {**_API_EXPORTS, **_LAZY_EXPORTS}


# Submodules that are part of the public surface and so must resolve as
# attributes of the package, not only via ``import synora.<name>``.
_SUBMODULE_EXPORTS = ("api",)


def __getattr__(name: str) -> Any:
    """Lazily import public symbols on first access."""

    if name in _SUBMODULE_EXPORTS:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
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


__all__ = ["__version__", *_SUBMODULE_EXPORTS, *_EXPORTS]
