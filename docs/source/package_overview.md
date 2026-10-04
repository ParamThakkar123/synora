# Package Overview

Synora is organized into focused modules so you can use only the pieces you need.

## Quick Import (Public API)

For applications and examples, prefer the installed package name, `synora`. It
mirrors the Synora implementation package and exposes the same lazy public
API without importing optional training backends until you use them.

```python
import synora

print(synora.list_models())
agent = synora.create_model("dreamer", env="walker-walk", total_steps=1_000_000)
env = synora.make_env("CartPole-v1", backend="gym")
```

Use `synora` for direct component imports as well as factory helpers:

```python
from synora import DreamerAgent, DreamerConfig

cfg = DreamerConfig()
cfg.env = "walker-walk"
agent = DreamerAgent(cfg)
```

### Available Exports

| Category | Exports |
|----------|--------|
| **Friendly factories** | `create_config`, `create_model`, `make_env`, `list_models`, `list_env_backends`, `list_envs` |
| **Models / Agents** | `Dreamer`, `DreamerV1`, `DreamerV2`, `DreamerV3`, `DreamerAgent`, `Planet`, `JEPAAgent`, `IRISAgent`, `Genie`, `create_genie`, `DiT`, `create_dit` |
| **State-space models** | `RSSM`, `RecurrentStateSpaceModel`, `DreamerRSSM`, `ModularRSSM`, `create_modular_rssm` |
| **Vision** | `ConvEncoder`, `CNNEncoder`, `IRISEncoder`, `ConvDecoder`, `CNNDecoder`, `DenseDecoder`, `ActionDecoder`, `IRISDecoder`, `VideoTokenizer`, `create_video_tokenizer` |
| **Quantization** | `VectorQuantizer`, `VectorQuantizerEMA` |
| **Configs** | `DreamerConfig`, `JEPAConfig`, `DiTConfig`, `DiamondConfig`, `IRISConfig`, `GenieConfig`, `GenieSmallConfig`, `STTransformerConfig`, `VideoTokenizerConfig`, `LatentActionModelConfig`, `DynamicsModelConfig` |
| **Environments** | `make_atari_env`, `make_gym_env`, `make_mujoco_env`, `make_robotics_env`, `make_brax_env`, `make_procgen_env`, `GymImageEnv`, `ProcgenImageEnv`, `DeepMindControlEnv`, `DMLabEnv`, `make_dmlab_env`, `UnityMLAgentsEnv`, `TimeLimit`, `ActionRepeat`, wrappers, etc. |
| **Memory** | `ReplayBuffer`, `Memory`, `Episode`, `IRISReplayBuffer`, `IRISOnPolicyBuffer` |
| **Reward / Value** | `RewardModel`, `ValueModel`, `DreamerRewardModel`, `DreamerValueModel` |
| **Controllers** | `RSSMPolicy`, `RolloutGenerator`, `IRISPolicy`, `IRISActor`, `IRISCritic`, `CNNFeatureExtractor` |
| **Transformer blocks** | `STTransformer`, `MultiHeadSelfAttention`, `MultiHeadAttention`, `AdaLNNormalization`, `RMSNorm` |
| **Diffusion** | `DiT`, `DDPM`, `PatchEmbed`, `PatchUnEmbed`, `ActorCriticNetwork`, `RewardTerminationModel` |
| **Genie subcomponents** | `LatentActionModel`, `DynamicsModel`, `create_latent_action_model`, `create_dynamics_model` |
| **Export** | `export_any`, `export_model`, `ExportableAgentMixin` |
| **Registry / plugins** | `register_world_model`, `deregister_world_model`, `register_env_backend`, `deregister_env_backend` |
| **Deprecation** | `deprecated`, `deprecated_class`, `deprecated_function` |
| **Utilities** | `Logger`, `FreezeParameters`, `compute_return`, `preprocess_obs`, `__version__` |

Example usage:

```python
import synora

# Training
agent = synora.create_model("dreamer", env="walker-walk", total_steps=1_000_000)
agent.train()
```

## Core Modules

The module paths below are the public `synora.*` surface
(`from synora.models import Dreamer`).

- `synora.models`: High-level models and agents (`Dreamer`, `DreamerAgent`, `Planet`, `JEPAAgent`)
- `synora.configs`: Configuration containers for Dreamer, JEPA, and diffusion runs
- `synora.training`: Script-style training entrypoints for world models (VAE, MDNRNN, Controller, Planet, RSSM, JEPA)

## Environment Integration

- `synora.envs`: DMC, Gym/Gymnasium, Atari, MuJoCo, Unity ML-Agents adapters
- `synora.envs.wrappers`: Action repeat, normalization, time limits

## World Model Building Blocks

- `synora.models.dreamer_rssm`: Recurrent state-space model used by Dreamer
- `synora.models.modular_rssm`: Modular RSSM with swappable encoder/decoder/backbone for research experiments
- `synora.vision`: Encoders/decoders and action heads for latent dynamics models
- `synora.reward`: Reward and value prediction heads
- `synora.observations`: Symbolic and visual observation reconstruction modules

## Representation Learning and Diffusion

- `synora.models.vit`: Vision Transformer and JEPA predictor components
- `synora.models.diffusion`: DDPM scheduler and DiT model implementation
- `synora.masks`: Mask collators for JEPA-style context/target masking

## Data and Memory

- `synora.datasets`: CIFAR-10, ImageNet-1K, and generic `ImageFolder` dataset loaders
- `synora.memory`: Replay buffers for Dreamer and episode-based memory for PlaNet/RSSM

## Utilities

- `synora.utils`: Logging, parameter freezing, transforms
- `synora.transforms`: Data augmentation pipelines
- `synora.benchmarks`: CLI and reporting utilities

## Which API Should I Use?

- End-to-end Dreamer training: `DreamerAgent`
- End-to-end JEPA training: `JEPAAgent`
- World model training scripts: `synora.training` modules (e.g., `train_world_model` for VAE+MDNRNN+Controller pipeline)
- Low-level model experimentation: `Dreamer`, `RSSM`, decoder/encoder modules
- Custom world model architectures: `ModularRSSM` with swappable encoder/decoder/backbone
- Custom data pipelines: `make_cifar10`, `make_imagenet1k`, `make_imagefolder`
