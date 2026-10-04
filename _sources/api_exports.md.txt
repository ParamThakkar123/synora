# Exported Names by Category

This page lists every public class, function, and constant exported from the
`synora` top-level package, grouped by category. For signatures and docstrings
generated from source, see {doc}`api_reference`.

```{contents} Contents
:depth: 2
```

## Factory helpers

| Name | Description |
|---|---|
| `create_config(model, **overrides)` | Return a default config dict for a model family, with optional overrides applied. Models: `dreamer`, `jepa`, `iris`, `dit`, `genie`, `planet`. |
| `create_model(model, config=None, **overrides)` | Instantiate a model or high-level agent by canonical name (see {doc}`public_api`). |
| `make_env(env_id, backend="auto", **kwargs)` | Create a Synora environment through a named backend. Backends: `dmc`, `gym`, `atari`, `mujoco`, `robotics`, `procgen`, `brax`, `bsuite`, `unity`. |
| `list_models()` | Return canonical model names accepted by `create_model`. |
| `list_env_backends()` | Return backend names accepted by `make_env`. |
| `list_envs(model=None)` | Return known environment IDs, optionally filtered by model family. |
| `get_model_spec(name)` | Return metadata ({py:class}`ModelSpec`) for a model name or alias. |
| `get_env_backend_spec(name)` | Return metadata ({py:class}`EnvBackendSpec`) for an environment backend. |

## Data classes

| Name | Description |
|---|---|
| `ModelSpec` | Named tuple describing a registered model (name, import_path, config_path, aliases, description). |
| `EnvBackendSpec` | Named tuple describing a registered environment backend (name, factory_path, aliases, description). |

## World model agents

| Name | Source | Description |
|---|---|---|
| `Dreamer` | `synora.models.dreamer` | Base Dreamer world model (RSSM-based, V1-style). |
| `DreamerV1` | `synora.models.dreamer_v1` | DreamerV1 (alias for base Dreamer). |
| `DreamerV2` | `synora.models.dreamer_v2` | DreamerV2 (symlog two-hot heads, balanced KL). |
| `DreamerAgent` | `synora.models.dreamer` | High-level Dreamer agent with train/evaluate helpers. |
| `Planet` | `synora.models.planet` | PlaNet: Deep Planning Network. |
| `JEPAAgent` | `synora.models.jepa_agent` | I-JEPA agent for self-supervised visual representation learning. |
| `IRISAgent` | `synora.models.iris_agent` | IRIS agent for sample-efficient RL with Transformers. |
| `Genie` | `synora.models.genie` | Genie generative interactive environment. |
| `create_genie` | `synora.models.genie` | Create a Genie model with specified parameters. |
| `create_genie_small` | `synora.models.genie` | Create Genie-small variant (~50M params). |
| `create_genie_large` | `synora.models.genie` | Create Genie-large variant (~11B params). |

## Genie subcomponents

| Name | Source | Description |
|---|---|---|
| `LatentActionModel` | `synora.models.latent_action_model` | Learns latent actions from pairs of video frames. |
| `DynamicsModel` | `synora.models.dynamics_model` | Transformer-based dynamics for future token prediction. |
| `create_latent_action_model` | `synora.models.latent_action_model` | Factory for LatentActionModel. |
| `create_dynamics_model` | `synora.models.dynamics_model` | Factory for DynamicsModel. |

## State-space models

| Name | Source | Description |
|---|---|---|
| `RSSM` | `synora.models.rssm` | Recurrent State-Space Model (standalone). |
| `RecurrentStateSpaceModel` | `synora.models.rssm` | Alias for RSSM. |
| `DreamerRSSM` | `synora.models.dreamer_rssm` | RSSM variant used in Dreamer training loop. |
| `ModularRSSM` | `synora.models.modular_rssm` | Modular RSSM with swappable encoder/backbone/decoder. |
| `create_modular_rssm` | `synora.models.modular_rssm` | Factory for ModularRSSM. |

## Diffusion models

| Name | Source | Description |
|---|---|---|
| `DiT` | `synora.models.diffusion` | Diffusion Transformer model. |
| `create_dit` | `synora.models.diffusion` | Factory for DiT. |
| `PatchEmbed` | `synora.models.diffusion` | Image-to-patch embedding layer. |
| `PatchUnEmbed` | `synora.models.diffusion` | Patch-to-image un-embedding layer. |
| `DDPM` | `synora.models.diffusion` | Denoising Diffusion Probabilistic Model. |
| `ActorCriticNetwork` | `synora.models.diffusion` | Actor-critic head for DIAMOND-style RL. |
| `RewardTerminationModel` | `synora.models.diffusion` | Reward + termination predictor for DIAMOND. |
| `sinusoidal_time_embedding` | `synora.models.diffusion` | Time-step embedding for diffusion. |

## Vision components

| Name | Source | Description |
|---|---|---|
| `ConvEncoder` | `synora.vision.dreamer_encoder` | Dreamer convolutional encoder (image → embedding). |
| `CNNEncoder` | `synora.vision.planet_encoder` | PlaNet CNN encoder (image → embedding). |
| `IRISEncoder` | `synora.vision.iris_encoder` | IRIS encoder (image → discrete tokens). |
| `ConvDecoder` | `synora.vision.dreamer_decoder` | Dreamer convolutional decoder (latent → image distribution). |
| `CNNDecoder` | `synora.vision.planet_decoder` | PlaNet CNN decoder. |
| `DenseDecoder` | `synora.vision.dreamer_decoder` | MLP decoder for reward/value/discount. |
| `ActionDecoder` | `synora.vision.dreamer_decoder` | Dreamer policy head (latent → tanh-squashed action). |
| `IRISDecoder` | `synora.vision.iris_decoder` | IRIS decoder (tokens → image). |
| `VideoTokenizer` | `synora.vision.video_tokenizer` | Genie VQ-VAE video tokenizer. |
| `create_video_tokenizer` | `synora.vision.video_tokenizer` | Factory for VideoTokenizer. |
| `VectorQuantizer` | `synora.vision.vq_layer` | VQ-VAE vector quantization layer. |
| `VectorQuantizerEMA` | `synora.vision.vq_layer` | VQ-VAE with EMA codebook updates. |
| `TanhBijector` | `synora.vision.dreamer_decoder` | Tanh bijection for action squashing. |
| `SampleDist` | `synora.vision.dreamer_decoder` | MC-sampled distribution statistics. |

## Config classes

| Name | Source | Description |
|---|---|---|
| `DreamerConfig` | `synora.configs.dreamer_config` | Dreamer hyperparameter config. |
| `JEPAConfig` | `synora.configs.jepa_config` | JEPA hyperparameter config. |
| `DiTConfig` | `synora.configs.dit_config` | DiT hyperparameter config. |
| `get_dit_config` | `synora.configs.dit_config` | Factory for DiTConfig with presets. |
| `DiamondConfig` | `synora.configs.diamond_config` | DIAMOND hyperparameter config. |
| `IRISConfig` | `synora.configs.iris_config` | IRIS hyperparameter config. |
| `GenieConfig` | `synora.configs.genie_config` | Genie hyperparameter config. |
| `GenieSmallConfig` | `synora.configs.genie_config` | Genie-small preset config. |
| `STTransformerConfig` | `synora.configs.st_transformer_config` | ST-Transformer config. |
| `VideoTokenizerConfig` | `synora.configs.video_tokenizer_config` | Video tokenizer config. |
| `LatentActionModelConfig` | `synora.configs.lam_config` | Latent action model config. |
| `DynamicsModelConfig` | `synora.configs.dynamics_config` | Dynamics model config. |

## Constants

| Name | Description |
|---|---|
| `MODEL_SPECS` | Dict of all built-in model specs (name → {py:class}`ModelSpec`). |
| `ENV_BACKEND_SPECS` | Dict of all built-in environment backend specs. |
| `ATARI_100K_GAMES` | List of Atari 100K benchmark game names. |
| `HUMAN_SCORES` | Dict of human baseline scores for Atari 100K. |
| `RANDOM_SCORES` | Dict of random baseline scores for Atari 100K. |

## Memory / replay buffers

| Name | Source | Description |
|---|---|---|
| `ReplayBuffer` | `synora.memory.dreamer_memory` | Dreamer ring buffer (transitions → sequences). |
| `Memory` | `synora.memory.planet_memory` | Episode-based memory for PlaNet. |
| `Episode` | `synora.memory.planet_memory` | Single episode recording. |
| `IRISReplayBuffer` | `synora.memory.iris_memory` | Ring buffer for IRIS (uint8 images). |
| `IRISOnPolicyBuffer` | `synora.memory.iris_memory` | On-policy buffer for episode collection. |

## Environments and wrappers

| Name | Source | Description |
|---|---|---|
| `DeepMindControlEnv` | `synora.envs.dmc_env` | DeepMind Control Suite adapter. |
| `DMLabEnv` | `synora.envs.dmlab_env` | DeepMind Lab adapter. |
| `make_dmlab_env` | `synora.envs.dmlab_env` | Factory for DeepMind Lab. |
| `DMLAB_LEVELS` | `synora.envs.dmlab_env` | Available DMLab level names. |
| `GymImageEnv` | `synora.envs.gym_env` | Gymnasium image adapter. |
| `make_gym_env` | `synora.envs.gym_env` | Factory for GymImageEnv. |
| `MuJoCoImageEnv` | `synora.envs.mujoco_env` | MuJoCo image adapter. |
| `make_mujoco_env` | `synora.envs.mujoco_env` | Factory for MuJoCo environments. |
| `MujocoEnv` | `synora.envs.mujoco_env` | Alias of `MuJoCoImageEnv`. |
| `BraxImageEnv` | `synora.envs.brax_env` | Brax image adapter. |
| `make_brax_env` | `synora.envs.brax_env` | Factory for BraxImageEnv. |
| `BSuiteImageEnv` | `synora.envs.bsuite_env` | BSuite image adapter. |
| `make_bsuite_env` | `synora.envs.bsuite_env` | Factory for BSuiteImageEnv. |
| `list_available_bsuite_ids` | `synora.envs.bsuite_env` | List BSuite environment IDs. |
| `make_atari_env` | `synora.envs.atari_env` | Factory for Atari ALE environments. |
| `list_available_atari_envs` | `synora.envs.atari_env` | List available Atari game IDs. |
| `make_atari_vector_env` | `synora.envs.atari_env` | Factory for vectorized Atari. |
| `make_diamond_atari_env` | `synora.envs.diamond_atari` | DIAMOND-style Atari preprocessing. |
| `make_procgen_env` | `synora.envs.procgen_env` | Factory for Procgen environments. |
| `make_robotics_env` | `synora.envs.robotics_env` | Factory for Gymnasium Robotics. |
| `register_gymnasium_robotics_envs` | `synora.envs.robotics_env` | Register Robotics envs. |
| `list_gymnasium_robotics_envs` | `synora.envs.robotics_env` | List installed Robotics envs. |
| `UnityMLAgentsEnv` | `synora.envs.unity_env` | Unity ML-Agents adapter. |
| `make_unity_mlagents_env` | `synora.envs.unity_env` | Factory for UnityMLAgentsEnv. |
| `WorldModelEnv` | `synora.envs.world_model_env` | Environment inside a learned world model. |
| `make_world_model_env` | `synora.envs.world_model_env` | Factory for WorldModelEnv. |
| `TimeLimit` | `synora.envs.wrappers` | Episode time limit wrapper. |
| `ActionRepeat` | `synora.envs.wrappers` | Action repeat wrapper. |
| `NormalizeActions` | `synora.envs.wrappers` | Action normalization to [-1, 1]. |
| `ObsDict` | `synora.envs.wrappers` | Observation-to-dict conversion. |
| `OneHotAction` | `synora.envs.wrappers` | Discrete to one-hot action conversion. |
| `RewardObs` | `synora.envs.wrappers` | Reward observation injection. |
| `ResizeImage` | `synora.envs.wrappers` | Image resizing wrapper. |
| `RenderImage` | `synora.envs.wrappers` | Render-based image observation. |
| `SelectAction` | `synora.envs.wrappers` | Action selection wrapper. |

## Controllers and policies

| Name | Source | Description |
|---|---|---|
| `RSSMPolicy` | `synora.controller` | RSSM-based policy for Dreamer. |
| `RolloutGenerator` | `synora.controller` | Policy rollouts in the environment. |
| `IRISActor` | `synora.controller` | IRIS actor head. |
| `IRISCritic` | `synora.controller` | IRIS critic head. |
| `IRISPolicy` | `synora.controller` | IRIS combined actor-critic policy. |
| `CNNFeatureExtractor` | `synora.controller` | CNN feature extractor for policy inputs. |

## Export

| Name | Source | Description |
|---|---|---|
| `export_any(obj, path, format, ...)` | `synora.export` | Export a model or agent (`exported_program`, `aoti`, ONNX, TensorRT, TorchScript). |
| `export_model(module, path, format, ...)` | `synora.export` | Export a raw nn.Module. |
| `load_exported(path, format=None)` | `synora.export` | Load any exported artifact as a callable. |
| `verify_export(module, exported, example_inputs)` | `synora.export` | Check an artifact against its eager module; returns max abs error. |
| `ExportableAgentMixin` | `synora.export` | Mixin that adds `.export()` to custom agents. |

## Inference and deployment

See {doc}`deployment_guide`.

| Name | Source | Description |
|---|---|---|
| `optimize_for_inference(module, ...)` | `synora.inference` | Wrap a module with a precision policy and optional `torch.compile` / CUDA graphs. |
| `InferenceModel` | `synora.inference` | The wrapper returned by `optimize_for_inference`. |
| `inference_context(device, precision)` | `synora.inference` | `inference_mode` plus autocast at one resolved precision. |
| `make_stepper(agent)` | `synora.inference` | Uniform `init_state` / `observe` / `act` / `imagine` step interface. |
| `DreamerStepper`, `IRISStepper` | `synora.inference` | Steppers for Dreamer and IRIS. |
| `DreamerStepModule` | `synora.inference` | One Dreamer observe+act step as a pure, exportable module. |
| `benchmark_step(fn, *args)` | `synora.inference` | Latency percentiles, throughput and peak memory. |
| `rollout_drift(reference, candidate, state, inputs)` | `synora.inference` | Closed-loop divergence of an optimized step from a reference. |
| `quantize_weights(module)` | `synora.inference` | Weight-only int8 quantization of `nn.Linear` layers. |
| `save_bundle(dir, module, ...)` / `load_bundle(dir)` | `synora.inference` | Deployment bundles: weights, config, artifacts, manifest. |
| `STKVCache` | `synora.blocks.st_transformer` | Temporal KV cache for incremental Genie / ST-transformer generation. |

## Reward and value models

| Name | Source | Description |
|---|---|---|
| `RewardModel` | `synora.reward` | Base reward model. |
| `ValueModel` | `synora.reward` | Base value model. |
| `DreamerRewardModel` | `synora.reward` | Dreamer reward predictor. |
| `DreamerValueModel` | `synora.reward` | Dreamer value function. |

## Transformer blocks

| Name | Source | Description |
|---|---|---|
| `STTransformer` | `synora.blocks` | Spatiotemporal Transformer (Genie). |
| `MultiHeadSelfAttention` | `synora.blocks` | Multi-head self-attention. |
| `MultiHeadAttention` | `synora.blocks` | Multi-head cross-attention. |
| `AdaLNNormalization` | `synora.blocks` | Adaptive layer norm for diffusion. |
| `RMSNorm` | `synora.blocks` | Root mean square layer norm. |

## Plugin registry

| Name | Source | Description |
|---|---|---|
| `register_world_model(name, import_path, ...)` | `synora.registry` | Register a custom world model architecture. |
| `deregister_world_model(name)` | `synora.registry` | Remove a registered model. |
| `get_registered_model_spec(name)` | `synora.registry` | Look up a registered model spec. |
| `list_registered_models()` | `synora.registry` | List all externally registered model names. |
| `register_env_backend(name, factory_path, ...)` | `synora.registry` | Register a custom environment backend. |
| `deregister_env_backend(name)` | `synora.registry` | Remove a registered env backend. |
| `list_registered_env_backends()` | `synora.registry` | List all registered env backends. |

## Deprecation utilities

| Name | Source | Description |
|---|---|---|
| `deprecated(version, reason)` | `synora.utils.deprecation` | Decorator to mark functions/classes as deprecated. |
| `deprecated_class(version, alternative)` | `synora.utils.deprecation` | Shortcut for deprecating a class. |
| `deprecated_function(version, alternative)` | `synora.utils.deprecation` | Shortcut for deprecating a function. |

## General utilities

| Name | Source | Description |
|---|---|---|
| `Logger` | `synora.utils` | Logging utility. |
| `FreezeParameters` | `synora.utils` | Context manager to freeze model parameters. |
| `compute_return(rewards, values, gamma, lambda_)` | `synora.utils` | Compute GAE or λ-return. |
| `preprocess_obs(obs)` | `synora.utils` | Observation preprocessing (resize, normalize). |

## Version

| Name | Description |
|---|---|
| `__version__` | Package version string (semver). |
