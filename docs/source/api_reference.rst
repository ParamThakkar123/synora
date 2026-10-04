API Reference
=============

This reference is generated from source docstrings and grouped by workflow. Use
:doc:`world_models_guide` for conceptual explanations and this page for exact
classes, functions, and module-level APIs.

Public package surface
----------------------

These modules expose the most common imports and lazy constructors.

Use ``synora`` for common workflows::

   import synora
   agent = synora.create_model("dreamer", env="walker-walk")

**Primary modules:** ``synora``, ``synora.models``, ``synora.configs``, ``synora.catalog``, and ``synora.envs``.

.. automodule:: synora
   :no-index:

.. automodule:: synora.api
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.export
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.inference.runtime
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.inference.precision
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.inference.steppers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.inference.benchmark
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.inference.quantize
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.inference.bundle
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models
   :no-index:

.. automodule:: synora.catalog
   :members:
   :undoc-members:
   :show-inheritance:

Model catalog
-------------

Core model families
~~~~~~~~~~~~~~~~~~~

**Key classes:** ``Dreamer``, ``DreamerAgent``, ``RSSM``, ``RecurrentStateSpaceModel``, ``Planet``, ``ModularRSSM``, ``JEPAAgent``, ``VisionTransformer``, ``IRISAgent``, ``IRISTransformer``, ``IRISWorldModel``, ``Genie``, ``LatentActionModel``, and ``DynamicsModel``.

.. automodule:: synora.models.dreamer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.dreamer_rssm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.rssm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.planet
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.mdrnn
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.controller
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.modular_rssm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.jepa_agent
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.vit
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.iris_agent
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.iris_transformer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.genie
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.latent_action_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.dynamics_model
   :members:
   :undoc-members:
   :show-inheritance:

Diffusion and DIAMOND components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Key classes:** ``DiamondAgent``, ``DDPM``, ``DiT``, ``DiffusionUNet``, ``EDMPreconditioner``, ``EulerSampler``, ``RewardTerminationModel``, and ``ActorCriticNetwork``.

DIAMOND exposes ``DiamondAgent`` from ``synora.training.train_diamond``; there is no separate ``DIAMONDAgent`` class name in the package.

.. automodule:: synora.models.diffusion
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.diffusion.DDPM
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.diffusion.DiT
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.diffusion.diamond_diffusion
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.diffusion.reward_termination
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.models.diffusion.actor_critic
   :members:
   :undoc-members:
   :show-inheritance:

Vision, tokenization, and layers
--------------------------------

**Key classes:** ``ConvEncoder``, ``ConvDecoder``, ``DenseDecoder``, ``ActionDecoder``, ``CNNEncoder``, ``CNNDecoder``, ``IRISEncoder``, ``IRISDecoder``, ``DiscreteAutoencoder``, ``VectorQuantizer``, ``VectorQuantizerEMA``, ``VideoTokenizer``, ``MultiHeadSelfAttention``, and ``STTransformer``.

.. automodule:: synora.vision.VAE.ConvVAE
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.vision.dreamer_encoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.vision.dreamer_decoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.vision.planet_encoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.vision.planet_decoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.vision.iris_encoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.vision.iris_decoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.vision.vq_layer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.vision.video_tokenizer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.blocks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.blocks.mhsa
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.blocks.st_transformer
   :members:
   :undoc-members:
   :show-inheritance:

Configuration objects
---------------------

.. automodule:: synora.configs
   :no-index:

.. automodule:: synora.configs.wm_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.configs.dreamer_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.configs.jepa_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.configs.iris_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.configs.genie_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.configs.dit_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.configs.diamond_config
   :members:
   :undoc-members:
   :show-inheritance:

Training entry points
---------------------

**Key classes and functions:** ``DiamondAgent``, ``train_diamond``, ``train_dreamer``, ``GenieTrainer``, ``IRISTrainer``, and related training entry points.

.. automodule:: synora.training
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.training.train_world_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.training.train_convvae
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.training.train_mdn_rnn
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.training.train_controller
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.training.train_jepa
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.training.train_iris
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.training.train_genie
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.training.train_planet
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.training.train_rssm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.training.train_diamond
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.training.rl_harness
   :members:
   :undoc-members:
   :show-inheritance:

Memory and controllers
----------------------

.. automodule:: synora.memory.dreamer_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.memory.planet_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.memory.iris_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.controller.rssm_policy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.controller.iris_policy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.controller.rollout_generator
   :members:
   :undoc-members:
   :show-inheritance:

Datasets, environments, and transforms
--------------------------------------

Environment adapters
~~~~~~~~~~~~~~~~~~~~

The environment APIs below mirror the dedicated environment guide pages: DMC,
DeepMind Lab, Gym/Gymnasium, Atari/ALE, Procgen, MuJoCo, Unity ML-Agents, and vectorization utilities.
DIAMOND-style Atari support is intentionally not listed as an environment
adapter because it is Atari preprocessing rather than a separate environment
family.

.. automodule:: synora.envs
   :no-index:

.. automodule:: synora.envs.dmc
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.envs.dmlab
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.envs.gym_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.envs.ale_atari_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.envs.ale_atari_vector_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.envs.procgen_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.envs.mujoco_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.envs.robotics_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.envs.unity_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.envs.vector_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.envs.wrappers
   :members:
   :undoc-members:
   :show-inheritance:

Atari preprocessing helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These helpers wrap Atari environments for specific training recipes. They are
not separate environment families.

.. automodule:: synora.envs.diamond_atari
   :members:
   :undoc-members:
   :show-inheritance:

Datasets and transforms
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: synora.datasets.wm_dataset
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.datasets.video_datasets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.datasets.tinyworlds
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.datasets.diamond_dataset
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.datasets.cifar10
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.datasets.imagenet1k
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.datasets.nuplan
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.transforms.image
   :members:
   :undoc-members:
   :show-inheritance:

Masking and JEPA helpers
------------------------

.. automodule:: synora.masks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.masks.default
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.masks.multiblock
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.masks.random
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.helpers.jepa_helper
   :members:
   :undoc-members:
   :show-inheritance:

Benchmarks and reports
----------------------

.. automodule:: synora.benchmarks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.benchmarks.runner
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.benchmarks.adapters
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.benchmarks.metrics
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.benchmarks.reporting
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: synora.losses.convae_loss
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.losses.gmm_loss
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.utils.train_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.utils.dreamer_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.utils.jepa_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.utils.data_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.utils.jit_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.utils.memory_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.utils.logging_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: synora.utils.utils
   :members:
   :undoc-members:
   :show-inheritance:
