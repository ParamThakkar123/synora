import re

import pytest

import synora
from synora import api


def _missing_optional_dependency(exc):
    """True when ``exc`` is an extra that simply is not installed here.

    A base ``pip install synora`` has no gymnasium, ale_py, cv2 and so on, so
    the env-adapter exports legitimately fail to import. Those are not export
    map bugs. A ``ModuleNotFoundError`` naming a ``synora`` module *is* a bug -
    it means the map points somewhere that does not exist.
    """

    if not isinstance(exc, ModuleNotFoundError):
        return False
    # synora/utils/gym_compat.py re-raises with an explanatory message and no
    # ``name``, so a nameless miss is an optional backend rather than our code.
    if exc.name is None:
        return True
    return not exc.name.startswith("synora")


def _partition_exports(module):
    """Split ``__all__`` into (broken, skipped) by why each name failed."""

    broken, skipped = [], []
    for name in module.__all__:
        try:
            getattr(module, name)
        except Exception as exc:  # noqa: BLE001 - the failure mode is the point
            if _missing_optional_dependency(exc):
                skipped.append((name, exc.name))
            else:
                broken.append((name, f"{type(exc).__name__}: {exc}"))
    return broken, skipped


def test_every_public_export_resolves():
    # The lazy ``__getattr__`` export map means a typo (wrong module, renamed
    # symbol) stays invisible until a user touches that exact name.  Walk the
    # whole surface so the export map can never drift from the implementation.
    pytest.importorskip("torch")

    broken, _ = _partition_exports(synora)
    assert not broken, f"unresolvable synora exports: {broken}"


def test_export_check_is_not_vacuous():
    # If a future refactor made every export raise ModuleNotFoundError, the test
    # above would pass by skipping everything. Pin the torch-only core so the
    # sweep always has something real to check.
    pytest.importorskip("torch")

    core = [
        "create_config",
        "create_model",
        "make_env",
        "DreamerAgent",
        "ReplayBuffer",
        "ConvEncoder",
    ]
    for name in core:
        assert name in synora.__all__, f"{name} dropped out of the public API"
        getattr(synora, name)


def test_all_is_free_of_duplicates_and_exports_the_api_module():
    assert "api" in synora.__all__
    duplicates = {n for n in synora.__all__ if synora.__all__.count(n) > 1}
    assert not duplicates, f"duplicate names in synora.__all__: {sorted(duplicates)}"


def test_top_level_synora_exports_user_facing_factories():
    assert re.match(r"^\d+\.\d+\.\d+$", synora.__version__)
    assert synora.create_config is api.create_config
    assert "dreamer" in synora.list_models()
    assert "gym" in synora.list_env_backends()


def test_create_config_accepts_aliases_and_overrides():
    cfg = synora.create_config("dreamerv1", env="cartpole-swingup", seed=123)
    assert cfg.env == "cartpole-swingup"
    assert cfg.seed == 123


def test_model_and_backend_specs_resolve_aliases():
    assert synora.get_model_spec("i-jepa").name == "jepa"
    assert synora.get_env_backend_spec("gymnasium").name == "gym"
    assert synora.get_env_backend_spec("wm").name == "world-model"


def test_make_env_dispatches_to_selected_backend(monkeypatch):
    calls = {}

    def fake_loader(import_path):
        calls["import_path"] = import_path

        def factory(env_id, **kwargs):
            return {"env_id": env_id, "kwargs": kwargs}

        return factory

    monkeypatch.setattr(api, "_load_object", fake_loader)
    env = api.make_env("CartPole-v1", backend="gym", render_mode="rgb_array")

    assert calls["import_path"] == "synora.envs:make_gym_env"
    assert env == {
        "env_id": "CartPole-v1",
        "kwargs": {"render_mode": "rgb_array"},
    }


def test_create_model_for_factory_only_spec_filters_through_signature(monkeypatch):
    spec = api.ModelSpec(
        name="dummy",
        import_path="tests.dummy:create_dummy",
        description="Test-only factory",
    )

    def fake_loader(import_path):
        assert import_path == spec.import_path

        def create_dummy(required, optional=1):
            return {"required": required, "optional": optional}

        return create_dummy

    monkeypatch.setitem(api.MODEL_SPECS, "dummy", spec)
    monkeypatch.setattr(api, "_load_object", fake_loader)

    assert api.create_model("dummy", required=3, optional=5) == {
        "required": 3,
        "optional": 5,
    }


def test_make_env_dispatches_world_model_backend(monkeypatch):
    calls = {}

    def fake_loader(import_path):
        calls["import_path"] = import_path

        def factory(world_model, **kwargs):
            return {"world_model": world_model, "kwargs": kwargs}

        return factory

    model = object()
    monkeypatch.setattr(api, "_load_object", fake_loader)
    env = api.make_env(model, backend="wm", observation_space="obs", action_space="act")

    assert calls["import_path"] == "synora.envs:make_world_model_env"
    assert env == {
        "world_model": model,
        "kwargs": {"observation_space": "obs", "action_space": "act"},
    }


def test_export_model_torchscript_writes_file(tmp_path):
    import pytest

    torch = pytest.importorskip("torch")
    from synora import export_model

    class TinyAgent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

    path = export_model(
        TinyAgent(),
        tmp_path / "tiny.pt",
        format="torchscript",
        example_inputs=torch.zeros(1, 2),
    )

    assert path.exists()
    loaded = torch.jit.load(str(path))
    assert loaded(torch.zeros(1, 2)).shape == (1, 1)


def test_top_level_exports_export_helpers():
    import pytest
    import synora

    pytest.importorskip("torch")
    from synora.export import ExportableAgentMixin, export_any, export_model

    assert synora.export_any is export_any
    assert synora.export_model is export_model
    assert synora.ExportableAgentMixin is ExportableAgentMixin


def test_layer_and_helper_packages_are_importable():
    import synora.helpers as helpers
    from synora.layers import AdaLNNormalization, RMSNorm

    assert "load_checkpoint" in dir(helpers)
    assert RMSNorm.__name__ == "RMSNorm"
    assert AdaLNNormalization.__name__ == "AdaLNNormalization"


def test_synora_submodules_alias_synora():
    import synora.envs
    import synora.models
    import synora.utils.deprecation

    import synora.envs
    import synora.models
    import synora.utils.deprecation

    # The friendly ``synora.<name>`` surface resolves to the same module object
    # as the internal ``synora.<name>`` implementation.
    assert synora.models is synora.models
    assert synora.envs is synora.envs
    assert synora.utils.deprecation is synora.utils.deprecation
    # Canonical module identity stays on the internal package.
    assert synora.models.__name__ == "synora.models"


def test_synora_submodule_from_imports_resolve():
    from synora.envs import make_gym_env
    from synora.models import Dreamer
    from synora.utils.deprecation import deprecated

    assert Dreamer.__name__ == "Dreamer"
    assert callable(make_gym_env)
    assert callable(deprecated)


def test_synora_cli_is_the_real_submodule_not_an_alias():
    # ``synora.cli`` is a genuine module shipped in the ``synora`` package and
    # must not be shadowed by the ``synora`` alias finder.
    import synora.cli

    assert synora.cli.__name__ == "synora.cli"
    assert synora.cli.__file__.replace("\\", "/").endswith("synora/cli.py")


def test_diamond_and_dit_are_registered_in_public_api():
    assert "diamond" in synora.list_models()
    assert "dit" in synora.list_models()

    diamond_cfg = synora.create_config("diamond", game="Pong-v5", seed=11)
    dit_cfg = synora.create_config("diffusion-transformer", IMG_SIZE=8, PATCH=4)

    assert diamond_cfg.game == "Pong-v5"
    assert diamond_cfg.seed == 11
    assert dit_cfg.IMG_SIZE == 8
    assert dit_cfg.PATCH == 4
    assert synora.get_model_spec("diamond_agent").name == "diamond"
    assert synora.get_model_spec("diffusion_transformer").name == "dit"


def test_create_model_uses_dit_config_adapter():
    model = synora.create_model(
        "dit",
        IMG_SIZE=8,
        PATCH=4,
        CHANNELS=3,
        WIDTH=16,
        DEPTH=1,
        HEADS=4,
        DROP=0.0,
    )

    assert model.patchify.proj.in_channels == 3
    assert len(model.transformer_blocks) == 1


def test_create_model_dispatches_diamond_agent_with_config(monkeypatch):
    captured = {}

    class FakeDiamondAgent:
        def __init__(self, config):
            captured["config"] = config

    original_loader = api._load_object

    def fake_loader(import_path):
        if import_path == "synora.training.train_diamond:DiamondAgent":
            return FakeDiamondAgent
        return original_loader(import_path)

    monkeypatch.setattr(api, "_load_object", fake_loader)
    agent = api.create_model("diamond", game="Pong-v5")

    assert isinstance(agent, FakeDiamondAgent)
    assert captured["config"].game == "Pong-v5"
