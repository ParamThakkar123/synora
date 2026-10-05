"""IRIS: a discrete autoencoder + Transformer world model, and a policy trained in it.

    python demos/iris_demo.py train  --minutes 100
    python demos/iris_demo.py record --run demos/runs/iris

The paper configuration is used unchanged. Only the amount of work per epoch
and the batch sizes are reduced so that a run fits a laptop GPU; the values are
written to ``<run>/args.json``.

``record`` writes into ``<run>/media``:

* ``tokens.mp4`` -- real frames beside their reconstruction from 16 discrete
  tokens, the only thing the world model ever sees.
* ``dream.mp4`` -- the policy acting inside the Transformer's imagination,
  starting from a real frame.
* ``policy.mp4`` -- the policy playing the real game.
* ``curve.png`` -- autoencoder and world-model losses over training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]
from _media import hstack, label, plot_curves, read_metrics, upscale, write_mp4  # noqa: E402

# Per-epoch work and batch sizes, scaled down for a 4 GB GPU. Architecture,
# tokenizer, horizon and losses stay at the paper's values.
RUNTIME_OVERRIDES = dict(
    autoencoder_batch_size=64,
    transformer_batch_size=16,
    actor_critic_batch_size=16,
    training_steps_per_epoch=100,
    transformer_steps_per_epoch=50,
    actor_critic_steps_per_epoch=20,
    start_transformer_after=10,
    start_actor_critic_after=20,
)


def train(args: argparse.Namespace) -> None:
    from synora.configs.iris_config import IRISConfig
    from synora.training.train_iris import IRISTrainer

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(
        json.dumps({**vars(args), "overrides": RUNTIME_OVERRIDES}, indent=2)
    )
    config = IRISConfig()
    for key, value in RUNTIME_OVERRIDES.items():
        setattr(config, key, value)
    trainer = IRISTrainer(game=args.game, device="cuda", seed=args.seed, config=config)

    deadline = time.time() + args.minutes * 60
    epoch = 0
    while time.time() < deadline:
        started = time.time()
        metrics = trainer.train_epoch(epoch)
        row = {"epoch": epoch, "seconds": time.time() - started}
        row.update(
            {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        )
        with (run / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        epoch += 1
        if epoch % 5 == 0:
            trainer.agent.save(str(run / "iris.pt"))
    trainer.agent.save(str(run / "iris.pt"))


def record(args: argparse.Namespace) -> None:
    import cv2
    import torch

    from synora.configs.iris_config import IRISConfig
    from synora.envs.ale_atari_env import make_atari_env
    from synora.models.iris_agent import IRISAgent

    run = Path(args.run)
    out = run / "media"
    saved = json.loads((run / "args.json").read_text())
    ckpt = torch.load(run / "iris.pt", map_location="cpu", weights_only=True)
    config = IRISConfig.from_dict(ckpt["config"])
    config.perceptual_weight = 0.0  # skip building VGG16; it holds no saved weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = IRISAgent(config, action_size=ckpt["action_size"], device=device)
    agent.load(str(run / "iris.pt"))
    agent.eval()

    # The same emulator settings the trainer collects with.
    env = make_atari_env(
        saved["game"],
        obs_type="rgb",
        frameskip=config.action_repeat,
        repeat_action_probability=config.repeat_action_probability,
        max_episode_steps=config.max_episode_steps,
    )

    def to_input(raw: np.ndarray) -> torch.Tensor:
        size = (config.frame_width, config.frame_height)
        small = cv2.resize(np.asarray(raw), size, interpolation=cv2.INTER_LINEAR)
        return (
            torch.from_numpy(small).permute(2, 0, 1).float().div(255.0)[None].to(device)
        )

    def show(x: torch.Tensor) -> np.ndarray:
        return x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    raw, _ = env.reset(seed=args.seed)
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    score = 0.0
    policy_frames, token_frames, history = [], [], []
    for step in range(args.policy_steps):
        x = to_input(raw)
        with torch.no_grad():
            recon = agent.reconstruct(x)
            out_act = agent.act(recon, hidden=hidden, return_hidden=True)
            assert isinstance(out_act, tuple)
            action, hidden = out_act
        history.append(recon)
        policy_frames.append(label(upscale(np.asarray(raw), 2), f"return {score:+.0f}"))
        if step < args.token_steps:
            token_frames.append(
                hstack(
                    [
                        label(upscale(show(x), 5), "Real frame (64x64)"),
                        label(
                            upscale(show(recon), 5),
                            "Decoded from 16 tokens",
                            accent=True,
                        ),
                    ]
                )
            )
        raw, reward, terminated, truncated, _ = env.step(int(action.item()))
        score += float(reward)
        if terminated or truncated:
            break
    print(f"real game: {len(policy_frames)} steps, return {score:+.0f}")
    print("wrote", write_mp4(policy_frames, out / "policy.mp4", fps=args.fps))
    print("wrote", write_mp4(token_frames, out / "tokens.mp4", fps=args.fps))

    # Imagination from a real frame partway through the episode, with the
    # preceding 20 reconstructed frames as burn-in for the policy's LSTM.
    start = min(len(history) - 1, args.dream_start)
    burn_in = (
        torch.cat(history[max(0, start - 20) : start], dim=0)[None] if start else None
    )
    torch.manual_seed(args.seed)
    with torch.no_grad():
        traj = agent.imagine_rollout(
            history[start],
            horizon=args.dream_steps,
            burn_in_frames=burn_in,
            stop_on_termination=False,
        )
    dream = traj["frames"][0]
    rewards = traj["rewards"][0].cpu().numpy()
    frames, total = [], 0.0
    for t in range(dream.shape[0]):
        if t:
            total += float(rewards[t - 1])
        frames.append(
            label(
                upscale(dream[t].clamp(0, 1).permute(1, 2, 0).cpu().numpy(), 5),
                f"Imagined step {t}  reward {total:+.1f}",
                accent=True,
            )
        )
    print("wrote", write_mp4(frames, out / "dream.mp4", fps=args.fps))

    rows = read_metrics(run / "metrics.jsonl")
    series = {}
    for key, name in (
        ("recon_loss", "autoencoder recon"),
        ("token_loss", "next-token loss"),
    ):
        mine = [r for r in rows if r.get(key)]
        if mine:
            series[name] = ([r["epoch"] for r in mine], [r[key] for r in mine])
    plot_curves(
        series,
        out / "curve.png",
        title=f"IRIS on {saved['game']}",
        xlabel="epoch",
        ylabel="loss",
    )
    print("wrote", out / "curve.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--game", default="ALE/Pong-v5")
    t.add_argument("--minutes", type=float, default=100)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--run", default="demos/runs/iris")

    r = sub.add_parser("record")
    r.add_argument("--run", default="demos/runs/iris")
    r.add_argument("--policy-steps", type=int, default=600)
    r.add_argument("--token-steps", type=int, default=120)
    r.add_argument("--dream-start", type=int, default=60)
    r.add_argument("--dream-steps", type=int, default=80)
    r.add_argument("--fps", type=int, default=15)
    r.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()
    {"train": train, "record": record}[args.command](args)


if __name__ == "__main__":
    main()
