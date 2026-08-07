#!/usr/bin/env python3
"""Supervised warm-start from Lichess replay shards (Chess RL v2)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import PolicyNetwork
from replay_buffer import ReplayBuffer
from train import load_checkpoint, save_checkpoint
from training_config import DEFAULT_CONFIG_PATH, load_training_config
from training_runtime import (
    apply_lr_schedule,
    eval_batch,
    make_grad_scaler,
    seed_everything,
    train_batch,
)

RL_DIR = Path(__file__).resolve().parent


def load_data(path: Path, replay_buffer: ReplayBuffer):
    """Load Lichess SL shards from `path` (file or directory) into `replay_buffer`."""
    path = Path(path)
    if path.is_file():
        shard_paths = [path]
    else:
        shard_paths = sorted(path.glob("shard_*.pt"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.pt files under {path}")

    for shard_path in shard_paths:
        state = torch.load(shard_path, map_location="cpu", weights_only=False)
        if "boards" not in state:
            continue
        replay_buffer.load_state_dict(state)

    return replay_buffer


def split_replay_buffer(source: ReplayBuffer, val_fraction: float, seed: int):
    """Partition `source` into train/val ReplayBuffers (same capacity)."""
    n = len(source)
    if n == 0:
        raise ValueError("Replay buffer is empty; build shards with prepare_lichess_dataset.py")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = int(n * val_fraction)
    val_idxs = set(indices[:n_val])
    train_idxs = indices[n_val:]

    train_buffer = ReplayBuffer(capacity=source.capacity)
    val_buffer = ReplayBuffer(capacity=source.capacity)

    def _take(idxs):
        boards = [source.boards[i] for i in idxs]
        pis = [source.pis[i] for i in idxs]
        outcomes = [source.outcomes[i] for i in idxs]
        weights = [source.value_weights[i] for i in idxs]
        return boards, pis, outcomes, weights

    if train_idxs:
        train_buffer.add(*_take(train_idxs))
    if val_idxs:
        val_buffer.add(*_take(sorted(val_idxs)))
    return train_buffer, val_buffer


def mean_losses(loss_dicts):
    if not loss_dicts:
        return {"total": float("nan"), "policy": float("nan"), "value": float("nan")}
    keys = ("total", "policy", "value")
    return {key: sum(item[key] for item in loss_dicts) / len(loss_dicts) for key in keys}


def evaluate_split(policy, buffer, batch_size, value_coef, device, amp_enabled):
    if len(buffer) == 0:
        return {"total": float("nan"), "policy": float("nan"), "value": float("nan")}
    policy.eval()
    losses = []
    n_batches = max(1, len(buffer) // batch_size)
    for _ in range(n_batches):
        positions, policies, outcomes, weights = buffer.sample(batch_size)
        losses.append(
            eval_batch(
                policy,
                positions,
                policies,
                outcomes,
                weights,
                value_coef,
                device,
                amp_enabled,
            )
        )
    return mean_losses(losses)


def save_loss_plot(history, plot_path: Path):
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in history]
    train_total = [row["train_total"] for row in history]
    val_total = [row["val_total"] for row in history]
    train_policy = [row["train_policy"] for row in history]
    val_policy = [row["val_policy"] for row in history]
    train_value = [row["train_value"] for row in history]
    val_value = [row["val_value"] for row in history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    series = (
        (axes[0], "Total loss", train_total, val_total),
        (axes[1], "Policy loss", train_policy, val_policy),
        (axes[2], "Value loss", train_value, val_value),
    )
    for ax, title, train_y, val_y in series:
        ax.plot(epochs, train_y, label="train", marker="o", markersize=3)
        ax.plot(epochs, val_y, label="val", marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Supervised training loss")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Loss plot saved to {plot_path}")


def train_supervised(
    policy,
    optimizer,
    train_buffer,
    val_buffer,
    device,
    num_epochs,
    train_batch_size,
    batches_per_epoch,
    value_coef,
    base_lr,
    min_lr,
    warmup_epochs,
    amp_enabled,
    scaler,
    checkpoint_dir,
    checkpoint_every,
    checkpoint_keep,
    start_epoch=0,
    resolved_config=None,
    plot_path=None,
    model_out=None,
):
    policy.to(device)
    scaler = scaler or make_grad_scaler(amp_enabled)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resolved_config:
        (ckpt_dir / "resolved_config.json").write_text(
            json.dumps(resolved_config, indent=2, default=str), encoding="utf-8"
        )

    if batches_per_epoch is None:
        batches_per_epoch = max(1, len(train_buffer) // train_batch_size)

    history = []
    for epoch in range(start_epoch + 1, num_epochs + 1):
        policy.train()
        lr = apply_lr_schedule(
            optimizer,
            base_lr,
            min_lr,
            warmup_epochs,
            num_epochs,
            epoch,
        )
        train_losses = []
        for _ in range(batches_per_epoch):
            positions, policies, outcomes, weights = train_buffer.sample(train_batch_size)
            train_losses.append(
                train_batch(
                    policy,
                    optimizer,
                    scaler,
                    positions,
                    policies,
                    outcomes,
                    weights,
                    value_coef,
                    device,
                    amp_enabled,
                )
            )

        train_avg = mean_losses(train_losses)
        val_avg = evaluate_split(
            policy, val_buffer, train_batch_size, value_coef, device, amp_enabled
        )
        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_total": train_avg["total"],
                "train_policy": train_avg["policy"],
                "train_value": train_avg["value"],
                "val_total": val_avg["total"],
                "val_policy": val_avg["policy"],
                "val_value": val_avg["value"],
            }
        )
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train={train_avg['total']:.4f} "
            f"(p={train_avg['policy']:.4f}, v={train_avg['value']:.4f}) | "
            f"val={val_avg['total']:.4f} "
            f"(p={val_avg['policy']:.4f}, v={val_avg['value']:.4f}) | "
            f"lr={lr:.2e}"
        )

        if checkpoint_every and epoch % checkpoint_every == 0:
            for checkpoint_path in (
                ckpt_dir / f"ckpt_epoch_{epoch:04d}.pt",
                ckpt_dir / "latest.pt",
            ):
                save_checkpoint(
                    checkpoint_path,
                    policy,
                    optimizer,
                    games_done=epoch,
                    results_window=[],
                    scaler=scaler,
                    resolved_config=resolved_config,
                )
            numbered = sorted(ckpt_dir.glob("ckpt_epoch_*.pt"))
            for stale in numbered[:-checkpoint_keep]:
                stale.unlink()
            print(f"  ↳ checkpoint saved at epoch {epoch}")

    save_checkpoint(
        ckpt_dir / "latest.pt",
        policy,
        optimizer,
        games_done=num_epochs,
        results_window=[],
        scaler=scaler,
        resolved_config=resolved_config,
    )
    (ckpt_dir / "loss_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    if plot_path is not None and history:
        save_loss_plot(history, Path(plot_path))

    if model_out is not None:
        model_path = Path(model_out)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {**policy.checkpoint_metadata(), "policy_state_dict": policy.state_dict()},
            model_path,
        )
        print(f"Final model weights saved to {model_out}")

    return policy, history


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML training configuration file",
    )
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Supervised warm-start from Lichess shards.",
        parents=[config_parser],
    )
    parser.add_argument("--data-dir", type=str, default=str(RL_DIR / "data" / "lichess_sl"))
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=None,
        help="gradient steps per epoch (default: len(train) // batch_size)",
    )
    parser.add_argument("--buffer-capacity", type=int, default=1_000_000)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument(
        "--warmup-games",
        type=int,
        default=1,
        help="warmup epochs (reuses the optimizer.warmup_games config key)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--value-head-channels", type=int, default=32)
    parser.add_argument("--value-hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument(
        "--checkpoint-dir", type=str, default=str(RL_DIR / "checkpoints_sl")
    )
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--model-out", type=str, default=str(RL_DIR / "chess_sl_model.pt")
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=str(RL_DIR / "checkpoints_sl" / "loss_curve.png"),
    )
    # Keys present in shared YAML sections but unused by supervised training.
    parser.add_argument("--min-buffer-size", type=int, default=0)
    parser.add_argument("--replay-path", type=str, default=None)
    parser.add_argument("--replay-save-every", type=int, default=0)

    try:
        config_defaults = load_training_config(
            config_args.config, "supervised", include_common=False
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))

    valid_keys = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(config_defaults) - valid_keys)
    if unknown_keys:
        parser.error(f"unknown supervised config keys: {', '.join(unknown_keys)}")
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()

    if args.num_epochs < 1:
        parser.error("--num-epochs must be at least 1")
    if not 0.0 <= args.val_fraction < 1.0:
        parser.error("--val-fraction must be in [0, 1)")
    if args.train_batch_size < 1:
        parser.error("--train-batch-size must be at least 1")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    policy = PolicyNetwork(
        channels=args.channels,
        num_blocks=args.num_blocks,
        value_head_channels=args.value_head_channels,
        value_hidden=args.value_hidden,
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    amp_enabled = not args.disable_amp
    scaler = make_grad_scaler(amp_enabled and device.type == "cuda")

    start_epoch = 0
    if args.resume:
        meta = load_checkpoint(
            args.resume, policy, optimizer, device=device, scaler=scaler
        )
        start_epoch = meta["games_done"]
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity)
    load_data(Path(args.data_dir), replay_buffer)
    print(f"Loaded {len(replay_buffer):,} positions from {args.data_dir}")

    train_buffer, val_buffer = split_replay_buffer(
        replay_buffer, args.val_fraction, seed=args.seed
    )
    print(f"Train={len(train_buffer):,}  Val={len(val_buffer):,}")

    train_supervised(
        policy,
        optimizer,
        train_buffer,
        val_buffer,
        device=device,
        num_epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        batches_per_epoch=args.batches_per_epoch,
        value_coef=args.value_coef,
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_games,
        amp_enabled=amp_enabled,
        scaler=scaler,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        checkpoint_keep=args.checkpoint_keep,
        start_epoch=start_epoch,
        resolved_config=vars(args),
        plot_path=args.plot_path,
        model_out=args.model_out,
    )


if __name__ == "__main__":
    main()
