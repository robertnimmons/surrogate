"""Training loop with early stopping, LR scheduling, and best-checkpoint saving."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from src.utils import sync


@dataclass
class TrainConfig:
    epochs: int = 1000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 50            # epochs without val improvement before stopping
    min_rel_improvement: float = 1e-2   # val loss must drop by 1% to count as an improvement
    lr_patience: int = 20         # epochs without improvement before halving LR
    min_lr: float = 1e-6
    log_every: int = 10


@dataclass
class TrainResult:
    history: dict[str, list[float]] = field(default_factory=lambda: {"train_loss": [], "val_loss": [], "lr": []})
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    train_seconds: float = 0.0
    stopped_early: bool = False


def train(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor,
          y_val: torch.Tensor, cfg: TrainConfig, checkpoint_path: Path, checkpoint_extra: dict,
          seed: int) -> TrainResult:
    """Train with Adam + MSE. All tensors are expected to already be on the training device.

    The whole dataset lives on the device and is sliced per batch. At this size
    (tens of thousands of rows, 7 floats each) that is a few MB and avoids the
    per-batch host-to-device copies a DataLoader would add.
    """
    device = x_train.device
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg.lr_patience, min_lr=cfg.min_lr)
    loss_fn = nn.MSELoss()
    gen = torch.Generator(device=device).manual_seed(seed)
    n = x_train.shape[0]
    result = TrainResult()
    since_best = 0

    print(f"{'epoch':>6} {'train_loss':>12} {'val_loss':>12} {'lr':>10}")
    sync(device)
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device, generator=gen)
        running = torch.zeros((), device=device)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            running += loss.detach() * idx.numel()
        train_loss = (running / n).item()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val).item()
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        result.history["train_loss"].append(train_loss)
        result.history["val_loss"].append(val_loss)
        result.history["lr"].append(lr)

        improved = val_loss < result.best_val_loss * (1 - cfg.min_rel_improvement)
        if improved:
            result.best_val_loss, result.best_epoch, since_best = val_loss, epoch, 0
            torch.save({**checkpoint_extra, "state_dict": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss}, checkpoint_path)
        else:
            since_best += 1

        if epoch == 1 or epoch % cfg.log_every == 0 or improved and epoch < 10:
            marker = "  *" if improved else ""
            print(f"{epoch:>6} {train_loss:>12.3e} {val_loss:>12.3e} {lr:>10.2e}{marker}")

        if since_best >= cfg.patience:
            result.stopped_early = True
            print(f"[train] early stop at epoch {epoch}: no val improvement for {cfg.patience} epochs")
            break

    sync(device)
    result.train_seconds = time.perf_counter() - t0
    return result
