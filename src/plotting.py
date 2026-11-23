"""Result figures: loss curves and predicted-vs-actual parity plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: works in WSL2 and CI without a display
import matplotlib.pyplot as plt
import numpy as np

SERIES_1 = "#2a78d6"   # blue
SERIES_2 = "#eb6834"   # orange
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

PRETTY = {
    "max_von_mises_mpa": "Max von Mises stress (MPa)",
    "safety_factor": "Safety factor (Sy / σmax)",
}


def plot_loss_curves(history: dict[str, list[float]], best_epoch: int, path: Path) -> None:
    """Train and validation MSE (in normalised units) per epoch, log scale."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.plot(epochs, history["train_loss"], color=SERIES_1, lw=2, label="Train")
    ax.plot(epochs, history["val_loss"], color=SERIES_2, lw=2, label="Validation")
    ax.axvline(best_epoch, color=INK_MUTED, lw=1, ls="--")
    ax.annotate(f"best epoch {best_epoch}", xy=(best_epoch, ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", color=INK_MUTED, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (normalised targets)")
    ax.set_title("Training and validation loss", loc="left", color=INK)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str],
                r2: dict[str, float], log_axes: bool, path: Path) -> None:
    """Predicted vs. actual on the test set, one panel per target, with a y = x line."""
    n = len(target_names)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8), dpi=150, squeeze=False)
    for i, (ax, name) in enumerate(zip(axes[0], target_names)):
        t, p = y_true[:, i], y_pred[:, i]
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color=INK_MUTED, lw=1, ls="--", label="y = x", zorder=1)
        ax.scatter(t, p, s=9, color=SERIES_1, alpha=0.5, linewidths=0, label=f"test points (n = {len(t):,})",
                   zorder=2)
        if log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_aspect("equal", adjustable="box")
        label = PRETTY.get(name, name)
        ax.set_xlabel(f"Analytical: {label}")
        ax.set_ylabel(f"Surrogate: {label}")
        ax.set_title(f"{label}\nTest set, R² = {r2[name]:.5f}", loc="left", color=INK, fontsize=10)
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
