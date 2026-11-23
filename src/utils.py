"""Reproducibility and device helpers."""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches and request deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def select_device(requested: str = "auto") -> torch.device:
    """Use CUDA when available; fall back to CPU with a visible warning."""
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        print(f"[device] {props.name}, {props.total_memory / 1024**3:.1f} GB VRAM")
        return dev
    if requested == "cuda":
        sys.exit("[device] --device cuda was requested but CUDA is not available. "
                 "Inside WSL2, check that `nvidia-smi` works and that torch was installed with CUDA support.")
    print("[device] WARNING: CUDA not available, training on CPU. "
          "Results are identical in quality, but timings will not reflect GPU performance.",
          file=sys.stderr)
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """Block until queued GPU work finishes, so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
