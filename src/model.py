"""The surrogate network: a small fully connected MLP."""

from __future__ import annotations

import torch
from torch import nn


class SurrogateMLP(nn.Module):
    """Maps normalised design parameters to normalised outputs.

    Default: 5 inputs -> 3 x 128 GELU -> 2 outputs, about 35k parameters. That is
    roughly 0.14 MB of weights, so even a large batch is nowhere near 6 GB of VRAM.
    """

    def __init__(self, n_inputs: int, n_outputs: int, hidden: int = 128, n_layers: int = 3,
                 dropout: float = 0.0, activation: str = "gelu") -> None:
        super().__init__()
        act: type[nn.Module] = {"gelu": nn.GELU, "relu": nn.ReLU}[activation]
        layers: list[nn.Module] = []
        width_in = n_inputs
        for _ in range(n_layers):
            layers += [nn.Linear(width_in, hidden), act()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width_in = hidden
        layers.append(nn.Linear(width_in, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
