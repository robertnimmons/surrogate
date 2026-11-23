"""Load a trained surrogate and run predictions. Shared by train.py (timing) and predict.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data import Scaler
from src.model import SurrogateMLP


class Surrogate:
    """A trained model plus the scaler it was trained with."""

    def __init__(self, model: SurrogateMLP, scaler: Scaler, device: torch.device, meta: dict) -> None:
        self.model = model.eval()
        self.scaler = scaler
        self.device = device
        self.meta = meta
        # Cache the scaler arrays as tensors so single predictions avoid repeated conversion.
        self._x_mean = torch.tensor(scaler.x_mean, dtype=torch.float32, device=device)
        self._x_std = torch.tensor(scaler.x_std, dtype=torch.float32, device=device)

    @classmethod
    def load(cls, model_dir: str | Path, device: torch.device | None = None) -> "Surrogate":
        model_dir = Path(model_dir)
        device = device or torch.device("cpu")
        ckpt = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
        scaler = Scaler.load(model_dir / "scaler.json")
        model = SurrogateMLP(**ckpt["model_kwargs"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
        return cls(model, scaler, device, meta)

    @property
    def input_names(self) -> list[str]:
        return self.scaler.input_names

    @property
    def target_names(self) -> list[str]:
        return self.scaler.target_names

    @torch.no_grad()
    def predict_array(self, x: np.ndarray) -> np.ndarray:
        """Predict for an (n, n_inputs) array of raw (unscaled) inputs; returns physical units."""
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y_scaled = self.model((xt - self._x_mean) / self._x_std)
        return self.scaler.inverse_y(y_scaled.cpu().numpy().astype(np.float64))

    def predict_one(self, row: dict[str, float]) -> dict[str, float]:
        """Predict for one design point given as {input_name: value}."""
        x = np.array([[row[name] for name in self.input_names]])
        y = self.predict_array(x)[0]
        return dict(zip(self.target_names, map(float, y)))
