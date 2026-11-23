"""Dataset generation, CSV loading, splitting, and normalisation.

The pipeline is source-agnostic: everything downstream of `load_dataset` only
sees a DataFrame plus a list of input and target column names. Synthetic data
and a real CSV go through exactly the same split, scaler, and training code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import physics


@dataclass
class DataConfig:
    """Describes where the data comes from and which columns matter.

    For a real CSV, write a small JSON file with these keys and pass it to
    train.py via --data-config. See data/example_config.json.
    """

    source: str = "synthetic"                 # "synthetic" or "csv"
    path: str | None = None                   # CSV path when source == "csv"
    inputs: list[str] = field(default_factory=lambda: list(physics.INPUT_COLUMNS))
    targets: list[str] = field(default_factory=lambda: list(physics.TARGET_COLUMNS))
    # "log" trains on log(target). Stress spans several orders of magnitude, so
    # without it the loss is dominated by the largest values.
    target_transform: str = "log"

    @classmethod
    def from_json(cls, path: str | Path) -> "DataConfig":
        with open(path) as fh:
            cfg = cls(**json.load(fh))
        if cfg.source == "csv" and not cfg.path:
            raise ValueError(f"{path}: source is 'csv' but no 'path' was given")
        if cfg.target_transform not in ("log", "none"):
            raise ValueError(f"target_transform must be 'log' or 'none', got {cfg.target_transform!r}")
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


def generate_synthetic(n_samples: int, seed: int) -> pd.DataFrame:
    """Latin-hypercube sample of the design space, labelled with the closed-form solution.

    LHS covers the 5-D parameter space more evenly than plain uniform sampling for
    the same sample count, which matters for a fair test split.
    """
    rng = np.random.default_rng(seed)
    keys = list(physics.PARAM_RANGES)
    # One stratified permutation per dimension.
    u = (rng.permuted(np.tile(np.arange(n_samples), (len(keys), 1)), axis=1).T + rng.random((n_samples, len(keys))))
    u /= n_samples
    lo = np.array([physics.PARAM_RANGES[k][0] for k in keys])
    hi = np.array([physics.PARAM_RANGES[k][1] for k in keys])
    x = lo + u * (hi - lo)
    sample = dict(zip(keys, x.T))

    df = pd.DataFrame(
        {
            "plate_width_mm": sample["plate_width_mm"],
            "hole_diameter_mm": sample["hole_to_width_ratio"] * sample["plate_width_mm"],
            "thickness_mm": sample["thickness_mm"],
            "load_kn": sample["load_kn"],
            "yield_strength_mpa": sample["yield_strength_mpa"],
        }
    )
    sigma = physics.max_stress_mpa(df.plate_width_mm, df.hole_diameter_mm, df.thickness_mm, df.load_kn)
    df["max_von_mises_mpa"] = sigma
    df["safety_factor"] = df.yield_strength_mpa / sigma
    return df


def load_dataset(cfg: DataConfig, n_samples: int, seed: int) -> pd.DataFrame:
    """Return a clean DataFrame containing exactly cfg.inputs + cfg.targets."""
    if cfg.source == "synthetic":
        df = generate_synthetic(n_samples, seed)
    elif cfg.source == "csv":
        df = pd.read_csv(cfg.path)
    else:
        raise ValueError(f"Unknown data source {cfg.source!r}")

    missing = [c for c in cfg.inputs + cfg.targets if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}. Available: {list(df.columns)}")

    df = df[cfg.inputs + cfg.targets].apply(pd.to_numeric, errors="coerce")
    n_before = len(df)
    df = df.dropna()
    if cfg.target_transform == "log":
        df = df[(df[cfg.targets] > 0).all(axis=1)]
    if len(df) < n_before:
        print(f"[data] dropped {n_before - len(df)} rows with missing/non-numeric/non-positive values")
    if len(df) < 50:
        raise ValueError(f"Only {len(df)} usable rows; need at least 50 to train and validate")
    return df.reset_index(drop=True)


def split_indices(n: int, fractions: tuple[float, float, float], seed: int) -> dict[str, np.ndarray]:
    """Random train/val/test index split. Fractions must sum to 1."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"split fractions must sum to 1, got {fractions}")
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return {
        "train": perm[:n_train],
        "val": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:],
    }


@dataclass
class Scaler:
    """Standardises inputs and (optionally log-transformed) targets.

    Fitted on the training split only, then saved to JSON so predict.py applies
    the identical transform. Also records the training input range so inference
    can warn about extrapolation.
    """

    input_names: list[str]
    target_names: list[str]
    target_transform: str
    x_mean: list[float]
    x_std: list[float]
    y_mean: list[float]
    y_std: list[float]
    x_min: list[float]
    x_max: list[float]

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, input_names: list[str], target_names: list[str],
            target_transform: str) -> "Scaler":
        y_t = np.log(y) if target_transform == "log" else y
        return cls(
            input_names=input_names,
            target_names=target_names,
            target_transform=target_transform,
            x_mean=x.mean(0).tolist(),
            x_std=np.maximum(x.std(0), 1e-12).tolist(),
            y_mean=y_t.mean(0).tolist(),
            y_std=np.maximum(y_t.std(0), 1e-12).tolist(),
            x_min=x.min(0).tolist(),
            x_max=x.max(0).tolist(),
        )

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        return (x - np.asarray(self.x_mean)) / np.asarray(self.x_std)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        y_t = np.log(y) if self.target_transform == "log" else y
        return (y_t - np.asarray(self.y_mean)) / np.asarray(self.y_std)

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        y_t = y_scaled * np.asarray(self.y_std) + np.asarray(self.y_mean)
        return np.exp(y_t) if self.target_transform == "log" else y_t

    def out_of_range(self, x: np.ndarray) -> list[str]:
        """Names of inputs outside the range seen in training (for a single row)."""
        lo, hi = np.asarray(self.x_min), np.asarray(self.x_max)
        return [n for n, v, a, b in zip(self.input_names, x, lo, hi) if v < a or v > b]

    def save(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Scaler":
        with open(path) as fh:
            return cls(**json.load(fh))
