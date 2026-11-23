"""Analytical ground truth: a finite-width plate with a central circular hole in tension.

The plate has width W, thickness t, and a through hole of diameter d at its
centre. An axial load P is applied along the plate length. The peak stress
sits at the edge of the hole on the net section, and is

    sigma_max = Kt_n * sigma_net,     sigma_net = P / ((W - d) * t)

where Kt_n is the net-section stress concentration factor. We use the curve
fit to Howland's elasticity solution given in Peterson's Stress Concentration
Factors (Pilkey & Pilkey, 3rd ed., Chart 4.1):

    Kt_n = 3.00 - 3.13 (d/W) + 3.66 (d/W)^2 - 1.53 (d/W)^3,   0 <= d/W <= 1

At the hole edge the stress state is uniaxial (the hoop stress is the only
non-zero component on a traction-free boundary), so sigma_max is also the
maximum von Mises stress. The factor of safety against first yield is
Sy / sigma_max. This is linear-elastic: values of sigma_max above Sy mean the
real part would yield locally, which this model does not capture.
"""

from __future__ import annotations

import numpy as np

# Columns produced by the synthetic generator, in the order the model sees them.
INPUT_COLUMNS: list[str] = [
    "plate_width_mm",
    "hole_diameter_mm",
    "thickness_mm",
    "load_kn",
    "yield_strength_mpa",
]
TARGET_COLUMNS: list[str] = ["max_von_mises_mpa", "safety_factor"]

# Sampling ranges for the synthetic design space. d/W is sampled rather than d
# directly so that every sample is a valid geometry (hole narrower than plate).
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "plate_width_mm": (40.0, 200.0),
    "hole_to_width_ratio": (0.10, 0.60),
    "thickness_mm": (3.0, 20.0),
    "load_kn": (5.0, 100.0),
    # Roughly 6061-T6 aluminium up to high-strength alloy steel.
    "yield_strength_mpa": (250.0, 1100.0),
}


def kt_net(d_over_w: np.ndarray | float) -> np.ndarray | float:
    """Net-section stress concentration factor (Peterson Chart 4.1 fit)."""
    r = d_over_w
    return 3.00 - 3.13 * r + 3.66 * r**2 - 1.53 * r**3


def max_stress_mpa(
    plate_width_mm: np.ndarray | float,
    hole_diameter_mm: np.ndarray | float,
    thickness_mm: np.ndarray | float,
    load_kn: np.ndarray | float,
) -> np.ndarray | float:
    """Peak (von Mises) stress at the hole edge in MPa. N/mm^2 == MPa."""
    net_area_mm2 = (plate_width_mm - hole_diameter_mm) * thickness_mm
    sigma_net = load_kn * 1_000.0 / net_area_mm2
    return kt_net(hole_diameter_mm / plate_width_mm) * sigma_net


def analytical_targets(row: dict[str, float]) -> dict[str, float]:
    """Ground-truth targets for one design point given as a column->value dict."""
    sigma = float(
        max_stress_mpa(
            row["plate_width_mm"],
            row["hole_diameter_mm"],
            row["thickness_mm"],
            row["load_kn"],
        )
    )
    return {
        "max_von_mises_mpa": sigma,
        "safety_factor": row["yield_strength_mpa"] / sigma,
    }


def validate_geometry(row: dict[str, float]) -> None:
    """Raise ValueError for inputs where the closed-form solution is meaningless."""
    for key in ("plate_width_mm", "hole_diameter_mm", "thickness_mm", "load_kn", "yield_strength_mpa"):
        if row[key] <= 0:
            raise ValueError(f"{key} must be positive, got {row[key]}")
    if row["hole_diameter_mm"] >= row["plate_width_mm"]:
        raise ValueError("hole_diameter_mm must be smaller than plate_width_mm")
