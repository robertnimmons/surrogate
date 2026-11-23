"""Regression metrics, reported in physical units (after undoing normalisation)."""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, max absolute error, R^2, plus relative errors.

    Relative errors are included because the targets span orders of magnitude:
    a 5 MPa miss is negligible at 2000 MPa and serious at 10 MPa.
    """
    err = y_pred - y_true
    abs_err = np.abs(err)
    rel = abs_err / np.abs(y_true)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs_error": float(abs_err.max()),
        "r2": 1.0 - ss_res / ss_tot,
        "mean_rel_error_pct": float(100 * rel.mean()),
        "p99_rel_error_pct": float(100 * np.percentile(rel, 99)),
        "max_rel_error_pct": float(100 * rel.max()),
    }
