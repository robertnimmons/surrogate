"""Minimal plane-stress finite element solver for the plate-with-hole problem.

This exists for two reasons:

1. To check the analytical ground truth independently. If a from-scratch FE
   model agrees with the Peterson Kt curve, the synthetic labels are defensible.
2. To give the "speedup vs. simulation" number a real, measured baseline
   instead of a guessed one.

Model: a quarter of the plate (two planes of symmetry), 4-node bilinear
quadrilaterals (Q4) with 2x2 Gauss integration, linear elastic, plane stress.
The mesh is mapped: straight rays run from the hole edge to the outer
boundary, graded geometrically so elements are small near the hole where the
stress gradient is steep.

Coordinates: x across the width, y along the length (load direction). The
peak stress is sigma_yy at (x = d/2, y = 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_GAUSS = 1.0 / np.sqrt(3.0)
_GAUSS_POINTS = [(-_GAUSS, -_GAUSS), (_GAUSS, -_GAUSS), (_GAUSS, _GAUSS), (-_GAUSS, _GAUSS)]


@dataclass
class FEResult:
    """Output of one FE solve."""

    max_stress_mpa: float
    n_dof: int


def _shape_derivatives(xi: float, eta: float) -> np.ndarray:
    """dN/d(xi, eta) for the Q4 element, shape (2, 4). Node order: (-1,-1),(1,-1),(1,1),(-1,1)."""
    return 0.25 * np.array(
        [
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
        ]
    )


def _b_matrices(elem_xy: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Strain-displacement matrices for every element at one natural point.

    elem_xy: (n_elem, 4, 2) nodal coordinates. Returns B (n_elem, 3, 8) and det(J) (n_elem,).
    """
    dn = _shape_derivatives(xi, eta)                        # (2, 4)
    jac = np.einsum("ak,eki->eai", dn, elem_xy)              # (n_elem, 2, 2)
    det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
    inv = np.empty_like(jac)
    inv[:, 0, 0] = jac[:, 1, 1] / det
    inv[:, 0, 1] = -jac[:, 0, 1] / det
    inv[:, 1, 0] = -jac[:, 1, 0] / det
    inv[:, 1, 1] = jac[:, 0, 0] / det
    dn_xy = np.einsum("eia,ak->eik", inv, dn)                # (n_elem, 2, 4): dN/dx, dN/dy
    b = np.zeros((elem_xy.shape[0], 3, 8))
    b[:, 0, 0::2] = dn_xy[:, 0]
    b[:, 1, 1::2] = dn_xy[:, 1]
    b[:, 2, 0::2] = dn_xy[:, 1]
    b[:, 2, 1::2] = dn_xy[:, 0]
    return b, det


def _build_mesh(
    half_width: float, half_length: float, radius: float, n_theta: int, n_radial: int, growth: float
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Mapped quarter-plate mesh. Node (i, j): i radial (0 = hole edge), j angular (0 = theta 0)."""
    theta_corner = np.arctan2(half_length, half_width)
    # Split angular divisions between the right edge and the top edge so the corner is a node.
    n_right = max(2, int(round(n_theta * theta_corner / (np.pi / 2))))
    n_top = max(2, n_theta - n_right)
    thetas = np.concatenate(
        [np.linspace(0.0, theta_corner, n_right + 1), np.linspace(theta_corner, np.pi / 2, n_top + 1)[1:]]
    )
    hole = np.stack([radius * np.cos(thetas), radius * np.sin(thetas)], axis=1)
    outer = np.empty_like(hole)
    for j, th in enumerate(thetas):
        if th <= theta_corner:
            outer[j] = (half_width, half_width * np.tan(th))
        else:
            outer[j] = (half_length / np.tan(th) if th < np.pi / 2 else 0.0, half_length)

    # Geometric grading along each ray: s in [0, 1], small steps near the hole.
    steps = growth ** np.arange(n_radial)
    s = np.concatenate([[0.0], np.cumsum(steps) / steps.sum()])
    nodes = hole[None, :, :] + s[:, None, None] * (outer - hole)[None, :, :]   # (n_r+1, n_t+1, 2)
    n_r, n_t = n_radial + 1, len(thetas)
    nodes = nodes.reshape(-1, 2)

    def nid(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        return i * n_t + j

    ii, jj = np.meshgrid(np.arange(n_radial), np.arange(n_t - 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    elems = np.stack([nid(ii, jj), nid(ii + 1, jj), nid(ii + 1, jj + 1), nid(ii, jj + 1)], axis=1)
    return nodes, elems, n_r, n_t


def solve_plate_with_hole(
    plate_width_mm: float,
    hole_diameter_mm: float,
    thickness_mm: float,
    load_kn: float,
    youngs_modulus_mpa: float = 200_000.0,
    poisson: float = 0.3,
    length_to_width: float = 4.0,
    n_theta: int = 64,
    n_radial: int = 48,
    growth: float = 1.08,
) -> FEResult:
    """Solve the quarter-symmetric plate and return the peak hole-edge stress.

    Peak stress is linear in load and independent of E for a single material, so E and nu
    only matter for the displacement field; defaults are steel.
    """
    half_w = plate_width_mm / 2.0
    half_l = length_to_width * plate_width_mm / 2.0
    radius = hole_diameter_mm / 2.0
    nodes, elems, n_r, n_t = _build_mesh(half_w, half_l, radius, n_theta, n_radial, growth)
    n_dof = 2 * nodes.shape[0]

    c = youngs_modulus_mpa / (1.0 - poisson**2)
    d_mat = c * np.array([[1.0, poisson, 0.0], [poisson, 1.0, 0.0], [0.0, 0.0, (1.0 - poisson) / 2.0]])

    elem_xy = nodes[elems]                                   # (n_elem, 4, 2)
    ke = np.zeros((elems.shape[0], 8, 8))
    for xi, eta in _GAUSS_POINTS:
        b, det = _b_matrices(elem_xy, xi, eta)
        ke += np.einsum("eia,ij,ejb->eab", b, d_mat, b) * (det * thickness_mm)[:, None, None]

    dofs = np.empty((elems.shape[0], 8), dtype=np.int64)
    dofs[:, 0::2] = 2 * elems
    dofs[:, 1::2] = 2 * elems + 1
    rows = np.repeat(dofs, 8, axis=1).ravel()
    cols = np.tile(dofs, (1, 8)).ravel()
    k = sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(n_dof, n_dof)).tocsr()

    # Traction on the top edge (y = half_l): outer-boundary nodes past the corner.
    f = np.zeros(n_dof)
    gross_stress = load_kn * 1_000.0 / (plate_width_mm * thickness_mm)
    outer_ids = (n_r - 1) * n_t + np.arange(n_t)
    top = outer_ids[np.isclose(nodes[outer_ids, 1], half_l)]
    top = top[np.argsort(nodes[top, 0])]
    for a, b_ in zip(top[:-1], top[1:]):
        edge_force = gross_stress * thickness_mm * abs(nodes[b_, 0] - nodes[a, 0]) / 2.0
        f[2 * a + 1] += edge_force
        f[2 * b_ + 1] += edge_force

    # Symmetry: u_y = 0 on y = 0 (j = 0), u_x = 0 on x = 0 (j = last).
    j0 = np.arange(n_r) * n_t
    j_last = np.arange(n_r) * n_t + (n_t - 1)
    fixed = np.concatenate([2 * j0 + 1, 2 * j_last])
    free = np.setdiff1d(np.arange(n_dof), fixed)

    u = np.zeros(n_dof)
    u[free] = spla.spsolve(k[free][:, free].tocsc(), f[free])

    # Peak stress: element touching node (0, 0), evaluated at Gauss points and
    # extrapolated bilinearly to the corner at natural coordinates (-1, -1).
    e0 = 0
    sig_yy = []
    for xi, eta in _GAUSS_POINTS:
        b, _ = _b_matrices(elem_xy[e0:e0 + 1], xi, eta)
        sig_yy.append((d_mat @ b[0] @ u[dofs[e0]])[1])
    sig_yy = np.array(sig_yy)
    # Extrapolate from Gauss points (at +/- 1/sqrt(3)) to node at (-1, -1): scaled coords -sqrt(3).
    r = -np.sqrt(3.0)
    shape_at_node = 0.25 * np.array([(1 - r) * (1 - r), (1 + r) * (1 - r), (1 + r) * (1 + r), (1 - r) * (1 + r)])
    peak = float(shape_at_node @ sig_yy)
    return FEResult(max_stress_mpa=peak, n_dof=n_dof)
