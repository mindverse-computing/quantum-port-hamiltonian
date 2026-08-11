"""
network/pendulum_chain.py
=========================
Coupled nonlinear pendulums with an on-site restoring term.

The network model already in this codebase (``network.data_generator``) is the
second-order Kuramoto system

    phi_dot_i = omega_i
    omega_dot_i = -sum_j K_ij sin(phi_i - phi_j) - gamma_i omega_i

whose coupling depends only on phase *differences*.  Its Hamiltonian is
therefore invariant under the global shift ``phi_i -> phi_i + c``: total
momentum is conserved and the system carries a zero mode.

This module adds the term that breaks that symmetry -- gravity acting on each
rotor:

    H = 1/2 sum_i omega_i^2
        + g sum_i (1 - cos phi_i)                      <- on-site, NEW
        + 1/2 sum_{i != j} K_ij [1 - cos(phi_i - phi_j)]

    omega_dot_i = -g sin(phi_i) - sum_j K_ij sin(phi_i - phi_j) - gamma_i omega_i

Setting ``g = 0`` recovers the Kuramoto system exactly, so the two models can be
compared on one axis rather than as separate benchmarks.  That is the point of
the construction: ``g`` interpolates between a system with a zero mode and one
without, and the structure-preservation claim can be tested as the symmetry is
broken rather than only at a single symmetric point.

Why this matters for the quantum ansatz: the QGNN read-out is

    H_theta = sum_i a_i <Z_i> + sum_(i,j) w_ij <Z_i Z_j>

The single-``Z`` terms are exactly where an on-site potential maps, and under
the ``Ry`` angle encoding ``<Z_i> = cos(angle)`` identically -- so the pendulum
potential ``(1 - cos phi)`` is the native form of this ansatz, not an
approximation to it.  No circuit change is required; only the training data and
the learned coefficients differ.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "pendulum_chain_field",
    "pendulum_chain_hamiltonian",
    "total_momentum",
    "integrate_rk4",
    "generate_dataset",
]


def pendulum_chain_field(
    phi: np.ndarray,
    omega: np.ndarray,
    K: np.ndarray,
    gamma: np.ndarray,
    g: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vector field of the gravity-bearing coupled-pendulum chain.

    phi, omega : (B, N) batched states.
    K          : (N, N) symmetric coupling, zero diagonal.
    gamma      : (N,)   per-node damping (0 for the conservative case).
    g          : on-site restoring strength.  g = 0 gives Kuramoto exactly.

    Returns (dphi, domega), each (B, N).
    """
    B, N = phi.shape
    dphi = omega.copy()
    domega = -g * np.sin(phi)                       # on-site term
    for i in range(N):
        coupling = np.zeros(B)
        for j in range(N):
            if i != j and K[i, j] != 0.0:
                coupling += K[i, j] * np.sin(phi[:, i] - phi[:, j])
        domega[:, i] -= coupling + gamma[i] * omega[:, i]
    return dphi, domega


def pendulum_chain_hamiltonian(
    phi: np.ndarray, omega: np.ndarray, K: np.ndarray, g: float = 1.0
) -> np.ndarray:
    """H = 1/2 sum omega^2 + g sum (1 - cos phi) + 1/2 sum_{i!=j} K_ij [1 - cos dphi]."""
    kinetic = 0.5 * (omega ** 2).sum(axis=1)
    onsite = g * (1.0 - np.cos(phi)).sum(axis=1)
    dphi = phi[:, :, None] - phi[:, None, :]
    coupling = 0.5 * (K[None] * (1.0 - np.cos(dphi))).sum(axis=(1, 2))
    return kinetic + onsite + coupling


def total_momentum(omega: np.ndarray) -> np.ndarray:
    """sum_i omega_i -- conserved when g = 0, not conserved when g > 0.

    This is the diagnostic that separates the two models: it is the Noether
    charge of the global phase shift, so watching it decay as ``g`` grows is a
    direct measurement of the symmetry being broken.
    """
    return omega.sum(axis=1)


def integrate_rk4(
    phi0: np.ndarray,
    omega0: np.ndarray,
    K: np.ndarray,
    gamma: np.ndarray,
    g: float,
    dt: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """RK4 rollout.  Returns (phi, omega) each (n_steps + 1, B, N)."""
    phi, omega = phi0.copy(), omega0.copy()
    P, W = [phi.copy()], [omega.copy()]
    for _ in range(n_steps):
        k1p, k1w = pendulum_chain_field(phi, omega, K, gamma, g)
        k2p, k2w = pendulum_chain_field(phi + 0.5 * dt * k1p, omega + 0.5 * dt * k1w, K, gamma, g)
        k3p, k3w = pendulum_chain_field(phi + 0.5 * dt * k2p, omega + 0.5 * dt * k2w, K, gamma, g)
        k4p, k4w = pendulum_chain_field(phi + dt * k3p, omega + dt * k3w, K, gamma, g)
        phi = phi + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        omega = omega + (dt / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)
        P.append(phi.copy())
        W.append(omega.copy())
    return np.asarray(P), np.asarray(W)


def generate_dataset(
    N: int,
    edges: list[tuple[int, int]],
    n_samples: int = 600,
    g: float = 1.0,
    k_coupling: float = 0.5,
    phi_range: float = np.pi / 2,
    omega_range: float = 1.0,
    seed: int = 0,
) -> dict:
    """
    Phase-space samples with their exact vector field and energy.

    Sampling is over the phase-space box rather than along trajectories, so the
    training set is not biased toward whatever region a particular initial
    condition happens to visit.
    """
    rng = np.random.default_rng(seed)
    K = np.zeros((N, N))
    for (i, j) in edges:
        K[i, j] = K[j, i] = k_coupling
    gamma = np.zeros(N)

    phi = rng.uniform(-phi_range, phi_range, size=(n_samples, N))
    omega = rng.uniform(-omega_range, omega_range, size=(n_samples, N))
    dphi, domega = pendulum_chain_field(phi, omega, K, gamma, g)
    H = pendulum_chain_hamiltonian(phi, omega, K, g)

    return {
        "N": N, "edges": edges, "K": K, "gamma": gamma, "g": g,
        "state": np.concatenate([phi, omega], axis=1),   # (B, 2N)
        "dstate": np.concatenate([dphi, domega], axis=1),
        "H": H,
        "momentum": total_momentum(omega),
    }
