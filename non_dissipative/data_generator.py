"""
non_dissipative/data_generator.py
==================================
Data generators for closed (conservative) dynamical systems.

All generators return random phase-space points and their exact
time derivatives (the vector field), which are used to train the Q-HNN.

This is the "correct" approach identified in Q-Hamiltonian.ipynb §2:
  - Sample random (q, p) from phase space
  - Compute true (q̇, ṗ) analytically
  - Train on the vector field, NOT on an integrated trajectory

Systems
-------
1. NonlinearPendulum  — H = ½p² + (1 - cos q)
2. HarmonicOscillator — H = ½kq² + ½p²/m
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class VectorFieldDataset:
    """Dataset of phase-space points and their vector-field values."""

    q: np.ndarray        # (N,) position samples
    p: np.ndarray        # (N,) momentum samples
    q_dot: np.ndarray    # (N,) true dq/dt
    p_dot: np.ndarray    # (N,) true dp/dt
    name: str = "dataset"

    @property
    def n_samples(self) -> int:
        return len(self.q)

    def train_test_split(
        self, test_fraction: float = 0.2, seed: int = 42
    ) -> tuple["VectorFieldDataset", "VectorFieldDataset"]:
        """Split into train/test subsets."""
        rng = np.random.default_rng(seed)
        n = self.n_samples
        idx = rng.permutation(n)
        n_test = max(1, int(n * test_fraction))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        train = VectorFieldDataset(
            self.q[train_idx], self.p[train_idx],
            self.q_dot[train_idx], self.p_dot[train_idx],
            name=self.name + "_train",
        )
        test = VectorFieldDataset(
            self.q[test_idx], self.p[test_idx],
            self.q_dot[test_idx], self.p_dot[test_idx],
            name=self.name + "_test",
        )
        return train, test

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (q, p, q_dot, p_dot)."""
        return self.q, self.p, self.q_dot, self.p_dot


# ---------------------------------------------------------------------------
# 1. Nonlinear Pendulum
# ---------------------------------------------------------------------------

class NonlinearPendulum:
    """
    Undamped nonlinear pendulum (unit mass, unit length).

    Hamiltonian:
        H(q, p) = ½p² + (1 - cos q)

    Hamilton's equations:
        q̇ = p
        ṗ = −sin(q)

    Notes
    -----
    At small q: reduces to harmonic oscillator (sin q ≈ q).
    At q = π: unstable equilibrium (separatrix).
    Samples are drawn from q ∈ (-π/2, π/2) to stay in the libration regime.
    """

    def __init__(self, q_range: float = np.pi / 2, p_range: float = 1.0):
        self.q_range = q_range
        self.p_range = p_range

    def hamiltonian(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """H(q, p) = ½p² + (1 - cos q)."""
        return 0.5 * p**2 + (1.0 - np.cos(q))

    def vector_field(
        self, q: np.ndarray, p: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (q̇, ṗ) = (p, −sin q)."""
        return p, -np.sin(q)

    def generate(
        self, n_samples: int = 50, seed: int = 0
    ) -> VectorFieldDataset:
        """
        Sample random phase-space points and compute exact vector field.

        Parameters
        ----------
        n_samples : int
            Number of (q, p) points.
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        VectorFieldDataset
        """
        rng = np.random.default_rng(seed)
        q = rng.uniform(-self.q_range, self.q_range, n_samples)
        p = rng.uniform(-self.p_range, self.p_range, n_samples)
        q_dot, p_dot = self.vector_field(q, p)
        return VectorFieldDataset(q, p, q_dot, p_dot, name="nonlinear_pendulum")

    def integrate(
        self,
        q0: float,
        p0: float,
        dt: float = 0.05,
        n_steps: int = 200,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Integrate trajectory using 4th-order Runge-Kutta.

        Returns
        -------
        t, q, p : np.ndarray
        """
        t_vals = np.arange(n_steps) * dt
        q, p = np.zeros(n_steps), np.zeros(n_steps)
        q[0], p[0] = q0, p0

        for i in range(n_steps - 1):
            def f(state):
                qq, pp = state
                dq, dp = self.vector_field(np.array([qq]), np.array([pp]))
                return np.array([dq[0], dp[0]])

            s = np.array([q[i], p[i]])
            k1 = f(s)
            k2 = f(s + 0.5 * dt * k1)
            k3 = f(s + 0.5 * dt * k2)
            k4 = f(s + dt * k3)
            s_new = s + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            q[i+1], p[i+1] = s_new

        return t_vals, q, p


# ---------------------------------------------------------------------------
# 2. Harmonic Oscillator (conservative baseline)
# ---------------------------------------------------------------------------

class HarmonicOscillator:
    """
    Undamped 1D harmonic oscillator (linear, conservative).

    Hamiltonian:
        H(q, p) = ½kq² + ½p²/m

    Hamilton's equations:
        q̇ = p/m
        ṗ = −kq

    Used as a simpler validation target before the nonlinear pendulum.
    The Q-HNN should recover ω = √(k/m) exactly.
    """

    def __init__(
        self,
        k: float = 1.0,
        m: float = 1.0,
        q_range: float = 1.5,
        p_range: float = 1.5,
    ):
        self.k = k
        self.m = m
        self.q_range = q_range
        self.p_range = p_range
        self.omega = np.sqrt(k / m)

    def hamiltonian(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """H(q, p) = ½kq² + ½p²/m."""
        return 0.5 * self.k * q**2 + 0.5 * p**2 / self.m

    def vector_field(
        self, q: np.ndarray, p: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (q̇, ṗ) = (p/m, −kq)."""
        return p / self.m, -self.k * q

    def generate(
        self, n_samples: int = 50, seed: int = 0
    ) -> VectorFieldDataset:
        """Sample random phase-space points and compute exact vector field."""
        rng = np.random.default_rng(seed)
        q = rng.uniform(-self.q_range, self.q_range, n_samples)
        p = rng.uniform(-self.p_range, self.p_range, n_samples)
        q_dot, p_dot = self.vector_field(q, p)
        return VectorFieldDataset(q, p, q_dot, p_dot, name="harmonic_oscillator")

    def integrate(
        self,
        q0: float,
        p0: float,
        dt: float = 0.05,
        n_steps: int = 200,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact analytical trajectory."""
        t = np.arange(n_steps) * dt
        q = q0 * np.cos(self.omega * t) + (p0 / (self.m * self.omega)) * np.sin(self.omega * t)
        p = -q0 * self.m * self.omega * np.sin(self.omega * t) + p0 * np.cos(self.omega * t)
        return t, q, p
