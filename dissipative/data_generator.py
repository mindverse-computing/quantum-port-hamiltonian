"""
dissipative/data_generator.py
==============================
Data generators for open (dissipative) dynamical systems.

Provides both:
  A) Random phase-space vector-field samples for v2 (vector-field Q-pHNN)
  B) Integrated trajectory data for v1 (dynamic-circuit trajectory-fitting)

Systems
-------
1. DampedHarmonicOscillator — H = ½kq² + ½p²/m, dissipation: −γp
2. VanDerPolOscillator       — H + nonlinear damping: μ(1−q²)p
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class DissipativeVectorFieldDataset:
    """Dataset of phase-space points and port-Hamiltonian vector field."""

    q: np.ndarray
    p: np.ndarray
    q_dot: np.ndarray
    p_dot: np.ndarray
    name: str = "dissipative_dataset"
    # True parameters for verification
    true_gamma: float | None = None

    @property
    def n_samples(self) -> int:
        return len(self.q)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.q, self.p, self.q_dot, self.p_dot


@dataclass
class TrajectoryDataset:
    """Integrated trajectory dataset for dynamic-circuit (v1) training."""

    t: np.ndarray
    q: np.ndarray
    p: np.ndarray | None = None
    name: str = "trajectory"
    true_omega: float | None = None
    true_gamma: float | None = None


# ---------------------------------------------------------------------------
# 1. Damped Harmonic Oscillator
# ---------------------------------------------------------------------------

class DampedHarmonicOscillator:
    """
    1D damped harmonic oscillator.

    Port-Hamiltonian decomposition:
        H(q, p) = ½kq² + ½p²/m   (stored energy)
        J = [[0, 1], [-1, 0]]      (conservative rotation)
        R = [[0, 0], [0, γ]]       (dissipation on momentum)

    Vector field:
        q̇ =  ∂H/∂p = p/m
        ṗ = −∂H/∂q − γp = −kq − γp

    Trajectory (analytical):
        q(t) = A·cos(ω_d·t + φ)·exp(−γt/2m)
        where ω_d = √(k/m − γ²/4m²)

    Parameters
    ----------
    k : float
        Spring constant.
    m : float
        Mass.
    gamma : float
        Damping coefficient (R diagonal entry).
    """

    def __init__(
        self,
        k: float = 1.0,
        m: float = 1.0,
        gamma: float = 0.3,
        q_range: float = 2.0,
        p_range: float = 2.0,
    ):
        self.k = k
        self.m = m
        self.gamma = gamma
        self.q_range = q_range
        self.p_range = p_range
        self.omega0 = np.sqrt(k / m)

    def hamiltonian(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """H(q, p) = ½kq² + ½p²/m."""
        return 0.5 * self.k * q**2 + 0.5 * p**2 / self.m

    def vector_field(
        self, q: np.ndarray, p: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Port-Hamiltonian vector field (q̇, ṗ)."""
        q_dot = p / self.m
        p_dot = -self.k * q - self.gamma * p
        return q_dot, p_dot

    def generate_vector_field(
        self, n_samples: int = 50, seed: int = 0
    ) -> DissipativeVectorFieldDataset:
        """
        Sample random (q, p) and compute exact (q̇, ṗ) for v2 training.

        Uses a 70/30 blend: 70% trajectory-concentrated samples along the
        true damped orbit (perturbed with small noise) and 30% uniform
        phase-space samples for generalisation coverage.
        """
        rng = np.random.default_rng(seed)

        # --- 70% along the true orbit ---
        n_orbit = int(0.7 * n_samples)
        n_unif  = n_samples - n_orbit

        # Densely integrate the true trajectory to sample from
        omega_d = np.sqrt(max(0.0, self.k / self.m - (self.gamma / (2 * self.m))**2))
        T_d = 2 * np.pi / omega_d if omega_d > 1e-8 else 20.0  # natural period
        t_dense = np.linspace(0, 3 * T_d, 3000)  # 3 full cycles
        alpha = self.gamma / (2 * self.m)

        # RK4 integration along the true orbit from (q0_ref, p0_ref) = (1.5, 0)
        q_ref, p_ref = 1.5, 0.0
        q_traj = np.zeros(len(t_dense))
        p_traj = np.zeros(len(t_dense))
        q_traj[0], p_traj[0] = q_ref, p_ref
        dt_dense = t_dense[1] - t_dense[0]
        for i in range(len(t_dense) - 1):
            def _f(s):
                dq = s[1] / self.m
                dp = -self.k * s[0] - self.gamma * s[1]
                return np.array([dq, dp])
            s = np.array([q_traj[i], p_traj[i]])
            k1 = _f(s)
            k2 = _f(s + 0.5 * dt_dense * k1)
            k3 = _f(s + 0.5 * dt_dense * k2)
            k4 = _f(s + dt_dense * k3)
            sn = s + (dt_dense / 6) * (k1 + 2*k2 + 2*k3 + k4)
            q_traj[i+1], p_traj[i+1] = sn

        # Sample n_orbit points along the orbit + small noise
        idx = rng.choice(len(t_dense), size=n_orbit, replace=False)
        q_orb = q_traj[idx] + rng.normal(0, 0.05, n_orbit)
        p_orb = p_traj[idx] + rng.normal(0, 0.05, n_orbit)

        # --- 30% uniform coverage ---
        q_uni = rng.uniform(-self.q_range, self.q_range, n_unif)
        p_uni = rng.uniform(-self.p_range, self.p_range, n_unif)

        # Combine and shuffle
        q_all = np.concatenate([q_orb, q_uni])
        p_all = np.concatenate([p_orb, p_uni])
        perm  = rng.permutation(n_samples)
        q_all, p_all = q_all[perm], p_all[perm]

        q_dot, p_dot = self.vector_field(q_all, p_all)
        return DissipativeVectorFieldDataset(
            q_all, p_all, q_dot, p_dot,
            name="damped_oscillator_vf",
            true_gamma=self.gamma,
        )

    def integrate(
        self,
        q0: float,
        p0: float,
        dt: float = 1.0,
        n_steps: int = 10,
    ) -> TrajectoryDataset:
        """
        Integrate trajectory using RK4 — for v1 trajectory-fitting.

        Parameters
        ----------
        q0, p0 : float
            Initial conditions.
        dt : float
            Time step size.
        n_steps : int
            Number of steps (circuit depth in v1).

        Returns
        -------
        TrajectoryDataset
        """
        t_vals = np.arange(n_steps) * dt
        q_arr = np.zeros(n_steps)
        p_arr = np.zeros(n_steps)
        q_arr[0], p_arr[0] = q0, p0

        for i in range(n_steps - 1):
            def f(state):
                qq, pp = state
                dq, dp = self.vector_field(
                    np.array([qq]), np.array([pp])
                )
                return np.array([dq[0], dp[0]])

            s = np.array([q_arr[i], p_arr[i]])
            k1 = f(s)
            k2 = f(s + 0.5 * dt * k1)
            k3 = f(s + 0.5 * dt * k2)
            k4 = f(s + dt * k3)
            s_new = s + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            q_arr[i+1], p_arr[i+1] = s_new

        return TrajectoryDataset(
            t=t_vals, q=q_arr, p=p_arr,
            name="damped_oscillator_traj",
            true_omega=self.omega0,
            true_gamma=self.gamma,
        )

    def analytical_trajectory(
        self,
        q0: float,
        p0: float = 0.0,
        t_max: float = 10.0,
        n_points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute exact analytical trajectory q(t) = cos(ω_d·t)·exp(−γ/(2m)·t).

        Returns (t, q) arrays.
        """
        t = np.linspace(0, t_max, n_points)
        alpha = self.gamma / (2 * self.m)
        omega_d_sq = self.k / self.m - alpha**2

        if omega_d_sq > 0:
            omega_d = np.sqrt(omega_d_sq)
            q = q0 * np.cos(omega_d * t) * np.exp(-alpha * t)
        else:
            # Overdamped
            r1 = -alpha + np.sqrt(alpha**2 - self.k / self.m + 0j)
            r2 = -alpha - np.sqrt(alpha**2 - self.k / self.m + 0j)
            q = np.real(q0 * np.exp(r1 * t))

        return t, q


# ---------------------------------------------------------------------------
# 2. Van der Pol Oscillator (nonlinear dissipation)
# ---------------------------------------------------------------------------

class VanDerPolOscillator:
    """
    Van der Pol oscillator with nonlinear damping.

    Equation of motion:
        q̈ − μ(1 − q²)q̇ + q = 0

    As a first-order system (pseudo-port-Hamiltonian):
        q̇ = p
        ṗ = μ(1 − q²)p − q

    The conservative part corresponds to H = ½p² + ½q² (harmonic).
    The dissipative part is the nonlinear μ(1−q²) damping term.
    For μ > 0 the system exhibits a stable limit cycle.

    Parameters
    ----------
    mu : float
        Nonlinear damping coefficient (μ = 0 → undamped).
    """

    def __init__(
        self,
        mu: float = 0.5,
        q_range: float = 2.5,
        p_range: float = 2.5,
    ):
        self.mu = mu
        self.q_range = q_range
        self.p_range = p_range

    def vector_field(
        self, q: np.ndarray, p: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Van der Pol vector field (q̇, ṗ)."""
        q_dot = p
        p_dot = self.mu * (1 - q**2) * p - q
        return q_dot, p_dot

    def generate_vector_field(
        self, n_samples: int = 50, seed: int = 0
    ) -> DissipativeVectorFieldDataset:
        """Sample random (q, p) and compute exact vector field."""
        rng = np.random.default_rng(seed)
        q = rng.uniform(-self.q_range, self.q_range, n_samples)
        p = rng.uniform(-self.p_range, self.p_range, n_samples)
        q_dot, p_dot = self.vector_field(q, p)
        return DissipativeVectorFieldDataset(
            q, p, q_dot, p_dot,
            name=f"van_der_pol_mu{self.mu}",
        )

    def integrate(
        self,
        q0: float,
        p0: float,
        dt: float = 0.05,
        n_steps: int = 500,
    ) -> TrajectoryDataset:
        """Integrate Van der Pol trajectory via RK4."""
        t_vals = np.arange(n_steps) * dt
        q_arr = np.zeros(n_steps)
        p_arr = np.zeros(n_steps)
        q_arr[0], p_arr[0] = q0, p0

        for i in range(n_steps - 1):
            def f(state):
                qq, pp = state
                dq, dp = self.vector_field(
                    np.array([qq]), np.array([pp])
                )
                return np.array([dq[0], dp[0]])

            s = np.array([q_arr[i], p_arr[i]])
            k1 = f(s)
            k2 = f(s + 0.5 * dt * k1)
            k3 = f(s + 0.5 * dt * k2)
            k4 = f(s + dt * k3)
            s_new = s + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            q_arr[i+1], p_arr[i+1] = s_new

        return TrajectoryDataset(
            t=t_vals, q=q_arr, p=p_arr,
            name=f"van_der_pol_traj_mu{self.mu}",
        )
