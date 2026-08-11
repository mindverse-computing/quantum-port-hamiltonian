"""
common/metrics.py
=================
Experiment metrics dataclasses and evaluation utilities for Q-HNN and Q-pHNN.

Tracks both loss-based metrics and physics-informed metrics:
  - Train/validation MSE (per component: q̇ and ṗ)
  - Energy conservation error (Q-HNN: H should be constant along trajectory)
  - Damping recovery error (Q-pHNN v2: |γ_learned − γ_true|)
  - Relative trajectory error (normalized RMSE vs ground truth rollout)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Per-epoch snapshot
# ---------------------------------------------------------------------------

@dataclass
class EpochRecord:
    """Snapshot of metrics at a single optimiser iteration."""

    iteration: int
    train_loss: float
    val_loss: float | None = None      # None if no val set available at this step
    gamma: float | None = None          # Current γ estimate (v2 only)


# ---------------------------------------------------------------------------
# Conservative Q-HNN metrics
# ---------------------------------------------------------------------------

@dataclass
class QHNNMetrics:
    """
    Complete evaluation record for a Q-HNN training run.

    Attributes
    ----------
    train_loss_history : list[float]
        Training loss at each optimiser iteration.
    val_loss_history : list[float]
        Validation loss (same iterations as train, if computed).
    train_q_dot_mse : float
        MSE on training set q̇ component.
    train_p_dot_mse : float
        MSE on training set ṗ component.
    val_q_dot_mse : float
        MSE on validation set q̇ component.
    val_p_dot_mse : float
        MSE on validation set ṗ component.
    energy_conservation_error : float
        std(H(t)) along rolled-out trajectory (should be ~0 for good training).
    energy_conservation_rel : float
        std(H(t)) / |mean(H(t))| — relative energy drift.
    trajectory_rmse_q : float
        Normalized RMSE between predicted and true q(t) rollout.
    trajectory_rmse_p : float
        Normalized RMSE between predicted and true p(t) rollout.
    wall_time_s : float
        Total training time in seconds.
    n_iterations : int
        Number of optimiser iterations to convergence.
    converged : bool
        Whether the optimiser reported success.
    n_train : int
        Number of training samples.
    n_val : int
        Number of validation samples.
    """

    train_loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)

    train_q_dot_mse: float = float("nan")
    train_p_dot_mse: float = float("nan")
    val_q_dot_mse: float = float("nan")
    val_p_dot_mse: float = float("nan")

    energy_conservation_error: float = float("nan")
    energy_conservation_rel: float = float("nan")

    trajectory_rmse_q: float = float("nan")
    trajectory_rmse_p: float = float("nan")

    wall_time_s: float = 0.0
    n_iterations: int = 0
    converged: bool = False
    n_train: int = 0
    n_val: int = 0

    @property
    def best_train_loss(self) -> float:
        return min(self.train_loss_history) if self.train_loss_history else float("nan")

    @property
    def best_val_loss(self) -> float:
        return min(self.val_loss_history) if self.val_loss_history else float("nan")

    def summary_dict(self) -> dict:
        return {
            "train_loss_final":   self.train_loss_history[-1] if self.train_loss_history else float("nan"),
            "val_loss_final":     self.val_loss_history[-1] if self.val_loss_history else float("nan"),
            "train_q_dot_mse":    self.train_q_dot_mse,
            "train_p_dot_mse":    self.train_p_dot_mse,
            "val_q_dot_mse":      self.val_q_dot_mse,
            "val_p_dot_mse":      self.val_p_dot_mse,
            "energy_cons_error":  self.energy_conservation_error,
            "energy_cons_rel":    self.energy_conservation_rel,
            "traj_rmse_q":        self.trajectory_rmse_q,
            "traj_rmse_p":        self.trajectory_rmse_p,
            "wall_time_s":        self.wall_time_s,
            "n_iterations":       self.n_iterations,
            "converged":          self.converged,
            "n_train":            self.n_train,
            "n_val":              self.n_val,
        }

    def print_summary(self, label: str = "Q-HNN"):
        print(f"\n{'─'*55}")
        print(f"  {label} — Experiment Metrics")
        print(f"{'─'*55}")
        print(f"  Training samples  : {self.n_train}")
        print(f"  Validation samples: {self.n_val}")
        print(f"  Iterations        : {self.n_iterations}")
        print(f"  Converged         : {self.converged}")
        print(f"  Wall time         : {self.wall_time_s:.1f}s")
        print(f"\n  ── Vector Field MSE ──")
        print(f"  Train q̇ MSE : {self.train_q_dot_mse:.6f}")
        print(f"  Train ṗ MSE : {self.train_p_dot_mse:.6f}")
        print(f"  Val   q̇ MSE : {self.val_q_dot_mse:.6f}")
        print(f"  Val   ṗ MSE : {self.val_p_dot_mse:.6f}")
        print(f"\n  ── Physics Metrics ──")
        print(f"  Energy cons. error: {self.energy_conservation_error:.6f}")
        print(f"  Energy cons. rel  : {self.energy_conservation_rel:.4%}")
        print(f"  Trajectory RMSE q : {self.trajectory_rmse_q:.6f}")
        print(f"  Trajectory RMSE p : {self.trajectory_rmse_p:.6f}")
        print(f"{'─'*55}")


# ---------------------------------------------------------------------------
# Dissipative Q-pHNN metrics
# ---------------------------------------------------------------------------

@dataclass
class QpHNNMetrics:
    """
    Complete evaluation record for a Q-pHNN training run.

    Extends QHNNMetrics with dissipative-specific metrics:
        - Damping coefficient recovery
        - Energy decay profile (H should monotonically decrease)
    """

    train_loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)

    train_q_dot_mse: float = float("nan")
    train_p_dot_mse: float = float("nan")
    val_q_dot_mse: float = float("nan")
    val_p_dot_mse: float = float("nan")

    trajectory_rmse_q: float = float("nan")
    trajectory_rmse_p: float = float("nan")

    # Dissipative-specific
    learned_gamma: float | None = None
    true_gamma: float | None = None
    gamma_abs_error: float = float("nan")
    gamma_rel_error: float = float("nan")

    # Energy monotonicity check (H should decay)
    energy_monotone_fraction: float = float("nan")
    """Fraction of time steps where H(t+1) < H(t) — should be high for good training."""

    wall_time_s: float = 0.0
    n_iterations: int = 0
    converged: bool = False
    n_train: int = 0
    n_val: int = 0
    variant: str = "v2"   # "v1" or "v2"

    @property
    def best_train_loss(self) -> float:
        return min(self.train_loss_history) if self.train_loss_history else float("nan")

    def summary_dict(self) -> dict:
        return {
            "variant":            self.variant,
            "train_loss_final":   self.train_loss_history[-1] if self.train_loss_history else float("nan"),
            "val_loss_final":     self.val_loss_history[-1] if self.val_loss_history else float("nan"),
            "train_q_dot_mse":    self.train_q_dot_mse,
            "train_p_dot_mse":    self.train_p_dot_mse,
            "val_q_dot_mse":      self.val_q_dot_mse,
            "val_p_dot_mse":      self.val_p_dot_mse,
            "traj_rmse_q":        self.trajectory_rmse_q,
            "traj_rmse_p":        self.trajectory_rmse_p,
            "learned_gamma":      self.learned_gamma,
            "true_gamma":         self.true_gamma,
            "gamma_abs_error":    self.gamma_abs_error,
            "gamma_rel_error":    self.gamma_rel_error,
            "energy_monotone_frac": self.energy_monotone_fraction,
            "wall_time_s":        self.wall_time_s,
            "n_iterations":       self.n_iterations,
            "converged":          self.converged,
            "n_train":            self.n_train,
            "n_val":              self.n_val,
        }

    def print_summary(self, label: str = "Q-pHNN"):
        print(f"\n{'─'*55}")
        print(f"  {label} ({self.variant}) — Experiment Metrics")
        print(f"{'─'*55}")
        print(f"  Training samples  : {self.n_train}")
        print(f"  Validation samples: {self.n_val}")
        print(f"  Iterations        : {self.n_iterations}")
        print(f"  Converged         : {self.converged}")
        print(f"  Wall time         : {self.wall_time_s:.1f}s")
        print(f"\n  ── Vector Field MSE ──")
        print(f"  Train q̇ MSE : {self.train_q_dot_mse:.6f}")
        print(f"  Train ṗ MSE : {self.train_p_dot_mse:.6f}")
        print(f"  Val   q̇ MSE : {self.val_q_dot_mse:.6f}")
        print(f"  Val   ṗ MSE : {self.val_p_dot_mse:.6f}")
        print(f"  Traj RMSE q : {self.trajectory_rmse_q:.6f}")
        print(f"  Traj RMSE p : {self.trajectory_rmse_p:.6f}")
        if self.learned_gamma is not None:
            print(f"\n  ── Damping Recovery ──")
            print(f"  True  γ = {self.true_gamma:.4f}")
            print(f"  Learned γ = {self.learned_gamma:.4f}")
            print(f"  |Δγ|      = {self.gamma_abs_error:.4f}")
            print(f"  |Δγ|/γ    = {self.gamma_rel_error:.2%}")
        print(f"\n  ── Energy Dissipation ──")
        print(f"  Monotone fraction = {self.energy_monotone_fraction:.2%}")
        print(f"{'─'*55}")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_trajectory_rmse(
    q_pred: np.ndarray,
    p_pred: np.ndarray,
    q_true: np.ndarray,
    p_true: np.ndarray,
) -> tuple[float, float]:
    """
    Compute normalized RMSE between predicted and true trajectories.

    Normalizes by the range of the true trajectory to make the metric
    comparable across different systems.

    Returns (rmse_q, rmse_p).
    """
    n = min(len(q_pred), len(q_true))
    q_range = float(np.max(np.abs(q_true[:n]))) or 1.0
    p_range = float(np.max(np.abs(p_true[:n]))) or 1.0

    rmse_q = float(np.sqrt(np.mean((q_pred[:n] - q_true[:n])**2))) / q_range
    rmse_p = float(np.sqrt(np.mean((p_pred[:n] - p_true[:n])**2))) / p_range
    return rmse_q, rmse_p


def compute_energy_conservation(
    H_vals: np.ndarray,
) -> tuple[float, float]:
    """
    Compute energy conservation error along a trajectory.

    Returns
    -------
    abs_error : float
        std(H(t)) — absolute energy fluctuation.
    rel_error : float
        std(H(t)) / |mean(H(t))| — relative energy drift.
    """
    std_H = float(np.std(H_vals))
    mean_H = float(np.abs(np.mean(H_vals)))
    rel = std_H / mean_H if mean_H > 1e-10 else float("nan")
    return std_H, rel


def compute_energy_monotone_fraction(H_vals: np.ndarray) -> float:
    """
    Fraction of steps where H(t+1) <= H(t) (energy is non-increasing).
    Expected to be high for a well-trained dissipative Q-pHNN.
    """
    if len(H_vals) < 2:
        return float("nan")
    diffs = np.diff(H_vals)
    return float(np.mean(diffs <= 0))
