"""
common/visualization.py
=======================
Plotting utilities shared across Q-HNN and Q-pHNN experiments.

All functions save figures to disk (no plt.show()) so they can run
in non-interactive environments (training scripts, CI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Colour palette (consistent across modules)
# ---------------------------------------------------------------------------
COLOURS = {
    "true": "#1a1a2e",       # dark navy
    "pred": "#e94560",       # vivid red
    "energy": "#0f3460",     # deep blue
    "loss": "#533483",       # purple
    "dissip": "#e94560",     # red (dissipation)
    "conserv": "#16213e",    # navy (conservative)
}


def plot_trajectory_comparison(
    t: np.ndarray,
    q_true: np.ndarray,
    q_pred: np.ndarray,
    p_true: np.ndarray | None = None,
    p_pred: np.ndarray | None = None,
    title: str = "Trajectory Comparison",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Compare true vs predicted trajectories for q (and optionally p).

    Parameters
    ----------
    t : np.ndarray
        Time array.
    q_true, q_pred : np.ndarray
        True and predicted position trajectories.
    p_true, p_pred : np.ndarray, optional
        True and predicted momentum trajectories.
    title : str
        Figure title.
    save_path : str or Path, optional
        If provided, saves the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_axes = 2 if (p_true is not None and p_pred is not None) else 1
    fig, axes = plt.subplots(1, n_axes, figsize=(6 * n_axes, 4))
    if n_axes == 1:
        axes = [axes]

    axes[0].plot(t, q_true, "--", color=COLOURS["true"], lw=2, label="True q(t)")
    axes[0].plot(t, q_pred, "-",  color=COLOURS["pred"], lw=2, alpha=0.85, label="Predicted q(t)")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Position q")
    axes[0].set_title("Position Trajectory")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if n_axes == 2:
        axes[1].plot(t, p_true, "--", color=COLOURS["true"], lw=2, label="True p(t)")
        axes[1].plot(t, p_pred, "-",  color=COLOURS["pred"], lw=2, alpha=0.85, label="Predicted p(t)")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Momentum p")
        axes[1].set_title("Momentum Trajectory")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [viz] Saved trajectory plot → {save_path}")

    return fig


def plot_phase_portrait(
    q_true: np.ndarray,
    p_true: np.ndarray,
    q_pred: np.ndarray,
    p_pred: np.ndarray,
    title: str = "Phase Portrait",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Phase portrait: true orbit vs predicted orbit in (q, p) space.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(q_true, p_true, "--", color=COLOURS["true"], lw=2, label="True orbit")
    ax.plot(q_pred, p_pred, "-",  color=COLOURS["pred"], lw=2, alpha=0.85, label="Predicted orbit")
    ax.set_xlabel("Position q")
    ax.set_ylabel("Momentum p")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [viz] Saved phase portrait → {save_path}")

    return fig


def plot_energy_curve(
    t: np.ndarray,
    H_vals: np.ndarray,
    label: str = "H(t)",
    title: str = "Hamiltonian Energy over Time",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot energy H(t) along the predicted trajectory, verifying
    conservation (Q-HNN) or monotone decay (Q-pHNN).
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(t, H_vals, "-", color=COLOURS["energy"], lw=2, label=label)
    ax.axhline(H_vals[0], color="gray", lw=1, ls="--", alpha=0.6, label="H(0)")
    ax.set_xlabel("Time")
    ax.set_ylabel("H(q, p)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [viz] Saved energy curve → {save_path}")

    return fig


def plot_training_loss(
    losses: list[float] | np.ndarray,
    title: str = "Training Loss",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot scalar training loss vs optimisation iterations.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(losses, color=COLOURS["loss"], lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [viz] Saved loss curve → {save_path}")

    return fig


def plot_vector_field_comparison(
    q_data: np.ndarray,
    p_data: np.ndarray,
    q_dot_true: np.ndarray,
    p_dot_true: np.ndarray,
    q_dot_pred: np.ndarray,
    p_dot_pred: np.ndarray,
    title: str = "Vector Field: True vs Predicted",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Scatter comparison of true vs predicted vector field components (q̇, ṗ).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, true_vals, pred_vals, ylabel in zip(
        axes,
        [q_dot_true, p_dot_true],
        [q_dot_pred, p_dot_pred],
        ["q̇ (True)", "ṗ (True)"],
    ):
        mn = min(true_vals.min(), pred_vals.min()) * 1.1
        mx = max(true_vals.max(), pred_vals.max()) * 1.1
        ax.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.5, label="y = x")
        ax.scatter(true_vals, pred_vals, s=20, color=COLOURS["pred"], alpha=0.7)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(ylabel.replace("True", "Predicted"))
        ax.set_title(ylabel.replace(" (True)", ""))
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [viz] Saved vector field comparison → {save_path}")

    return fig


def plot_qphnn_summary(
    t: np.ndarray,
    q_true: np.ndarray,
    q_pred: np.ndarray,
    losses: list[float] | np.ndarray,
    title: str = "Q-pHNN Training Summary",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Combined 2-panel summary: (left) trajectory comparison, (right) loss curve.
    """
    fig = plt.figure(figsize=(11, 4))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, q_true, "--", color=COLOURS["true"], lw=2, label="True q(t)")
    ax1.plot(t, q_pred, "-",  color=COLOURS["pred"], lw=2, alpha=0.85, label="Predicted q(t)")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Position q")
    ax1.set_title("Trajectory")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.semilogy(losses, color=COLOURS["loss"], lw=2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss (log scale)")
    ax2.set_title("Training Loss")
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle(title, fontsize=13, fontweight="bold")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [viz] Saved summary plot → {save_path}")

    return fig
