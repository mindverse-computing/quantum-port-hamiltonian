"""
common/report_writer.py
=======================
Structured markdown report generation for Q-HNN and Q-pHNN experiments.

Modelled on qrn_qiskit/main.py write_report() pattern but tailored for
Hamiltonian learning experiments.

Each experiment produces a self-contained report.md in:
    codes/report/<experiment_name>/report.md

The report contains:
  - Run configuration (system, hyperparameters)
  - Environment (Qiskit version, NumPy, Python)
  - Train / Validation metrics table
  - Physics metrics (energy conservation / damping recovery)
  - Figures index (links to PNG files in same directory)
  - Reproducibility commands
"""

from __future__ import annotations

import sys
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


def _env_table() -> list[str]:
    """Return markdown rows for environment table."""
    try:
        import qiskit
        qiskit_ver = qiskit.__version__
    except ImportError:
        qiskit_ver = "N/A"

    try:
        import scipy
        scipy_ver = scipy.__version__
    except ImportError:
        scipy_ver = "N/A"

    rows = [
        "| Package | Version |",
        "|---------|---------|",
        f"| Python  | {platform.python_version()} |",
        f"| Qiskit  | {qiskit_ver} |",
        f"| NumPy   | {np.__version__} |",
        f"| SciPy   | {scipy_ver} |",
        f"| Engine  | Qiskit 2.x StatevectorEstimator + Statevector.measure() MINL |",
    ]
    return rows


def write_qhnn_report(
    report_dir: Path,
    metrics,               # QHNNMetrics
    config: dict,
    system_name: str,
    theta_opt: np.ndarray,
    figures: list[str],
    elapsed: float,
) -> Path:
    """
    Write report.md for a Q-HNN (conservative) experiment.

    Parameters
    ----------
    report_dir : Path
        Directory to write report.md into. Created if needed.
    metrics : QHNNMetrics
        Populated metrics object from training.
    config : dict
        Experiment hyperparameters (n_layers, max_iter, etc.).
    system_name : str
        Human-readable system name (e.g., "Nonlinear Pendulum").
    theta_opt : np.ndarray
        Learned circuit weight vector.
    figures : list[str]
        List of figure filenames in report_dir.
    elapsed : float
        Total wall-clock time in seconds.

    Returns
    -------
    Path
        Path to the written report.md file.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    def h(n: int, text: str): lines.append(f"{'#' * n} {text}\n")
    def p(text: str): lines.append(f"{text}\n")
    def sep(): lines.append("---\n")

    # Header
    h(1, f"Q-HNN Experiment Report — {system_name}")
    p(f"**Generated:** {ts}  |  **Runtime:** {elapsed:.1f}s")
    sep()

    # TOC
    h(2, "Table of Contents")
    for i, section in enumerate([
        "Overview", "Environment", "Hyperparameters",
        "Training Results", "Physics Validation", "Figures", "Reproduce"
    ], 1):
        p(f"{i}. [{section}](#{section.lower().replace(' ', '-')})")
    sep()

    # Overview
    h(2, "Overview")
    p(
        "The **Quantum Hamiltonian Neural Network (Q-HNN)** learns the scalar energy "
        "manifold $H_\\theta(q, p)$ of a conservative dynamical system using a 2-qubit "
        "parameterized quantum circuit. Symplectic gradients (Hamilton's equations) are "
        "extracted via the **Parameter-Shift Rule on data-encoding gates**:\n"
    )
    p(r"$$\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q}$$")
    p("\nThe Q-HNN is strictly energy-conserving by construction (all gates unitary).")
    sep()

    # Environment
    h(2, "Environment")
    for row in _env_table():
        p(row)
    sep()

    # Hyperparameters
    h(2, "Hyperparameters")
    p("| Parameter | Value |")
    p("|-----------|-------|")
    p(f"| System | {system_name} |")
    for k, v in config.items():
        p(f"| {k} | {v} |")
    p(f"| n_weights | {len(theta_opt)} |")
    sep()

    # Training Results
    h(2, "Training Results")
    h(3, "Loss History")
    p("| Metric | Value |")
    p("|--------|-------|")
    p(f"| Final Train Loss | {metrics.train_loss_history[-1]:.6f} |" if metrics.train_loss_history else "| Final Train Loss | N/A |")
    p(f"| Best Train Loss | {metrics.best_train_loss:.6f} |")
    p(f"| Final Val Loss | {metrics.val_loss_history[-1]:.6f} |" if metrics.val_loss_history else "| Final Val Loss | N/A |")
    p(f"| Iterations | {metrics.n_iterations} |")
    p(f"| Converged | {metrics.converged} |")
    p(f"| Wall Time | {metrics.wall_time_s:.1f}s |")

    h(3, "Vector Field MSE")
    p("| Split | q̇ MSE | ṗ MSE |")
    p("|-------|--------|--------|")
    p(f"| Train | {metrics.train_q_dot_mse:.6f} | {metrics.train_p_dot_mse:.6f} |")
    p(f"| Val   | {metrics.val_q_dot_mse:.6f} | {metrics.val_p_dot_mse:.6f} |")

    h(3, "Learned Parameters")
    p(f"```\nθ_opt = {np.round(theta_opt, 4)}\n```")
    sep()

    # Physics Validation
    h(2, "Physics Validation")
    p("| Physics Metric | Value |")
    p("|----------------|-------|")
    p(f"| Energy Conservation Error std(H) | {metrics.energy_conservation_error:.6f} |")
    p(f"| Relative Energy Drift std(H)/|⟨H⟩| | {metrics.energy_conservation_rel:.4%} |")
    p(f"| Trajectory RMSE q(t) (normalised) | {metrics.trajectory_rmse_q:.6f} |")
    p(f"| Trajectory RMSE p(t) (normalised) | {metrics.trajectory_rmse_p:.6f} |")
    p(f"\n> **Note**: Energy conservation error < 1% relative drift indicates the Q-HNN "
      f"has learned a physically consistent energy manifold.")
    sep()

    # Figures
    h(2, "Figures")
    for fname in sorted(figures):
        p(f"![{fname}]({fname})")
        p("")
    sep()

    # Reproducibility
    h(2, "Reproduce")
    p("```bash")
    p("cd hnn-quantum/codes")
    p(f"python experiments/run_qhnn_pendulum.py")
    p("```")
    p(f"\nAll figures and this report are saved to `codes/report/qhnn_{system_name.lower().replace(' ', '_')}/`.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [report] Written → {report_path}")
    return report_path


def write_qphnn_report(
    report_dir: Path,
    metrics,               # QpHNNMetrics
    config: dict,
    system_name: str,
    params_opt: np.ndarray,
    figures: list[str],
    elapsed: float,
    variant: str = "v2",
) -> Path:
    """
    Write report.md for a Q-pHNN (dissipative) experiment.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    def h(n: int, text: str): lines.append(f"{'#' * n} {text}\n")
    def p(text: str): lines.append(f"{text}\n")
    def sep(): lines.append("---\n")

    variant_label = "Dynamic Circuit (COBYLA)" if variant == "v1" else "Vector Field (BFGS + γ)"

    h(1, f"Q-pHNN Experiment Report — {system_name} [{variant_label}]")
    p(f"**Generated:** {ts}  |  **Runtime:** {elapsed:.1f}s")
    sep()

    h(2, "Table of Contents")
    for i, section in enumerate([
        "Overview", "Environment", "Hyperparameters",
        "Training Results", "Damping Recovery", "Physics Validation", "Figures", "Reproduce"
    ], 1):
        p(f"{i}. [{section}](#{section.lower().replace(' ', '-')})")
    sep()

    h(2, "Overview")
    if variant == "v1":
        p(
            "**Q-pHNN v1 (Dynamic Circuit)** uses a 2-qubit Statevector circuit with "
            "Measurement-Induced NonLinearity (MINL) to model dissipation:\n"
            "- Conservative block: `Rz(θ_J)` on system qubit\n"
            "- Dissipative coupling: `CRY(θ_R)` entangles system + ancilla\n"
            "- MINL collapse: `bitstr, sv = sv.measure([ancilla])` — Born-rule projection\n"
            "- Feedforward: `if bitstr == '1': sv = sv.evolve(Rx(θ_kick))`"
        )
    else:
        p(
            "**Q-pHNN v2 (Vector Field)** learns the port-Hamiltonian vector field "
            "by combining a quantum circuit energy ansatz with a classical damping parameter:\n"
        )
        p(r"$$\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q} - \gamma p$$")
        p("\nThe quantum circuit learns $H_\\theta(q,p)$; $\\gamma$ is a trainable scalar.")
    sep()

    h(2, "Environment")
    for row in _env_table():
        p(row)
    sep()

    h(2, "Hyperparameters")
    p("| Parameter | Value |")
    p("|-----------|-------|")
    p(f"| System | {system_name} |")
    p(f"| Variant | {variant_label} |")
    for k, v in config.items():
        p(f"| {k} | {v} |")
    sep()

    h(2, "Training Results")
    h(3, "Loss History")
    p("| Metric | Value |")
    p("|--------|-------|")
    p(f"| Final Train Loss | {metrics.train_loss_history[-1]:.6f} |" if metrics.train_loss_history else "| Final Train Loss | N/A |")
    p(f"| Final Val Loss | {metrics.val_loss_history[-1]:.6f} |" if metrics.val_loss_history else "| Final Val Loss | N/A |")
    p(f"| Iterations | {metrics.n_iterations} |")
    p(f"| Converged | {metrics.converged} |")
    p(f"| Wall Time | {metrics.wall_time_s:.1f}s |")

    if variant == "v2":
        h(3, "Vector Field MSE")
        p("| Split | q̇ MSE | ṗ MSE |")
        p("|-------|--------|--------|")
        p(f"| Train | {metrics.train_q_dot_mse:.6f} | {metrics.train_p_dot_mse:.6f} |")
        p(f"| Val   | {metrics.val_q_dot_mse:.6f} | {metrics.val_p_dot_mse:.6f} |")

    h(3, "Learned Parameters")
    p(f"```\nparams_opt = {np.round(params_opt, 4)}\n```")
    sep()

    if variant == "v2" and metrics.learned_gamma is not None:
        h(2, "Damping Recovery")
        p("| Parameter | Value |")
        p("|-----------|-------|")
        p(f"| True γ | {metrics.true_gamma:.4f} |")
        p(f"| Learned γ | {metrics.learned_gamma:.4f} |")
        p(f"| Absolute Error \\|Δγ\\| | {metrics.gamma_abs_error:.4f} |")
        p(f"| Relative Error \\|Δγ\\|/γ | {metrics.gamma_rel_error:.2%} |")
        p(f"\n> **Key result**: A relative error < 10% indicates successful separation "
          f"of conservative (J) and dissipative (R) dynamics on the quantum circuit.")
        sep()

    h(2, "Physics Validation")
    p("| Physics Metric | Value |")
    p("|----------------|-------|")
    p(f"| Trajectory RMSE q(t) (normalised) | {metrics.trajectory_rmse_q:.6f} |")
    p(f"| Trajectory RMSE p(t) (normalised) | {metrics.trajectory_rmse_p:.6f} |")
    p(f"| Energy Monotone Fraction | {metrics.energy_monotone_fraction:.2%} |")
    p(f"\n> Energy monotone fraction > 60% indicates the dissipative channel is correctly "
      f"removing energy from the system.")
    sep()

    h(2, "Figures")
    for fname in sorted(figures):
        p(f"![{fname}]({fname})")
        p("")
    sep()

    h(2, "Reproduce")
    p("```bash")
    p("cd hnn-quantum/codes")
    p("python experiments/run_qphnn_damped.py")
    p("```")
    p(f"\nAll figures and this report are saved to `codes/report/qphnn_{system_name.lower().replace(' ', '_')}/`.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [report] Written → {report_path}")
    return report_path
