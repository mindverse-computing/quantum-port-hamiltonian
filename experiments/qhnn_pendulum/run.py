"""
experiments/run_qhnn_pendulum.py  (v3 — symplectic integrator + larger dataset)
===============================================================
End-to-end experiment: Q-HNN on the Nonlinear Pendulum (conservative).

Pipeline
--------
1. Generate phase-space dataset (nonlinear pendulum vector field)
2. 80/20 train/validation split
3. Train QuantumHNN with BFGS on vector-field MSE
4. Compute per-component train/val MSE, energy conservation, trajectory RMSE
5. Roll out trajectory with learned vector field
6. Save 4 figures to codes/report/qhnn_pendulum/
7. Write structured report.md

Usage
-----
    cd hnn-quantum/codes
    python experiments/run_qhnn_pendulum.py

Qiskit API
----------
Uses StatevectorEstimator with sorted parameter binding (Qiskit 2.x).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib
matplotlib.use("Agg")

from non_dissipative import (
    QuantumHNN,
    NonlinearPendulum,
    train_qhnn,
    VectorFieldDataset,
)
from common.metrics import (
    QHNNMetrics,
    compute_trajectory_rmse,
    compute_energy_conservation,
)
from common.visualization import (
    plot_trajectory_comparison,
    plot_phase_portrait,
    plot_vector_field_comparison,
    plot_training_loss,
)
from common.report_writer import write_qhnn_report

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "n_samples":      200,   # increased from 50 → 200 for better generalisation
    "val_fraction":  0.20,
    "n_layers":       1,
    "max_iter":      200,    # increased from 80 → 200
    "seed":          42,
    "dt":           0.05,
    "n_rollout_steps": 300,  # increased from 150 → 300 (symplectic integrator handles long rollouts)
    "q0":            0.8,
    "p0":            0.0,
    "q_range":       np.pi / 2,
    "p_range":       1.0,
}

REPORT_DIR = Path(__file__).resolve().parent / "figures"


def evaluate_split(
    model: QuantumHNN,
    theta: np.ndarray,
    dataset: VectorFieldDataset,
) -> tuple[float, float]:
    """Compute per-component MSE on a dataset split."""
    q_dot_errs, p_dot_errs = [], []
    for i in range(dataset.n_samples):
        qd = model.q_dot(dataset.q[i], dataset.p[i], theta)
        pd = model.p_dot(dataset.q[i], dataset.p[i], theta)
        q_dot_errs.append((qd - dataset.q_dot[i])**2)
        p_dot_errs.append((pd - dataset.p_dot[i])**2)
    return float(np.mean(q_dot_errs)), float(np.mean(p_dot_errs))


def main():
    t_start = time.time()

    print("=" * 60)
    print("  Q-HNN  |  Nonlinear Pendulum  |  Conservative System")
    print("=" * 60)
    print(f"  Config: {CONFIG}")

    # ─── 1. Data ──────────────────────────────────────────────────────────────
    pendulum = NonlinearPendulum(
        q_range=CONFIG["q_range"], p_range=CONFIG["p_range"]
    )
    all_data = pendulum.generate(n_samples=CONFIG["n_samples"], seed=CONFIG["seed"])
    train_data, val_data = all_data.train_test_split(
        test_fraction=CONFIG["val_fraction"], seed=CONFIG["seed"]
    )
    print(f"\n[Data] Train: {train_data.n_samples} pts | Val: {val_data.n_samples} pts")
    print(f"  q range: [{train_data.q.min():.2f}, {train_data.q.max():.2f}]")
    print(f"  p range: [{train_data.p.min():.2f}, {train_data.p.max():.2f}]")

    # ─── 2. Model ─────────────────────────────────────────────────────────────
    model = QuantumHNN(n_layers=CONFIG["n_layers"], seed=CONFIG["seed"])
    print(f"\n[Model] {model}")
    print(f"\n[Circuit]\n{model.circuit_diagram()}\n")

    # ─── 3. Training ──────────────────────────────────────────────────────────
    print("[Training] Starting BFGS optimisation...")
    result = train_qhnn(
        model, train_data, val_data,
        max_iter=CONFIG["max_iter"],
        verbose=True,
    )
    theta_opt = result.theta_opt

    # ─── 4. Metrics ───────────────────────────────────────────────────────────
    print("\n[Metrics] Computing full evaluation...")

    # Component-wise MSE on train and val
    train_q_mse, train_p_mse = evaluate_split(model, theta_opt, train_data)
    val_q_mse, val_p_mse     = evaluate_split(model, theta_opt, val_data)

    # Trajectory rollout
    t_pred, q_pred, p_pred = model.symplectic_rollout(
        CONFIG["q0"], CONFIG["p0"], theta_opt,
        dt=CONFIG["dt"], n_steps=CONFIG["n_rollout_steps"],
    )
    t_true, q_true, p_true = pendulum.integrate(
        CONFIG["q0"], CONFIG["p0"],
        dt=CONFIG["dt"], n_steps=CONFIG["n_rollout_steps"],
    )

    # Energy along rollout (should be constant for conservative system)
    H_pred = model.energy_along_trajectory(q_pred, p_pred, theta_opt)
    H_true = pendulum.hamiltonian(q_true, p_true)

    energy_err, energy_rel = compute_energy_conservation(H_pred)
    traj_rmse_q, traj_rmse_p = compute_trajectory_rmse(q_pred, p_pred, q_true, p_true)

    # Pack metrics
    metrics = QHNNMetrics(
        train_loss_history=result.loss_history,
        val_loss_history=[],   # BFGS doesn't eval val every iter — computed at end
        train_q_dot_mse=train_q_mse,
        train_p_dot_mse=train_p_mse,
        val_q_dot_mse=val_q_mse,
        val_p_dot_mse=val_p_mse,
        energy_conservation_error=energy_err,
        energy_conservation_rel=energy_rel,
        trajectory_rmse_q=traj_rmse_q,
        trajectory_rmse_p=traj_rmse_p,
        wall_time_s=result.wall_time_s,
        n_iterations=result.n_iter,
        converged=result.optimizer_result.success,
        n_train=train_data.n_samples,
        n_val=val_data.n_samples,
    )
    metrics.print_summary("Q-HNN | Nonlinear Pendulum")

    # ─── 5. Figures ───────────────────────────────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[Figures] Saving to {REPORT_DIR}/")

    figs = []

    # Training loss
    plot_training_loss(
        result.loss_history,
        title="Q-HNN Training Loss (Nonlinear Pendulum)",
        save_path=REPORT_DIR / "training_loss.png",
    )
    figs.append("training_loss.png")

    # Trajectory comparison
    plot_trajectory_comparison(
        t_true, q_true, q_pred, p_true, p_pred,
        title="Q-HNN Trajectory Rollout (Nonlinear Pendulum)",
        save_path=REPORT_DIR / "trajectory.png",
    )
    figs.append("trajectory.png")

    # Phase portrait
    plot_phase_portrait(
        q_true, p_true, q_pred, p_pred,
        title="Phase Portrait: Nonlinear Pendulum",
        save_path=REPORT_DIR / "phase_portrait.png",
    )
    figs.append("phase_portrait.png")

    # Vector field comparison on validation set
    q_dot_pred_val = np.array([
        model.q_dot(val_data.q[i], val_data.p[i], theta_opt)
        for i in range(val_data.n_samples)
    ])
    p_dot_pred_val = np.array([
        model.p_dot(val_data.q[i], val_data.p[i], theta_opt)
        for i in range(val_data.n_samples)
    ])
    plot_vector_field_comparison(
        val_data.q, val_data.p,
        val_data.q_dot, val_data.p_dot,
        q_dot_pred_val, p_dot_pred_val,
        title="Vector Field: Q-HNN vs True (Validation Set)",
        save_path=REPORT_DIR / "vector_field_val.png",
    )
    figs.append("vector_field_val.png")

    # ─── 6. Report ────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    write_qhnn_report(
        report_dir=REPORT_DIR,
        metrics=metrics,
        config={
            "n_train": train_data.n_samples,
            "n_val": val_data.n_samples,
            "n_layers": CONFIG["n_layers"],
            "max_iter": CONFIG["max_iter"],
            "optimizer": "BFGS (exact statevector)",
            "q_range": f"(-{CONFIG['q_range']:.3f}, {CONFIG['q_range']:.3f})",
            "p_range": f"(-{CONFIG['p_range']:.2f}, {CONFIG['p_range']:.2f})",
            "rollout_steps": CONFIG["n_rollout_steps"],
            "dt": CONFIG["dt"],
        },
        system_name="Nonlinear Pendulum",
        theta_opt=theta_opt,
        figures=figs,
        elapsed=elapsed,
    )

    print(f"\n{'=' * 60}")
    print("  EXPERIMENT COMPLETE")
    print(f"  Report → {REPORT_DIR / 'report.md'}")
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
