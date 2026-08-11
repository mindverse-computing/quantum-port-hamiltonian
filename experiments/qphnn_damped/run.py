"""
experiments/run_qphnn_damped.py  (v3 — symplectic integrator + passivity loss + larger dataset)
=============================================================
End-to-end experiment: Q-pHNN on the Damped Harmonic Oscillator.

Runs BOTH Q-pHNN variants:
  v1 — DynamicQpHNN: Statevector MINL + COBYLA (trajectory fitting)
  v2 — VectorFieldQpHNN: parameter-shift + BFGS + learned γ

Pipeline (v2)
-------------
1. Generate dissipative vector-field dataset (damped oscillator)
2. 80/20 train/validation split
3. Train VectorFieldQpHNN with BFGS
4. Compute train/val MSE, γ recovery error, trajectory RMSE, energy monotonicity
5. Save figures; write report.md

Pipeline (v1)
-------------
1. Integrate classical trajectory
2. Train DynamicQpHNN with COBYLA (n_shots averaged MINL)
3. Compute trajectory MSE and save report

Qiskit MINL API (verified against qrn_qiskit sample)
------------------------------------------------------
    bitstr, sv = sv.measure([ancilla_idx])   ← Born-rule collapse
    if "1" in str(bitstr):                   ← classical feedforward
        sv = sv.evolve(Rx_operator)

Usage
-----
    cd hnn-quantum/codes
    python experiments/run_qphnn_damped.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib
matplotlib.use("Agg")

from dissipative import (
    DynamicQpHNN,
    VectorFieldQpHNN,
    DampedHarmonicOscillator,
    VanDerPolOscillator,
    DissipativeVectorFieldDataset,
    train_dynamic_qphnn,
    train_vector_field_qphnn,
)
from common.metrics import (
    QpHNNMetrics,
    compute_trajectory_rmse,
    compute_energy_monotone_fraction,
)
from common.visualization import (
    plot_trajectory_comparison,
    plot_phase_portrait,
    plot_training_loss,
    plot_qphnn_summary,
    plot_vector_field_comparison,
)
from common.report_writer import write_qphnn_report

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "k": 1.0,
    "m": 1.0,
    "gamma": 0.3,
    "n_vf_samples":  200,    # increased from 50 → 200
    "val_fraction":  0.20,
    "n_traj_steps":  6,
    "dt_traj":       1.0,
    "v1_max_iter":  80,
    "v1_n_shots":   30,
    "v2_max_iter":  200,     # increased from 80 → 200
    "v2_n_layers":  1,
    "v2_lambda_passivity": 0.1,   # NEW: passivity regularization weight
    "dt_rollout":   0.05,
    "n_rollout_steps": 200,
    "q0":           1.5,
    "p0":           0.0,
    "run_van_der_pol": False,
    "seed":          42,
}

REPORT_DIR_V1 = Path(__file__).resolve().parent / "figures"
REPORT_DIR_V2 = Path(__file__).resolve().parent / "figures"


# ─────────────────────────────────────────────────────────────────────────────
# V1: Dynamic circuit + COBYLA
# ─────────────────────────────────────────────────────────────────────────────

def run_v1(system: DampedHarmonicOscillator) -> None:
    print("\n" + "─" * 60)
    print("  Q-pHNN v1 | Statevector MINL + COBYLA")
    print("  MINL API: bitstr, sv = sv.measure([ancilla_idx])")
    print("─" * 60)

    t_start = time.time()

    # Classical ground-truth trajectory for training target
    traj = system.integrate(
        q0=CONFIG["q0"], p0=CONFIG["p0"],
        dt=CONFIG["dt_traj"],
        n_steps=CONFIG["n_traj_steps"] + 1,
    )

    model_v1 = DynamicQpHNN(
        n_steps=CONFIG["n_traj_steps"],
        seed=CONFIG["seed"],
    )
    print(f"\n[Model v1] {model_v1}")

    result_v1 = train_dynamic_qphnn(
        model_v1, traj,
        init_angle=np.pi / 2,
        max_iter=CONFIG["v1_max_iter"],
        n_shots=CONFIG["v1_n_shots"],
        verbose=True,
    )

    # Predict with higher shot count for evaluation
    pred_q, pred_p = model_v1.predict_trajectory(
        result_v1.params_opt,
        init_angle=np.pi / 2,
        n_shots=200,
    )
    target_q = traj.q[1: CONFIG["n_traj_steps"] + 1]
    target_p = traj.p[1: CONFIG["n_traj_steps"] + 1]

    # Trajectory MSE
    traj_rmse_q, traj_rmse_p = compute_trajectory_rmse(
        pred_q, pred_p, target_q, target_p
    )

    # Energy monotonicity (dummy H using spring energy approximation)
    k, m = CONFIG["k"], CONFIG["m"]
    H_pred = 0.5 * k * pred_q**2 + 0.5 * pred_p**2 / m
    mono_frac = compute_energy_monotone_fraction(H_pred)

    metrics_v1 = QpHNNMetrics(
        train_loss_history=result_v1.loss_history,
        trajectory_rmse_q=traj_rmse_q,
        trajectory_rmse_p=traj_rmse_p,
        energy_monotone_fraction=mono_frac,
        wall_time_s=result_v1.wall_time_s,
        n_iterations=result_v1.n_iter,
        converged=getattr(result_v1.optimizer_result, "success", False),
        n_train=CONFIG["n_traj_steps"],
        n_val=0,
        variant="v1",
        true_gamma=CONFIG["gamma"],
    )
    metrics_v1.print_summary("Q-pHNN v1 | Damped Harmonic Oscillator")

    # Figures
    REPORT_DIR_V1.mkdir(parents=True, exist_ok=True)
    t_traj = traj.t[1: CONFIG["n_traj_steps"] + 1]

    figs = []
    plot_training_loss(
        result_v1.loss_history,
        title="Q-pHNN v1 Training Loss (Statevector MINL + COBYLA)",
        save_path=REPORT_DIR_V1 / "training_loss.png",
    )
    figs.append("training_loss.png")

    plot_qphnn_summary(
        t_traj, target_q, pred_q,
        result_v1.loss_history,
        title="Q-pHNN v1 — Damped Oscillator (Dynamic Circuit)",
        save_path=REPORT_DIR_V1 / "summary.png",
    )
    figs.append("summary.png")

    elapsed = time.time() - t_start
    write_qphnn_report(
        report_dir=REPORT_DIR_V1,
        metrics=metrics_v1,
        config={
            "n_traj_steps": CONFIG["n_traj_steps"],
            "dt_traj": CONFIG["dt_traj"],
            "optimizer": "COBYLA (gradient-free)",
            "max_iter": CONFIG["v1_max_iter"],
            "n_shots_train": CONFIG["v1_n_shots"],
            "true_gamma": CONFIG["gamma"],
        },
        system_name="Damped Harmonic Oscillator",
        params_opt=result_v1.params_opt,
        figures=figs,
        elapsed=elapsed,
        variant="v1",
    )
    print(f"\n  [v1] Report → {REPORT_DIR_V1 / 'report.md'}")


# ─────────────────────────────────────────────────────────────────────────────
# V2: Vector field + BFGS + learned γ
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_split_v2(
    model: VectorFieldQpHNN,
    params: np.ndarray,
    dataset: DissipativeVectorFieldDataset,
) -> tuple[float, float]:
    """Per-component MSE on a dataset split. params = [θ, s, γ]"""
    theta = params[: model.n_circuit_weights]
    s     = float(params[-2])
    gamma = float(params[-1])
    q_errs, p_errs = [], []
    for i in range(dataset.n_samples):
        qd = model.q_dot(dataset.q[i], dataset.p[i], theta, s)
        pd = model.p_dot(dataset.q[i], dataset.p[i], theta, gamma, s)
        q_errs.append((qd - dataset.q_dot[i])**2)
        p_errs.append((pd - dataset.p_dot[i])**2)
    return float(np.mean(q_errs)), float(np.mean(p_errs))


def run_v2(system: DampedHarmonicOscillator) -> None:
    print("\n" + "─" * 60)
    print("  Q-pHNN v2 | Vector-Field + BFGS + Learned γ")
    print("─" * 60)

    t_start = time.time()

    # Generate vector-field dataset with 80/20 split
    all_data = system.generate_vector_field(
        n_samples=CONFIG["n_vf_samples"], seed=CONFIG["seed"]
    )

    # Manual 80/20 split
    n = all_data.n_samples
    n_val = max(1, int(n * CONFIG["val_fraction"]))
    n_train = n - n_val
    rng = np.random.default_rng(CONFIG["seed"])
    idx = rng.permutation(n)
    train_idx, val_idx = idx[n_val:], idx[:n_val]

    train_data = DissipativeVectorFieldDataset(
        q=all_data.q[train_idx], p=all_data.p[train_idx],
        q_dot=all_data.q_dot[train_idx], p_dot=all_data.p_dot[train_idx],
        name="damped_train", true_gamma=CONFIG["gamma"],
    )
    val_data = DissipativeVectorFieldDataset(
        q=all_data.q[val_idx], p=all_data.p[val_idx],
        q_dot=all_data.q_dot[val_idx], p_dot=all_data.p_dot[val_idx],
        name="damped_val", true_gamma=CONFIG["gamma"],
    )
    print(f"\n[Data] Train: {train_data.n_samples} pts | Val: {val_data.n_samples} pts")

    model_v2 = VectorFieldQpHNN(
        n_layers=CONFIG["v2_n_layers"], seed=CONFIG["seed"]
    )
    print(f"\n[Model v2] {model_v2}")
    print(f"\n[Circuit]\n{model_v2.circuit_diagram()}\n")

    result_v2 = train_vector_field_qphnn(
        model_v2, train_data,
        max_iter=CONFIG["v2_max_iter"],
        lambda_passivity=CONFIG["v2_lambda_passivity"],
        verbose=True,
    )
    params_opt = result_v2.params_opt
    s_opt     = float(params_opt[-2])
    gamma_opt = float(params_opt[-1])

    # Component-wise MSE
    train_q_mse, train_p_mse = evaluate_split_v2(model_v2, params_opt, train_data)
    val_q_mse,   val_p_mse   = evaluate_split_v2(model_v2, params_opt, val_data)

    # Rollout
    t_pred, q_pred, p_pred = model_v2.symplectic_rollout(
        CONFIG["q0"], CONFIG["p0"], params_opt,
        dt=CONFIG["dt_rollout"], n_steps=CONFIG["n_rollout_steps"],
    )
    gt_traj = system.integrate(
        q0=CONFIG["q0"], p0=CONFIG["p0"],
        dt=CONFIG["dt_rollout"], n_steps=CONFIG["n_rollout_steps"],
    )

    traj_rmse_q, traj_rmse_p = compute_trajectory_rmse(
        q_pred, p_pred, gt_traj.q, gt_traj.p
    )

    # Energy along rollout
    k, m = CONFIG["k"], CONFIG["m"]
    H_pred = 0.5 * k * q_pred**2 + 0.5 * p_pred**2 / m
    mono_frac = compute_energy_monotone_fraction(H_pred)

    gamma_abs_err = abs(gamma_opt - CONFIG["gamma"])
    gamma_rel_err = gamma_abs_err / CONFIG["gamma"]

    metrics_v2 = QpHNNMetrics(
        train_loss_history=result_v2.loss_history,
        val_loss_history=[],
        train_q_dot_mse=train_q_mse,
        train_p_dot_mse=train_p_mse,
        val_q_dot_mse=val_q_mse,
        val_p_dot_mse=val_p_mse,
        trajectory_rmse_q=traj_rmse_q,
        trajectory_rmse_p=traj_rmse_p,
        learned_gamma=gamma_opt,
        true_gamma=CONFIG["gamma"],
        gamma_abs_error=gamma_abs_err,
        gamma_rel_error=gamma_rel_err,
        energy_monotone_fraction=mono_frac,
        wall_time_s=result_v2.wall_time_s,
        n_iterations=result_v2.n_iter,
        converged=result_v2.optimizer_result.success if result_v2.optimizer_result else False,
        n_train=train_data.n_samples,
        n_val=val_data.n_samples,
        variant="v2",
    )
    metrics_v2.print_summary("Q-pHNN v2 | Damped Harmonic Oscillator")

    # Figures
    REPORT_DIR_V2.mkdir(parents=True, exist_ok=True)
    figs = []

    plot_training_loss(
        result_v2.loss_history,
        title="Q-pHNN v2 Training Loss (Vector Field + BFGS)",
        save_path=REPORT_DIR_V2 / "training_loss.png",
    )
    figs.append("training_loss.png")

    plot_trajectory_comparison(
        gt_traj.t, gt_traj.q, q_pred,
        gt_traj.p, p_pred,
        title=(
            f"Q-pHNN v2 Rollout\n"
            f"γ_true={CONFIG['gamma']:.2f}  γ_learned={gamma_opt:.4f}  "
            f"|Δγ|={gamma_abs_err:.4f}"
        ),
        save_path=REPORT_DIR_V2 / "trajectory.png",
    )
    figs.append("trajectory.png")

    plot_phase_portrait(
        gt_traj.q, gt_traj.p, q_pred, p_pred,
        title="Phase Portrait: Damped Oscillator (Q-pHNN v2)",
        save_path=REPORT_DIR_V2 / "phase_portrait.png",
    )
    figs.append("phase_portrait.png")

    # Val set vector field scatter
    q_dot_pred_val = np.array([
        model_v2.q_dot(val_data.q[i], val_data.p[i], params_opt[: model_v2.n_circuit_weights])
        for i in range(val_data.n_samples)
    ])
    p_dot_pred_val = np.array([
        model_v2.p_dot(val_data.q[i], val_data.p[i],
                        params_opt[: model_v2.n_circuit_weights], gamma_opt)
        for i in range(val_data.n_samples)
    ])
    plot_vector_field_comparison(
        val_data.q, val_data.p,
        val_data.q_dot, val_data.p_dot,
        q_dot_pred_val, p_dot_pred_val,
        title="Vector Field: Q-pHNN v2 vs True (Validation Set)",
        save_path=REPORT_DIR_V2 / "vector_field_val.png",
    )
    figs.append("vector_field_val.png")

    plot_qphnn_summary(
        gt_traj.t, gt_traj.q, q_pred,
        result_v2.loss_history,
        title="Q-pHNN v2 Summary — Damped Oscillator",
        save_path=REPORT_DIR_V2 / "summary.png",
    )
    figs.append("summary.png")

    elapsed = time.time() - t_start
    write_qphnn_report(
        report_dir=REPORT_DIR_V2,
        metrics=metrics_v2,
        config={
            "n_train": train_data.n_samples,
            "n_val": val_data.n_samples,
            "n_layers": CONFIG["v2_n_layers"],
            "max_iter": CONFIG["v2_max_iter"],
            "optimizer": "BFGS (exact statevector)",
            "true_gamma": CONFIG["gamma"],
            "k": CONFIG["k"],
            "m": CONFIG["m"],
            "rollout_steps": CONFIG["n_rollout_steps"],
        },
        system_name="Damped Harmonic Oscillator",
        params_opt=params_opt,
        figures=figs,
        elapsed=elapsed,
        variant="v2",
    )
    print(f"\n  [v2] Report → {REPORT_DIR_V2 / 'report.md'}")


# ─────────────────────────────────────────────────────────────────────────────
# Optional: Van der Pol
# ─────────────────────────────────────────────────────────────────────────────

def run_van_der_pol() -> None:
    print("\n" + "─" * 60)
    print("  Q-pHNN v2 | Van der Pol (nonlinear limit-cycle)")
    print("─" * 60)
    mu = 0.5
    vdp = VanDerPolOscillator(mu=mu)
    vf_data = vdp.generate_vector_field(n_samples=CONFIG["n_vf_samples"], seed=CONFIG["seed"])
    model = VectorFieldQpHNN(n_layers=1, seed=CONFIG["seed"])
    result = train_vector_field_qphnn(model, vf_data, max_iter=60, verbose=True)

    gt_traj = vdp.integrate(q0=2.0, p0=0.0, dt=0.05, n_steps=200)
    _, q_pred, p_pred = model.rollout(2.0, 0.0, result.params_opt, dt=0.05, n_steps=200)

    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plot_phase_portrait(
        gt_traj.q, gt_traj.p, q_pred, p_pred,
        title=f"Van der Pol (μ={mu}) — Q-pHNN v2",
        save_path=out / "phase_portrait.png",
    )
    plot_training_loss(
        result.loss_history,
        title="Van der Pol — Q-pHNN v2 Training Loss",
        save_path=out / "training_loss.png",
    )
    print(f"\n  Van der Pol → {out}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Q-pHNN  |  Damped Harmonic Oscillator  |  Open System")
    print("=" * 60)
    print(f"  k={CONFIG['k']}, m={CONFIG['m']}, γ_true={CONFIG['gamma']}")

    system = DampedHarmonicOscillator(
        k=CONFIG["k"], m=CONFIG["m"], gamma=CONFIG["gamma"]
    )

    # v1: trajectory fitting with Statevector MINL
    run_v1(system)

    # v2: principled vector-field with learned damping
    run_v2(system)

    # Optional: Van der Pol
    if CONFIG["run_van_der_pol"]:
        run_van_der_pol()

    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"  v1 Report → {REPORT_DIR_V1 / 'report.md'}")
    print(f"  v2 Report → {REPORT_DIR_V2 / 'report.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
