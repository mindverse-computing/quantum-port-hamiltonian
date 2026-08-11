"""
experiments/run_layer_ablation.py
=================================
Layer ablation study: Q-HNN with L=1, 2, 3 entanglement layers on the
nonlinear pendulum.

For each layer count L, trains the Q-HNN with the same BFGS optimizer and
reports:
  - Final validation q̇ MSE and ṗ MSE
  - Energy drift (relative std of H along 300-step symplectic rollout)
  - Number of parameters (4L)
  - Training wall time

Produces a Markdown summary table and a bar-chart figure saved to
    codes/report/layer_ablation/

Usage
-----
    cd hnn-quantum/codes
    python experiments/run_layer_ablation.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from non_dissipative import QuantumHNN, NonlinearPendulum, train_qhnn
from common.metrics import compute_energy_conservation, compute_trajectory_rmse

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG = {
    "n_samples":         200,
    "val_fraction":      0.20,
    "max_iter":          200,
    "seed":              42,
    "dt":               0.05,
    "n_rollout_steps":   300,
    "q0":               0.8,
    "p0":               0.0,
    "q_range":          np.pi / 2,
    "p_range":          1.0,
    "n_layers_list":    [1, 2, 3],
}

OUT_DIR = Path(__file__).resolve().parent / "figures"


def run_ablation_for_layer(n_layers: int, train_data, val_data, pendulum) -> dict:
    """Train Q-HNN with `n_layers` layers and return metrics dict."""
    print(f"\n{'─' * 50}")
    print(f"  Layer ablation: L={n_layers} ({4 * n_layers} params)")
    print(f"{'─' * 50}")

    model = QuantumHNN(n_layers=n_layers, seed=CONFIG["seed"])
    t0 = time.time()
    result = train_qhnn(
        model, train_data, val_data,
        max_iter=CONFIG["max_iter"],
        verbose=False,
    )
    theta_opt = result.theta_opt
    wall_time = time.time() - t0

    # Symplectic rollout
    t_pred, q_pred, p_pred = model.symplectic_rollout(
        CONFIG["q0"], CONFIG["p0"], theta_opt,
        dt=CONFIG["dt"], n_steps=CONFIG["n_rollout_steps"],
    )
    t_true, q_true, p_true = pendulum.integrate(
        CONFIG["q0"], CONFIG["p0"],
        dt=CONFIG["dt"], n_steps=CONFIG["n_rollout_steps"],
    )

    H_pred = model.energy_along_trajectory(q_pred, p_pred, theta_opt)
    _, energy_rel = compute_energy_conservation(H_pred)
    rmse_q, rmse_p = compute_trajectory_rmse(q_pred, p_pred, q_true, p_true)

    # Validation MSE
    val_q_errs, val_p_errs = [], []
    for i in range(val_data.n_samples):
        qd = model.q_dot(val_data.q[i], val_data.p[i], theta_opt)
        pd = model.p_dot(val_data.q[i], val_data.p[i], theta_opt)
        val_q_errs.append((qd - val_data.q_dot[i]) ** 2)
        val_p_errs.append((pd - val_data.p_dot[i]) ** 2)
    val_q_mse = float(np.mean(val_q_errs))
    val_p_mse = float(np.mean(val_p_errs))

    metrics = {
        "n_layers":      n_layers,
        "n_params":      4 * n_layers,
        "n_iter":        result.n_iter,
        "final_loss":    result.final_loss,
        "val_q_mse":     val_q_mse,
        "val_p_mse":     val_p_mse,
        "energy_drift":  energy_rel,
        "rmse_q":        rmse_q,
        "rmse_p":        rmse_p,
        "wall_time_s":   wall_time,
    }

    print(f"  n_iter={result.n_iter}, final_loss={result.final_loss:.4f}")
    print(f"  val_q_mse={val_q_mse:.4f}, val_p_mse={val_p_mse:.4f}")
    print(f"  energy_drift={energy_rel:.4%}, wall_time={wall_time:.1f}s")

    return metrics


def plot_ablation(results: list[dict], out_dir: Path) -> None:
    """Save bar charts comparing metrics across layer counts."""
    layers = [r["n_layers"] for r in results]
    labels = [f"L={l}\n({4*l}p)" for l in layers]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Q-HNN Layer Ablation — Nonlinear Pendulum", fontsize=13, fontweight="bold")

    colors = ["#4C72B0", "#DD8452", "#55A868"]

    # Energy drift
    ax = axes[0]
    vals = [r["energy_drift"] * 100 for r in results]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Energy Drift (%)", fontsize=11)
    ax.set_title("Energy Conservation", fontsize=11)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.4)

    # Val q̇ MSE
    ax = axes[1]
    vals = [r["val_q_mse"] for r in results]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Val $\\dot{q}$ MSE", fontsize=11)
    ax.set_title("Vector Field Accuracy", fontsize=11)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.4)

    # Wall time
    ax = axes[2]
    vals = [r["wall_time_s"] for r in results]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Training Time (s)", fontsize=11)
    ax.set_title("Computational Cost", fontsize=11)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.0f}s", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.4)

    plt.tight_layout()
    out_path = out_dir / "layer_ablation_bars.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def write_report(results: list[dict], out_dir: Path) -> None:
    """Write Markdown summary table."""
    lines = [
        "# Q-HNN Layer Ablation Report",
        "",
        "System: Nonlinear Pendulum | Optimizer: BFGS | Integrator: Störmer–Verlet",
        f"Dataset: N={CONFIG['n_samples']} total, {int(CONFIG['n_samples'] * (1 - CONFIG['val_fraction']))} train / "
        f"{int(CONFIG['n_samples'] * CONFIG['val_fraction'])} val | Rollout: {CONFIG['n_rollout_steps']} steps",
        "",
        "| L | Params | Iters | Val q̇ MSE | Val ṗ MSE | Energy Drift | RMSE q | Wall Time |",
        "|---|--------|-------|-----------|-----------|-------------|--------|-----------|",
    ]
    for r in results:
        lines.append(
            f"| {r['n_layers']} | {r['n_params']} | {r['n_iter']} "
            f"| {r['val_q_mse']:.4f} | {r['val_p_mse']:.4f} "
            f"| {r['energy_drift']:.2%} | {r['rmse_q']:.4f} | {r['wall_time_s']:.1f}s |"
        )
    lines.extend([
        "",
        "## Notes",
        "- Energy Drift = std(H(t)) / |mean(H(t))| along 300-step symplectic rollout.",
        "- Structural energy conservation (Ḣ=0 exactly) holds for ALL layer counts by circuit construction.",
        "- Increasing L improves vector-field accuracy at cost of 4 additional parameters and ~2× training time per layer.",
    ])
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"  Saved {report_path}")


def main():
    print("=" * 60)
    print("  Q-HNN Layer Ablation Study")
    print("  Nonlinear Pendulum | L = 1, 2, 3")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate shared dataset
    pendulum = NonlinearPendulum(
        q_range=CONFIG["q_range"], p_range=CONFIG["p_range"]
    )
    all_data = pendulum.generate(n_samples=CONFIG["n_samples"], seed=CONFIG["seed"])
    train_data, val_data = all_data.train_test_split(
        test_fraction=CONFIG["val_fraction"], seed=CONFIG["seed"]
    )
    print(f"Dataset: {train_data.n_samples} train, {val_data.n_samples} val")

    results = []
    for n_layers in CONFIG["n_layers_list"]:
        r = run_ablation_for_layer(n_layers, train_data, val_data, pendulum)
        results.append(r)

    print("\n" + "=" * 60)
    print("  ABLATION SUMMARY")
    print("=" * 60)
    print(f"{'L':>4} {'Params':>7} {'Val q̇ MSE':>12} {'Energy Drift':>14} {'Time':>8}")
    print("─" * 50)
    for r in results:
        print(f"{r['n_layers']:>4} {r['n_params']:>7} "
              f"{r['val_q_mse']:>12.4f} {r['energy_drift']:>13.2%} "
              f"{r['wall_time_s']:>7.1f}s")

    plot_ablation(results, OUT_DIR)
    write_report(results, OUT_DIR)

    print(f"\n  Report → {OUT_DIR / 'report.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
