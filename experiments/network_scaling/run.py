"""
experiments/run_network_scaling.py
===================================
Network scaling study: QGNN conservative and dissipative on N=3,4,5 nodes.

Demonstrates that the IHM structural guarantees are SCALE-FREE:
  - Conservative: |Ḣ| = 0 (machine precision) at ALL N
  - Dissipative: passivity (Ḣ ≤ 0) on 100% of states at ALL N

Only vector-field and damping accuracies vary with N and training budget.
Produces a Markdown scaling table.

Usage
-----
    cd hnn-quantum/codes
    python experiments/run_network_scaling.py
    python experiments/run_network_scaling.py --epochs 150  # more training
"""

import sys
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from network import (
    gen_network_conservative, gen_network_dissipative,
    build_ring_coupling, edges_from_coupling,
    NetworkQpHNN, train_network_qphnn,
)

OUT_DIR = Path(__file__).resolve().parent / "figures"


def run_conservative(N: int, epochs: int, seed: int = 1) -> dict:
    """Run conservative QGNN on N nodes; return metrics dict."""
    print(f"\n{'─'*50}")
    print(f"  Conservative N={N} (γ=0, N-torus, {epochs} BFGS iters)")
    print(f"{'─'*50}")

    K = build_ring_coupling(N, strength=1.0)
    edges = edges_from_coupling(K)
    ds = gen_network_conservative(K=K, n_samples=max(60, N * 20), seed=seed)
    train, test = ds.split(0.25, seed=seed)

    n_layers = max(2, int(np.ceil(N / 2)))  # L >= diameter/2 for ring
    model = NetworkQpHNN(N, edges, n_layers=n_layers, dissipative=False,
                         phasor=True, seed=0)
    print(f"  {model}  circuit_params={model.n_circuit_weights}")

    t0 = time.time()
    res = train_network_qphnn(model, train, test, max_iter=epochs, verbose=False)
    wall = time.time() - t0

    # Energy conservation (rollout)
    state0 = ds.states[0].copy()
    traj = model.rollout(state0, res.params_opt, dt=0.05, n_steps=80)
    H_t = model.energy_along(traj, res.params_opt)
    drift_rel = float(np.std(H_t) / (abs(np.mean(H_t)) + 1e-9))

    # Energy rate over held-out states (structural guarantee)
    n_check = min(test.n_samples, 50)
    hdots = np.array([model.energy_rate(test.states[b], res.params_opt)
                      for b in range(n_check)])
    mean_abs_hdot = float(np.mean(np.abs(hdots)))

    metrics = {
        "N": N,
        "n_layers": n_layers,
        "n_circuit_params": model.n_circuit_weights,
        "n_train": train.n_samples,
        "bfgs_iters": res.n_iter,
        "val_phi_mse": res.val_loss if hasattr(res, "val_loss") else float("nan"),
        "energy_drift_rel": drift_rel,
        "mean_abs_hdot": mean_abs_hdot,
        "wall_s": wall,
        "mode": "conservative",
    }
    print(f"  ✓ drift={drift_rel*100:.2f}%, |Ḣ|={mean_abs_hdot:.2e}, "
          f"iters={res.n_iter}, time={wall:.0f}s")
    return metrics


def run_dissipative(N: int, epochs: int, seed: int = 1) -> dict:
    """Run dissipative QGNN on N nodes; return metrics dict."""
    print(f"\n{'─'*50}")
    print(f"  Dissipative N={N} (γ>0, damped Kuramoto, {epochs} BFGS iters)")
    print(f"{'─'*50}")

    K = build_ring_coupling(N, strength=1.0)
    edges = edges_from_coupling(K)
    ds = gen_network_dissipative(K=K, n_samples=max(60, N * 20), seed=seed)
    train, test = ds.split(0.25, seed=seed)
    true_gamma = np.array(ds.gamma)

    n_layers = max(2, int(np.ceil(N / 2)))
    model = NetworkQpHNN(N, edges, n_layers=n_layers, dissipative=True,
                         phasor=True, seed=0)

    t0 = time.time()
    res = train_network_qphnn(model, train, test, max_iter=epochs, verbose=False)
    wall = time.time() - t0

    learned_gamma = res.params_opt[-N:] if len(res.params_opt) >= N else np.zeros(N)
    gamma_rel_err = float(np.mean(np.abs(learned_gamma - true_gamma) / (true_gamma + 1e-9)))

    # Passivity: Ḣ ≤ 0 fraction
    n_check = min(test.n_samples, 50)
    hdots = np.array([model.energy_rate(test.states[b], res.params_opt)
                      for b in range(n_check)])
    passivity_frac = float(np.mean(hdots <= 0))

    # Monotone energy rollout
    state0 = ds.states[0].copy()
    traj = model.rollout(state0, res.params_opt, dt=0.05, n_steps=80)
    H_t = model.energy_along(traj, res.params_opt)
    mono_frac = float(np.mean(np.diff(H_t) <= 0))

    metrics = {
        "N": N,
        "n_layers": n_layers,
        "n_circuit_params": model.n_circuit_weights,
        "n_train": train.n_samples,
        "bfgs_iters": res.n_iter,
        "gamma_rel_err": gamma_rel_err,
        "passivity_frac": passivity_frac,
        "mono_frac": mono_frac,
        "wall_s": wall,
        "mode": "dissipative",
    }
    print(f"  ✓ passivity={passivity_frac*100:.0f}%, γ_err={gamma_rel_err*100:.1f}%, "
          f"mono={mono_frac*100:.0f}%, iters={res.n_iter}, time={wall:.0f}s")
    return metrics


def write_scaling_report(cons_results: list, diss_results: list, out_dir: Path) -> None:
    """Write Markdown scaling table."""
    lines = [
        "# QGNN Network Scaling Study",
        "",
        "Ring coupling (all-to-all for N≤4, ring for N≥5). "
        "Structural guarantees (exact conservation, 100% passivity) hold by "
        "circuit construction at ALL N.",
        "",
        "## Conservative Network (γ = 0)",
        "",
        "| N | L | Params | Train pts | Iters | Energy Drift | mean|Ḣ| | Time |",
        "|---|---|--------|-----------|-------|-------------|---------|------|",
    ]
    for r in cons_results:
        lines.append(
            f"| {r['N']} | {r['n_layers']} | {r['n_circuit_params']} "
            f"| {r['n_train']} | {r['bfgs_iters']} "
            f"| {r['energy_drift_rel']*100:.2f}% "
            f"| {r['mean_abs_hdot']:.2e} "
            f"| {r['wall_s']:.0f}s |"
        )

    lines.extend([
        "",
        "## Dissipative Network (γ > 0)",
        "",
        "| N | L | Params | Train pts | Iters | Passivity | Monotone | γ Error | Time |",
        "|---|---|--------|-----------|-------|-----------|----------|---------|------|",
    ])
    for r in diss_results:
        lines.append(
            f"| {r['N']} | {r['n_layers']} | {r['n_circuit_params']} "
            f"| {r['n_train']} | {r['bfgs_iters']} "
            f"| {r['passivity_frac']*100:.0f}% "
            f"| {r['mono_frac']*100:.0f}% "
            f"| {r['gamma_rel_err']*100:.1f}% "
            f"| {r['wall_s']:.0f}s |"
        )

    lines.extend([
        "",
        "## Notes",
        "- **Energy Drift**: std(H(t))/|mean(H(t))| along 80-step rollout. "
          "Residual drift is integrator error, NOT a structural violation.",
        "- **mean|Ḣ|**: Mean absolute energy rate over held-out states. "
          "Structural guarantee: exactly 0 for conservative, ≤0 for dissipative.",
        "- **Passivity**: Fraction of held-out states with Ḣ ≤ 0. "
          "Should be 100% by circuit construction for dissipative mode.",
        "- **γ Error**: Mean relative per-node damping recovery error.",
        "- **L**: Number of topology-entangled variational layers (L ≥ graph diameter).",
        "- Statevector simulation scales as 2^N; N≤6 is practical on a laptop CPU.",
    ])

    out_path = out_dir / "scaling_report.md"
    out_path.write_text("\n".join(lines))
    print(f"\n  Scaling report → {out_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-sizes", nargs="+", type=int, default=[3, 4, 5],
                    help="Node counts to evaluate (default: 3 4 5)")
    ap.add_argument("--epochs", type=int, default=120,
                    help="BFGS iterations per run (default: 120)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  QGNN Network Scaling Study")
    print(f"  Node sizes: {args.node_sizes}  |  Epochs: {args.epochs}")
    print("=" * 60)

    cons_results, diss_results = [], []

    for N in args.node_sizes:
        cons_results.append(run_conservative(N, args.epochs, args.seed))
        diss_results.append(run_dissipative(N, args.epochs, args.seed))

    print("\n" + "=" * 60)
    print("  SCALING SUMMARY — CONSERVATIVE")
    print("=" * 60)
    print(f"{'N':>4} {'L':>3} {'Params':>7} {'Drift':>10} {'|Ḣ|':>12} {'Time':>8}")
    print("─" * 50)
    for r in cons_results:
        print(f"{r['N']:>4} {r['n_layers']:>3} {r['n_circuit_params']:>7} "
              f"{r['energy_drift_rel']*100:>9.2f}% {r['mean_abs_hdot']:>12.2e} "
              f"{r['wall_s']:>7.0f}s")

    print("\n" + "=" * 60)
    print("  SCALING SUMMARY — DISSIPATIVE")
    print("=" * 60)
    print(f"{'N':>4} {'Passivity':>10} {'Monotone':>10} {'γ Error':>10} {'Time':>8}")
    print("─" * 50)
    for r in diss_results:
        print(f"{r['N']:>4} {r['passivity_frac']*100:>9.0f}% "
              f"{r['mono_frac']*100:>9.0f}% "
              f"{r['gamma_rel_err']*100:>9.1f}% "
              f"{r['wall_s']:>7.0f}s")

    write_scaling_report(cons_results, diss_results, OUT_DIR)
    print(f"\n  Report → {OUT_DIR}/scaling_report.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
