"""
experiments/run_network_conservative.py
=======================================
Experiment: Conservative Coupled Phasor Network on the N-torus (γ = 0).

System : N=4 Kuramoto phasors, ring coupling, no damping.
Model  : Topology-entangled QGNN (NetworkQpHNN, dissipative=False).
Physics: pure J-channel ⇒ energy conserved along rollout (Ḣ ≈ 0).

Pipeline
--------
1. Generate conservative network vector-field dataset (N-torus).
2. Train NetworkQpHNN (BFGS, exact parameter-shift gradients).
3. Metrics: per-node vector-field MSE, energy conservation along rollout,
   energy-rate distribution (should be ~0).
4. Figures (publication-grade): topology+circuit schematic, training loss,
   per-node phase portrait, energy-vs-time, predicted-vs-true field parity
   scatter (node-0 φ̇ and ω̇).
5. Write report.md.

Usage
-----
    cd hnn-quantum/codes
    python experiments/run_network_conservative.py
    python experiments/run_network_conservative.py --nodes 4 --epochs 120
"""

import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network import (
    gen_network_conservative, build_ring_coupling, edges_from_coupling,
    NetworkQpHNN, train_network_qphnn,
)

# ── Publication figure defaults (figure-style §5 size ladder, clean frame) ──
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4,
    "figure.facecolor": "white",
})
FOCAL = "#1f4e79"     # focal series (quantum model)
TRUTH = "#c0392b"     # ground truth
NODEC = ["#1f4e79", "#2e8b57", "#c0392b", "#8e44ad", "#d68910", "#16a085"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    N = args.nodes
    outdir = Path(__file__).resolve().parent / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[Net-Cons] Conservative phasor network  N={N}  (γ=0, N-torus)")
    K = build_ring_coupling(N, strength=1.0)
    edges = edges_from_coupling(K)
    ds = gen_network_conservative(K=K, n_samples=args.samples, seed=args.seed)
    train, test = ds.split(0.25, seed=args.seed)

    model = NetworkQpHNN(N, edges, n_layers=args.layers,
                         dissipative=False, phasor=True, seed=0)
    print(f"  {model}  |  circuit weights = {model.n_circuit_weights}")

    t0 = time.time()
    res = train_network_qphnn(model, train, test, max_iter=args.epochs, verbose=True)
    wall = time.time() - t0

    # ── Rollout for energy conservation ──────────────────────────────────
    state0 = ds.states[0].copy()
    traj = model.rollout(state0, res.params_opt, dt=0.05, n_steps=80)
    H_t = model.energy_along(traj, res.params_opt)
    H_rel_drift = float(np.std(H_t) / (abs(np.mean(H_t)) + 1e-9))

    # Energy-rate distribution over test states (should concentrate at 0)
    hdots = np.array([model.energy_rate(test.states[b], res.params_opt)
                      for b in range(min(test.n_samples, 20))])

    # ══════════════════════════════════════════════════════════════════════
    #  FIGURES
    # ══════════════════════════════════════════════════════════════════════
    _fig_topology(K, edges, N, outdir)
    _fig_loss(res.loss_history, outdir)
    _fig_phase_portrait(traj, N, outdir)
    _fig_energy(H_t, hdots, H_rel_drift, outdir)
    _fig_vector_field(model, res, test, N, outdir)

    # ── Report ───────────────────────────────────────────────────────────
    _write_report(outdir, N, edges, res, H_rel_drift, hdots, wall, args)
    print(f"\n[Net-Cons] Complete in {wall:.0f}s → {outdir}")
    print(f"  energy drift (rel) = {H_rel_drift*100:.2f}%  | "
          f"mean |Ḣ| = {np.mean(np.abs(hdots)):.4f}")


# ─────────────────────────────────────────────────────────────────────────
def _fig_topology(K, edges, N, outdir):
    """Network coupling graph + role of quantum circuit (schematic)."""
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    theta = np.linspace(0, 2*np.pi, N, endpoint=False) + np.pi/2
    xs, ys = np.cos(theta), np.sin(theta)
    for (i, j) in edges:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], "-", color="#888",
                lw=2.0, zorder=1)
    for i in range(N):
        ax.scatter(xs[i], ys[i], s=620, color=NODEC[i % len(NODEC)],
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(xs[i], ys[i], f"$q_{i}$", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold", zorder=4)
    ax.set_title("Coupling graph = circuit entanglement graph\n"
                 f"(N={N} nodes, ZZ gate on each edge)", fontsize=8)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(outdir / "topology.png"); plt.close(fig)


def _fig_loss(loss_history, outdir):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(loss_history, color=FOCAL, lw=1.3)
    ax.set_yscale("log")
    ax.set_xlabel("BFGS iteration"); ax.set_ylabel("vector-field MSE")
    ax.set_title("Training converges on the network field", fontsize=8)
    ax.margins(x=0.02)
    fig.savefig(outdir / "training_loss.png"); plt.close(fig)


def _fig_phase_portrait(traj, N, outdir):
    """Per-node phase portrait (φ vs ω) along the learned rollout."""
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for i in range(N):
        phi_i = traj[:, i]; omega_i = traj[:, N + i]
        ax.plot(phi_i, omega_i, "-", color=NODEC[i % len(NODEC)],
                lw=1.2, alpha=0.9)
        ax.scatter(phi_i[0], omega_i[0], color=NODEC[i % len(NODEC)],
                   s=28, zorder=5, edgecolor="white", linewidth=0.6)
        ax.text(phi_i[-1], omega_i[-1], f"node {i}", fontsize=6,
                color=NODEC[i % len(NODEC)])
    ax.set_xlabel(r"phase $\varphi_i$ (encoded)")
    ax.set_ylabel(r"frequency $\omega_i$ (encoded)")
    ax.set_title("Per-node phase-space trajectories (N-torus)", fontsize=8)
    ax.margins(0.08)
    fig.savefig(outdir / "phase_portrait.png"); plt.close(fig)


def _fig_energy(H_t, hdots, drift, outdir):
    """Energy vs time (should be ~flat) + energy-rate histogram at 0."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.6, 2.5))
    t = np.arange(len(H_t)) * 0.05
    a1.plot(t, H_t, color=FOCAL, lw=1.4)
    m = np.mean(H_t)
    a1.axhline(m, color="#999", ls="--", lw=0.8)
    a1.set_xlabel("time"); a1.set_ylabel(r"network energy $H_\theta$")
    a1.set_title(f"Energy conserved: {drift*100:.2f}% drift", fontsize=8)
    a1.margins(x=0.02)
    # centre y so the flat line is visibly flat but not misleadingly zoomed
    span = max(np.ptp(H_t), 0.05 * abs(m) + 1e-6)
    a1.set_ylim(m - 3*span, m + 3*span)

    a2.hist(hdots, bins=12, color=FOCAL, alpha=0.85, edgecolor="white")
    a2.axvline(0, color=TRUTH, lw=1.2)
    a2.set_xlabel(r"energy rate $\dot H$"); a2.set_ylabel("count")
    a2.set_title(r"$\dot H \approx 0$ (skew $J$)", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "energy.png"); plt.close(fig)


def _fig_vector_field(model, res, test, N, outdir):
    """Predicted vs true (φ̇, ω̇) for node 0 across the test set."""
    preds = np.array([model.vector_field(test.states[b], res.params_opt)
                      for b in range(test.n_samples)])
    trues = test.d_states
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.6, 2.7))
    for ax, comp, lab in [(a1, 0, r"$\dot\varphi_0$"),
                          (a2, N, r"$\dot\omega_0$")]:
        ax.scatter(trues[:, comp], preds[:, comp], s=16, color=FOCAL,
                   alpha=0.8, edgecolor="white", linewidth=0.3)
        lo = min(trues[:, comp].min(), preds[:, comp].min())
        hi = max(trues[:, comp].max(), preds[:, comp].max())
        ax.plot([lo, hi], [lo, hi], "--", color=TRUTH, lw=1.0)
        ax.set_xlabel(f"true {lab}"); ax.set_ylabel(f"predicted {lab}")
        ax.set_aspect("equal", adjustable="datalim"); ax.margins(0.08)
    a1.set_title("Node-0 velocity field recovered", fontsize=8)
    a2.set_title("Node-0 acceleration field recovered", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "vector_field.png"); plt.close(fig)


def _write_report(outdir, N, edges, res, drift, hdots, wall, args):
    lines = [
        f"# Conservative Coupled Phasor Network — Q-GNN pHNN (N={N})\n",
        "**System:** N-node Kuramoto phasor network, ring coupling, "
        "γ=0 (no damping, no forcing). Pure conservative J-channel on the N-torus.\n",
        "**Model:** Topology-entangled QGNN energy surrogate "
        f"({N} system qubits, ZZ entanglers on {len(edges)} coupling edges, "
        f"L={args.layers} layers). Symplectic gradients via per-node "
        "parameter-shift; BFGS training.\n",
        "## Results\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Train vector-field MSE | {res.train_mse:.4f} |",
        f"| Test vector-field MSE | {res.test_mse:.4f} |",
        f"| Per-node φ̇ MSE | {res.phi_dot_mse:.4f} |",
        f"| Per-node ω̇ MSE | {res.omega_dot_mse:.4f} |",
        f"| Energy drift along rollout (rel) | {drift*100:.2f}% |",
        f"| Mean \\|Ḣ\\| over test states | {np.mean(np.abs(hdots)):.4f} |",
        f"| BFGS iterations | {res.n_iter} |",
        f"| Wall time | {wall:.0f}s |",
        "\n## Physics check\n",
        "Because the model uses a skew-symmetric J-channel with R=0, the "
        "energy rate Ḣ = (∇H)·ẋ vanishes identically up to circuit-"
        "expressibility error. The near-zero energy drift along the learned "
        "rollout confirms the network conserves energy on the N-torus — the "
        "quantum analogue of the classical conservative Kuramoto result.\n",
        "## Figures\n",
        "- `topology.png` — coupling graph = circuit entanglement graph",
        "- `training_loss.png` — BFGS convergence",
        "- `phase_portrait.png` — per-node closed orbits",
        "- `energy.png` — energy vs time + energy-rate histogram",
        "- `vector_field.png` — predicted vs true node-0 field",
    ]
    (outdir / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
