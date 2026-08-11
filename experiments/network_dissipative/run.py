"""
experiments/run_network_dissipative.py
=======================================
Experiment: Dissipative Damped Kuramoto Network (γ > 0) + MINL.

System : N=4 damped Kuramoto phasors, ring coupling, per-node damping γ_i > 0.
Model  : Topology-entangled QGNN (NetworkQpHNN, dissipative=True):
           - gamma mode : analytic gradient-damping, learnable γ vector (BFGS)
           - minl  mode : genuine multi-ancilla measurement-induced nonlinearity
Physics: R = diag(γ) ⪰ 0 ⇒ energy decays monotonically (passivity, Ḣ ≤ 0).

Pipeline
--------
1. Generate dissipative network vector-field dataset.
2. Train NetworkQpHNN (gamma mode) — recover per-node damping γ_i.
3. Metrics: per-node γ identification error, energy monotone-decay fraction,
   vector-field MSE, energy-rate ≤ 0 (passivity).
4. MINL channel demo: multi-ancilla trajectory ensemble, show mean decay.
5. Figures: energy monotone decay, per-node γ true-vs-learned bar,
   phase portrait spiralling in, MINL trajectory ensemble, training loss.
6. Write report.md.

Usage
-----
    cd hnn-quantum/codes
    python experiments/run_network_dissipative.py --nodes 4 --epochs 140
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
    gen_network_dissipative, build_ring_coupling, edges_from_coupling,
    NetworkQpHNN, train_network_qphnn,
)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4, "figure.facecolor": "white",
})
FOCAL = "#1f4e79"; TRUTH = "#c0392b"
NODEC = ["#1f4e79", "#2e8b57", "#c0392b", "#8e44ad", "#d68910", "#16a085"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--minl-shots", type=int, default=24)
    args = ap.parse_args()

    N = args.nodes
    outdir = Path(__file__).resolve().parent / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[Net-Diss] Dissipative Kuramoto network  N={N}  (γ>0)")
    K = build_ring_coupling(N, strength=1.0)
    edges = edges_from_coupling(K)
    ds = gen_network_dissipative(K=K, n_samples=args.samples, seed=args.seed)
    train, test = ds.split(0.25, seed=args.seed)
    print(f"  true γ = {np.round(ds.gamma, 3)}")

    # ── Analytic gamma-mode training ─────────────────────────────────────
    model = NetworkQpHNN(N, edges, n_layers=args.layers, dissipative=True,
                         diss_mode="gamma", phasor=True, seed=0)
    print(f"  {model}  |  n_params = {model.n_params}")
    t0 = time.time()
    res = train_network_qphnn(model, train, test, max_iter=args.epochs, verbose=True)
    wall = time.time() - t0

    # ── Rollout: energy should decay monotonically ───────────────────────
    state0 = ds.states[0].copy()
    traj = model.rollout(state0, res.params_opt, dt=0.05, n_steps=100)
    H_t = model.energy_along(traj, res.params_opt)
    # monotone-decay fraction
    mono = float(np.mean(np.diff(H_t) <= 1e-9))
    # passivity: energy-rate over test states must be <= 0
    hdots = np.array([model.energy_rate(test.states[b], res.params_opt)
                      for b in range(min(test.n_samples, 20))])
    passivity_frac = float(np.mean(hdots <= 1e-9))

    # ── MINL channel demo (genuine multi-ancilla dissipation) ────────────
    minl = NetworkQpHNN(N, edges, n_layers=1, dissipative=True,
                        diss_mode="minl", phasor=True, seed=0)
    minl.minl_steps = 8
    params_minl = {
        "theta_J": np.full(N, 0.15),     # gentle conservative rotation
        "theta_R": np.full(N, 1.1),      # strong system-bath coupling
        "theta_k": np.full(N, 0.15),
    }
    x0 = np.zeros(2 * N)
    x0[:N] = np.pi / 2                    # start with max ⟨X⟩
    minl_mean = minl.predict_minl(params_minl, x0, n_shots=args.minl_shots)

    # ══════════════════════════════════════════════════════════════════════
    #  FIGURES
    # ══════════════════════════════════════════════════════════════════════
    _fig_loss(res.loss_history, outdir)
    _fig_gamma_bar(ds.gamma, res.learned_gamma, N, outdir)
    _fig_energy_decay(H_t, mono, outdir)
    _fig_phase_spiral(traj, N, outdir)
    _fig_minl(minl_mean, N, outdir)
    _fig_passivity(hdots, passivity_frac, outdir)

    _write_report(outdir, N, edges, res, ds.gamma, mono, passivity_frac,
                  wall, args)
    print(f"\n[Net-Diss] Complete in {wall:.0f}s → {outdir}")
    print(f"  γ rel-err = {res.gamma_rel_err*100:.1f}%  | "
          f"monotone decay = {mono*100:.0f}%  | passivity = {passivity_frac*100:.0f}%")


# ─────────────────────────────────────────────────────────────────────────
def _fig_loss(loss_history, outdir):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(loss_history, color=FOCAL, lw=1.3)
    ax.set_yscale("log")
    ax.set_xlabel("BFGS iteration"); ax.set_ylabel("vector-field MSE")
    ax.set_title("Joint circuit + damping training", fontsize=8)
    ax.margins(x=0.02)
    fig.savefig(outdir / "training_loss.png"); plt.close(fig)


def _fig_gamma_bar(true_g, learned_g, N, outdir):
    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    x = np.arange(N); w = 0.38
    ax.bar(x - w/2, true_g, w, label="true $\\gamma_i$", color=TRUTH,
           edgecolor="white")
    ax.bar(x + w/2, learned_g, w, label="learned $\\gamma_i$", color=FOCAL,
           edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([f"node {i}" for i in range(N)])
    ax.set_ylabel(r"damping $\gamma_i$")
    ax.set_title("Per-node damping identification", fontsize=8)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(outdir / "gamma_identification.png"); plt.close(fig)


def _fig_energy_decay(H_t, mono, outdir):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    t = np.arange(len(H_t)) * 0.05
    ax.plot(t, H_t, color=FOCAL, lw=1.5)
    ax.set_xlabel("time"); ax.set_ylabel(r"network energy $H_\theta$")
    ax.set_title(f"Monotone energy decay ({mono*100:.0f}% of steps)", fontsize=8)
    ax.margins(x=0.02)
    fig.savefig(outdir / "energy_decay.png"); plt.close(fig)


def _fig_phase_spiral(traj, N, outdir):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for i in range(N):
        phi_i = traj[:, i]; omega_i = traj[:, N + i]
        ax.plot(phi_i, omega_i, "-", color=NODEC[i % len(NODEC)],
                lw=1.1, alpha=0.9)
        ax.scatter(phi_i[0], omega_i[0], color=NODEC[i % len(NODEC)],
                   s=30, zorder=5, edgecolor="white", linewidth=0.6, marker="o")
        ax.scatter(phi_i[-1], omega_i[-1], color=NODEC[i % len(NODEC)],
                   s=42, zorder=6, edgecolor="black", linewidth=0.6, marker="X")
    ax.set_xlabel(r"phase $\varphi_i$ (encoded)")
    ax.set_ylabel(r"frequency $\omega_i$ (encoded)")
    ax.set_title("Per-node phase-space trajectories (○ start, ✕ end)", fontsize=8)
    ax.margins(0.08)
    fig.savefig(outdir / "phase_spiral.png"); plt.close(fig)


def _fig_minl(minl_mean, N, outdir):
    """Ensemble-mean ⟨X_i⟩(t) from multi-ancilla MINL — genuine dissipation."""
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    steps = np.arange(minl_mean.shape[0])
    for i in range(N):
        ax.plot(steps, minl_mean[:, i], "-o", color=NODEC[i % len(NODEC)],
                ms=3, lw=1.1, label=f"node {i}")
    ax.axhline(0, color="#bbb", lw=0.6, ls=":")
    ax.set_xlabel("Trotter step"); ax.set_ylabel(r"$\langle X_i\rangle$ (position)")
    ax.set_title("Multi-ancilla MINL: measurement-induced dynamics", fontsize=8)
    ax.legend(frameon=False, ncol=3, fontsize=6)
    ax.margins(x=0.03)
    fig.savefig(outdir / "minl_ensemble.png"); plt.close(fig)


def _fig_passivity(hdots, frac, outdir):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.hist(hdots, bins=12, color=FOCAL, alpha=0.85, edgecolor="white")
    ax.axvline(0, color=TRUTH, lw=1.2)
    ax.set_xlabel(r"energy rate $\dot H$"); ax.set_ylabel("count")
    ax.set_title(f"Passivity: Ḣ ≤ 0 on {frac*100:.0f}% of states", fontsize=8)
    fig.savefig(outdir / "passivity.png"); plt.close(fig)


def _write_report(outdir, N, edges, res, true_g, mono, passivity, wall, args):
    lines = [
        f"# Dissipative Damped Kuramoto Network — Q-GNN pHNN (N={N})\n",
        "**System:** N-node damped Kuramoto phasor network, ring coupling, "
        "per-node damping γ_i > 0. Open port-Hamiltonian with R = diag(γ) ⪰ 0.\n",
        "**Model:** Topology-entangled QGNN "
        f"({N} system qubits, ZZ entanglers on {len(edges)} edges, "
        f"L={args.layers} layers) + per-node learnable damping vector "
        "(analytic gradient-damping). A genuine multi-ancilla MINL channel "
        "(one bath ancilla per node) is demonstrated separately.\n",
        "## Results (analytic γ-mode)\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Train vector-field MSE | {res.train_mse:.4f} |",
        f"| Test vector-field MSE | {res.test_mse:.4f} |",
        f"| Per-node φ̇ MSE | {res.phi_dot_mse:.4f} |",
        f"| Per-node ω̇ MSE | {res.omega_dot_mse:.4f} |",
        f"| Mean per-node damping error \\|Δγ\\|/γ | {res.gamma_rel_err*100:.1f}% |",
        f"| Energy monotone-decay fraction | {mono*100:.0f}% |",
        f"| Passivity (Ḣ ≤ 0) fraction | {passivity*100:.0f}% |",
        f"| BFGS iterations | {res.n_iter} |",
        f"| Wall time | {wall:.0f}s |",
        "\n### Per-node damping",
        "| node | true γ | learned γ |",
        "|---|---|---|",
    ]
    for i in range(N):
        lines.append(f"| {i} | {true_g[i]:.3f} | {res.learned_gamma[i]:.3f} |")
    lines += [
        "\n## Physics check\n",
        "The R-channel damps the energy gradient ∂H/∂ω_i, so the network "
        "energy rate is Ḣ = −Σ_i γ_i (∂H/∂ω_i)² ≤ 0 by construction "
        "(passivity). The learned rollout shows monotone energy decay to "
        "equilibrium — the quantum analogue of the classical dissipative "
        "Kuramoto result. The multi-ancilla MINL channel realises the same "
        "dissipation through genuine Born-rule measurement + feedforward, a "
        "discrete-time network Lindblad (CPTP) map.\n",
        "## Figures\n",
        "- `training_loss.png` — joint circuit + damping BFGS training",
        "- `gamma_identification.png` — per-node true vs learned γ",
        "- `energy_decay.png` — monotone energy decay along rollout",
        "- `phase_spiral.png` — per-node spiral to equilibrium",
        "- `minl_ensemble.png` — multi-ancilla MINL per-node decay",
        "- `passivity.png` — energy-rate distribution (Ḣ ≤ 0)",
    ]
    (outdir / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
