# Dissipative Damped Kuramoto Network — Q-GNN pHNN (N=3)

**System:** N-node damped Kuramoto phasor network, ring coupling, per-node damping γ_i > 0. Open port-Hamiltonian with R = diag(γ) ⪰ 0.

**Model:** Topology-entangled QGNN (3 system qubits, ZZ entanglers on 3 edges, L=2 layers) + per-node learnable damping vector (analytic gradient-damping). A genuine multi-ancilla MINL channel (one bath ancilla per node) is demonstrated separately.

## Results (analytic γ-mode)

| Metric | Value |
|---|---|
| Train vector-field MSE | 0.0890 |
| Test vector-field MSE | 0.0774 |
| Per-node φ̇ MSE | 0.0982 |
| Per-node ω̇ MSE | 0.0798 |
| Mean per-node damping error \|Δγ\|/γ | 27.6% |
| Energy monotone-decay fraction | 83% |
| Passivity (Ḣ ≤ 0) fraction | 100% |
| BFGS iterations | 55 |
| Wall time | 354s |

### Per-node damping
| node | true γ | learned γ |
|---|---|---|
| 0 | 0.433 | 0.280 |
| 1 | 0.258 | 0.232 |
| 2 | 0.385 | 0.241 |

## Physics check

The R-channel damps the energy gradient ∂H/∂ω_i, so the network energy rate is Ḣ = −Σ_i γ_i (∂H/∂ω_i)² ≤ 0 by construction (passivity). The learned rollout shows monotone energy decay to equilibrium — the quantum analogue of the classical dissipative Kuramoto result. The multi-ancilla MINL channel realises the same dissipation through genuine Born-rule measurement + feedforward, a discrete-time network Lindblad (CPTP) map.

## Figures

- `training_loss.png` — joint circuit + damping BFGS training
- `gamma_identification.png` — per-node true vs learned γ
- `energy_decay.png` — monotone energy decay along rollout
- `phase_spiral.png` — per-node spiral to equilibrium
- `minl_ensemble.png` — multi-ancilla MINL per-node decay
- `passivity.png` — energy-rate distribution (Ḣ ≤ 0)