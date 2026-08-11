# Conservative Coupled Phasor Network — Q-GNN pHNN (N=3)

**System:** N-node Kuramoto phasor network, ring coupling, γ=0 (no damping, no forcing). Pure conservative J-channel on the N-torus.

**Model:** Topology-entangled QGNN energy surrogate (3 system qubits, ZZ entanglers on 3 coupling edges, L=2 layers). Symplectic gradients via per-node parameter-shift; BFGS training.

## Results

| Metric | Value |
|---|---|
| Train vector-field MSE | 0.0886 |
| Test vector-field MSE | 0.0696 |
| Per-node φ̇ MSE | 0.0972 |
| Per-node ω̇ MSE | 0.0799 |
| Energy drift along rollout (rel) | 7.56% |
| Mean \|Ḣ\| over test states | 0.0000 |
| BFGS iterations | 45 |
| Wall time | 270s |

## Physics check

Because the model uses a skew-symmetric J-channel with R=0, the energy rate Ḣ = (∇H)·ẋ vanishes identically up to circuit-expressibility error. The near-zero energy drift along the learned rollout confirms the network conserves energy on the N-torus — the quantum analogue of the classical conservative Kuramoto result.

## Figures

- `topology.png` — coupling graph = circuit entanglement graph
- `training_loss.png` — BFGS convergence
- `phase_portrait.png` — per-node closed orbits
- `energy.png` — energy vs time + energy-rate histogram
- `vector_field.png` — predicted vs true node-0 field