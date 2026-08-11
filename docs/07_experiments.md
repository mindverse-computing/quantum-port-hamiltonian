# Chapter 7 — Experiments

Four experiments span the three model families: two single-DOF (conservative,
dissipative) and two network (conservative, dissipative). All are in
`experiments/` and write reports under `codes/report/`.

## 7.1 Single-DOF experiments

### `qhnn_pendulum/run.py` — conservative

Trains a `QuantumHNN` on the nonlinear pendulum ($H = \tfrac{1}{2}p^{2} + (1 - \cos q)$). The
validation story is energy conservation: the learned circuit produces a flow whose
relative energy drift over a held-out trajectory is a few percent, confirming the
quantum circuit has learned an energy-conserving Hamiltonian.

### `qphnn_damped/run.py` — dissipative

Trains the dissipative models on the damped harmonic oscillator (`γ = 0.3`).
Demonstrates both dissipation routes: the MINL variant (`DynamicQpHNN`) shows a
high energy-monotonicity fraction (measurement-induced energy decay), and the
vector-field variant (`VectorFieldQpHNN`) recovers the damping coefficient with a
few tens of percent relative error at the demo budget.

## 7.2 Network experiments

These are the upgrade at the centre of the project.

### `network_conservative/run.py` — N-torus phasor network

An undamped Kuramoto network (`γ = 0`) modelled by a `NetworkQpHNN` with the
topology-entangled QGNN energy. Demonstrates that the network J-channel conserves
energy: the measured energy rate is identically zero (the two symplectic terms
cancel exactly), and the learned field reproduces the coupled phasor dynamics.

```
--nodes 4        network size N (= system qubits)
--samples 60     training samples
--epochs 120     BFGS iterations
--layers 2       QGNN entanglement layers
--seed 1
```

Figures: topology (the coupling graph = entanglement graph), training loss, phase
portrait, energy trace, and a parity scatter of the predicted vs. true field.

### `network_dissipative/run.py` — damped Kuramoto + network MINL

A damped Kuramoto network (`γ > 0`) with two dissipation demonstrations: the
analytic gamma-channel (trained, with per-node $\hat\gamma_i$ recovery) and the
multi-ancilla network MINL (one ancilla per node, showing per-node energy-envelope
decay).

```
--nodes 4        network size
--samples 60
--epochs 140
--layers 2
--seed 1
--minl-shots 24  Born-rule shots for the MINL ensemble
```

Figures: training loss, per-node damping identification, energy decay, phase-space
trajectories, the multi-ancilla MINL ensemble, and a passivity check.

## 7.3 Scale

Per the demo constraint the network experiments run at **N = 3–4 nodes**,
statevector-simulable (up to ~8 qubits including ancillas for MINL). This is the
"placeholder" scale that produces the manuscript figures quickly. Larger networks
are possible in principle but grow the statevector exponentially, so $N \le 6$ is
the practical ceiling for exact simulation.

## 7.4 Reproducing everything

```bash
cd hnn-quantum/codes
python -m pytest tests/ -q                              # QGNN + network tests
python experiments/qhnn_pendulum/run.py                 # conservative 1-DOF
python experiments/qphnn_damped/run.py                  # dissipative 1-DOF
python experiments/network_conservative/run.py --nodes 3 --epochs 45 --samples 40 --seed 1
python experiments/network_dissipative/run.py  --nodes 3 --epochs 55 --samples 40 --seed 1 --minl-shots 16
```

Reports (figures + `report.md`) are written to `codes/report/<experiment>/`.

## 7.5 What the experiments establish

| Experiment | Claim demonstrated |
|------------|--------------------|
| Q-HNN pendulum | a quantum circuit learns a conservative Hamiltonian (low energy drift) |
| Q-pHNN damped (MINL) | measurement induces genuine, monotone energy dissipation |
| Q-pHNN damped (vector-field) | a learnable γ recovers the physical damping rate |
| Network conservative | topology-entangled QGNN conserves energy on the N-torus (`Ḣ = 0`) |
| Network dissipative | network passivity + per-node damping recovery + multi-ancilla MINL |

Together they trace the arc from a single conservative degree of freedom to a
dissipative quantum network whose wiring is the physical coupling graph.
