# Chapter 3 — Codebase Overview

The map of the codebase: directory layout, the three model families, and the data
flow from a classical trajectory to a trained quantum model.

## 3.1 Directory layout

```
codes/
├── common/
│   ├── isomorphic_mapping.py   # classical (q,p) ↔ qubit angles / Pauli expvals
│   ├── parameter_shift.py       # exact ±π/2 circuit gradients          (Chapter 2)
│   ├── metrics.py               # energy drift, monotonicity, RMSE       (Chapter 6)
│   ├── visualization.py         # plotting helpers
│   └── report_writer.py         # markdown report writer
├── non_dissipative/
│   ├── quantum_hnn.py           # QuantumHNN — 2-qubit conservative      (Chapter 4)
│   ├── data_generator.py        # NonlinearPendulum, HarmonicOscillator
│   └── trainer.py               # train_qhnn (BFGS + parameter-shift)
├── dissipative/
│   ├── quantum_phnn.py          # DynamicQpHNN (MINL) + VectorFieldQpHNN (Chapter 4)
│   ├── data_generator.py        # DampedHarmonicOscillator, VanDerPol
│   └── trainer.py               # train_dynamic_qphnn, train_vector_field_qphnn
├── network/                     # the N-node upgrade                     (Chapter 5)
│   ├── qgnn_energy.py           # QGNNEnergy — topology-entangled energy
│   ├── quantum_network_phnn.py  # NetworkQpHNN — network pH + MINL
│   ├── pendulum_chain.py        # coupled nonlinear pendulum chain
│   ├── data_generator.py        # ring/modular couplings, Kuramoto fields
│   └── trainer.py               # train_network_qphnn
├── ibm/                         # IBM Quantum access                     (Chapter 7)
│   ├── connection.py            # credential loading + service handle
│   ├── hardware_harness.py      # transpile, submit, calibration snapshot
│   ├── gradients_hw.py          # parameter-shift on hardware
│   ├── minl_dynamic.py          # MINL circuits with mid-circuit measure
│   ├── preflight.py             # staged readiness check (offline → live)
│   └── sample_circuit.py        # Bell smoke test
├── experiments/                 # one directory per experiment
│   ├── qhnn_pendulum/           # run.py + report.md + figures/
│   ├── qphnn_damped/
│   ├── layer_ablation/
│   ├── network_conservative/
│   ├── network_dissipative/
│   ├── network_scaling/
│   └── ibm_hardware/            # run_*.py + results/ (hardware records)
├── tests/                       # 23 tests over the library
├── docs/                        # these chapters
├── legacy/                      # superseded scripts, feasibility notes
├── check_numbers.py             # manuscript number traceability guard
├── pyproject.toml               # pip install -e .
├── LICENSE / NOTICE             # Apache-2.0
└── README.md
```

## 3.2 The three model families

The codebase is organised by increasing complexity, mirroring the manuscript.

| Family | Directory | Qubits | Regime | Key class |
|--------|-----------|--------|--------|-----------|
| Single-DOF conservative | `non_dissipative/` | 2 | conservative | `QuantumHNN` |
| Single-DOF dissipative | `dissipative/` | 2 (+1 anc) | dissipative | `DynamicQpHNN`, `VectorFieldQpHNN` |
| N-node network | `network/` | N (+N anc) | both | `QGNNEnergy`, `NetworkQpHNN` |

The `common/` package is shared by all three: encoding, gradients, metrics.

## 3.3 The objects that flow through the pipeline

1. **A dataset** — a generator produces `states`, the true derivatives
   `d_states` (the training target), and metadata. Network generators also return
   the coupling `K`, damping `gamma`, and true Hamiltonian `H_true` (the
   `NetworkVectorFieldDataset` dataclass).
2. **A model** — one of the classes above. Its core methods are `energy(x)` (the
   circuit expectation) and a vector-field method (`conservative_field` /
   `vector_field`) that returns `ẋ` via parameter-shift.
3. **A training result** — a dataclass (`TrainingResult`,
   `NetworkTrainingResult`, …) holding optimized parameters, loss history, and
   evaluation metrics.

## 3.4 End-to-end data flow

```
  data_generator.py            model (e.g. NetworkQpHNN)        trainer.py
 ┌──────────────────┐  states, ┌────────────────────────┐  MSE ┌──────────────┐
 │ gen_*()           │─d_states─▶│ energy(x) = ⟨Ĥ_θ⟩       │──────▶│ scipy BFGS    │
 │  build K, integrate│  K, γ,   │ vector_field(x)=(J−R)∇H │       │ on field MSE  │
 │  the true field    │  H_true  │  via parameter-shift    │       └──────┬───────┘
 └──────────────────┘          └───────────┬────────────┘              │ θ*
         │                                  │ MINL path (dissipative)    │
         │                                  ▼                            │
         │                          run_minl_trajectory            ◀─────┘
         │                          (Born-rule measurement)
         ▼
  metrics.py + report_writer.py → energy drift, monotonicity, γ recovery → report.md
```

## 3.5 Where each physics guarantee lives

| Guarantee | Enforced in | How |
|-----------|-------------|-----|
| Energy conservation (J) | unitary evolution | norm-preserving circuit; $\langle ZZ \rangle$/$\langle Z \rangle$ observable |
| Passivity / decay (R) | MINL measurement | Born-rule ancilla collapse → CPTP channel |
| Momentum observability | `qgnn_energy.py` encoding | `Ry` (not `Rz`) for momentum |
| Exact gradients | `parameter_shift.py` | ±π/2 shift rule |
| Topology = wiring | `qgnn_energy.py` | edge entanglers only on `K`'s edges |

## 3.6 Statevector simulation

All circuits are simulated classically with Qiskit's `StatevectorEstimator` and
`Statevector` (no hardware backend). This keeps the demos exactly reproducible and
fast at the target scale (N = 3–6 nodes, i.e. up to ~12 qubits including
ancillas). The MINL path uses `Statevector.measure([site])` for the Born-rule
mid-circuit measurement — the correct Qiskit 2.x API.

The next chapters open each model family.
