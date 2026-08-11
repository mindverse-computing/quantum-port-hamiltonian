# Chapter 8 — Getting Started

Install, run something in five minutes, then a worked tutorial that builds a
quantum network model, checks the conservation/passivity guarantees, and runs a
measurement-induced-dissipation trajectory.

## 8.1 Requirements

- Python 3.11
- Qiskit 2.x, NumPy, SciPy, Matplotlib

The circuits are simulated classically (statevector), so no quantum hardware or
account is needed. Everything runs on CPU at the demo scale (N = 3–4 nodes).

## 8.2 Install

```bash
cd hnn-quantum/codes
pip install "qiskit>=2.0" numpy scipy matplotlib pytest
python -m pytest tests/ -q          # QGNN + network tests should pass
```

## 8.3 Five-minute quickstart

Run the smallest network experiment and read the report:

```bash
python experiments/network_conservative/run.py \
    --nodes 3 --epochs 45 --samples 40 --seed 1
# → writes report/network_conservative/{report.md, *.png}
```

Open `report/network_conservative/report.md` — it shows the topology (the coupling
graph, which is also the circuit's entanglement graph), the training loss, and the
energy trace (flat, because the J-channel conserves energy).

## 8.4 Worked tutorial: build, check, dissipate

Run from `codes/`. This builds a 3-node network, verifies the physics guarantees,
and runs a measurement-induced-nonlinearity trajectory.

```python
import numpy as np
from network.qgnn_energy import edges_from_coupling, QGNNEnergy
from network.quantum_network_phnn import NetworkQpHNN

# 1. Define a 3-node ring coupling and derive its edges
K = np.array([[0, 1, 1],
              [1, 0, 1],
              [1, 1, 0]], float)          # triangle
edges = edges_from_coupling(K)             # [(0,1), (0,2), (1,2)]
print("edges (= entangler placements):", edges)

# 2a. A conservative network model — energy must be exactly conserved
m_cons = NetworkQpHNN(num_nodes=3, edges=edges, n_layers=2,
                      dissipative=False, seed=1)
p = m_cons.init_params()
state = np.random.default_rng(0).uniform(-0.5, 0.5, 6)   # [phi(3) | omega(3)]
Hdot = m_cons.energy_rate(state, p)
print("conservative energy rate Hdot:", round(Hdot, 12), "(≈ 0 by construction)")

# 2b. A dissipative network model — energy rate must be ≤ 0 (passivity)
m_diss = NetworkQpHNN(num_nodes=3, edges=edges, n_layers=2,
                      dissipative=True, diss_mode="gamma", seed=2)
pd = m_diss.init_params()
theta, gamma = m_diss.split_params(pd)
print("per-node damping gamma:", np.round(gamma, 3), "(all ≥ 0)")
print("dissipative energy rate Hdot:", round(m_diss.energy_rate(state, pd), 6), "(≤ 0)")

# 3. The scalar energy from the graph-structured observable
E = m_cons.energy(state, p)
print("network energy H_theta(x):", round(E, 4))

# 4. Measurement-induced nonlinearity: run a multi-ancilla MINL trajectory
minl_params = {"theta_J": np.full(3, 0.15),
               "theta_R": np.full(3, 1.1),
               "theta_k": np.full(3, 0.15)}
traj = m_diss.run_minl_trajectory(minl_params, state,
                                  rng=np.random.default_rng(0))
print("MINL trajectory shape (steps, nodes):", traj.shape)
print("node-0 position read-out over time:", np.round(traj[:, 0], 3))
```

What each step shows:

1. **The coupling graph becomes the circuit wiring** — `edges` are exactly where
   two-qubit entanglers are placed.
2. **The guarantees are structural** — the conservative model's energy rate is
   zero to machine precision (the two symplectic terms cancel), and the
   dissipative model's rate is $\le 0$ for *any* weights ($\dot H = -\sum_i \gamma_i (\partial H/\partial \omega_i)^{2}$).
3. **Energy** is the graph-structured Pauli-Z observable `Σaᵢ⟨Zᵢ⟩ +
   Σwᵢⱼ⟨ZᵢZⱼ⟩`.
4. **MINL** runs the Born-rule ancilla-measurement dynamics; the returned array is
   each node's position read-out $\langle X_i \rangle$ over the Trotter steps.

## 8.5 Training on data

To actually fit a network to trajectories:

```python
from network.data_generator import gen_network_dissipative, build_ring_coupling
from network.trainer import train_network_qphnn

K = build_ring_coupling(3, strength=1.0)
train = gen_network_dissipative(K=K, n_samples=40, seed=1)
test  = gen_network_dissipative(K=K, n_samples=20, seed=2)

model = NetworkQpHNN(num_nodes=3, edges=edges_from_coupling(K),
                     n_layers=2, dissipative=True, diss_mode="gamma", seed=1)
result = train_network_qphnn(model, train, test, max_iter=45)
print("final loss:", result)   # NetworkTrainingResult with params + metrics
```

## 8.6 Single-DOF quickstart

For the two-qubit models:

```python
from non_dissipative.quantum_hnn import QuantumHNN
model = QuantumHNN(n_layers=1, seed=42)
theta = model.init_weights()                     # variational weights (or theta_opt after training)
E = model.energy(0.5, 0.3, theta)                # energy at (q=0.5, p=0.3)
qdot, pdot = model.vector_field(0.5, 0.3, theta) # symplectic field by parameter-shift
```

The energy and field methods take the weight vector `theta` explicitly (pass
`init_weights()` for an untrained circuit, or `model.theta_opt` after training).
See `experiments/qhnn_pendulum/run.py` for a full training run.

## 8.7 Where to go next

- To understand *why* measurement gives dissipation, re-read Chapter 2 (§2.6 MINL)
  and Chapter 4 (`DynamicQpHNN`).
- To see how the coupling graph becomes entanglement, Chapter 5 (`QGNNEnergy`).
- To add a new system, follow the network generator contract in Chapter 5 and the
  metric definitions in Chapter 6.

## 8.8 Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `No module named 'network'` | not run from `codes/` | run from `codes/`, or add it to `sys.path` (tests do this) |
| Momentum has no effect on energy | momentum encoded with `Rz` | use `Ry` (already the default in `QGNNEnergy`) |
| Conservative `Ḣ` not zero | `energy_rate` missing the position term | ensure the full chain-rule sum over both blocks (already fixed) |
| Statevector too large | too many qubits | keep $N \le 6$; MINL adds one ancilla per node |
