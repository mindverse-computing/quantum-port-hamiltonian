# Chapter 4 — Single-DOF Models

The two-qubit models: one conservative, two dissipative. These establish the
quantum-circuit realisation of Hamiltonian and port-Hamiltonian dynamics for a
single degree of freedom before the network generalisation in Chapter 5.

## 4.1 `QuantumHNN` — conservative (`non_dissipative/quantum_hnn.py`)

The quantum Hamiltonian Neural Network for one conservative degree of freedom.

### Constructor

```python
QuantumHNN(n_layers=1, seed=42)
```

- 2 qubits: qubit 0 encodes position `q` via `Rx(q_in)`, qubit 1 encodes momentum
  `p` via `Ry(p_in)`.
- `n_weights = 4 * n_layers` trainable variational angles.
- Energy observable: `SparsePauliOp("ZZ")` — the two-qubit $\langle Z \otimes Z \rangle$ expectation is
  the energy proxy `H(q, p)`.
- Evaluated with a `StatevectorEstimator`.

### The ansatz

Per entanglement layer the circuit applies

```
CZ(0,1) → Ry(θ, 0) → Ry(θ, 1) → CZ(0,1) → Rx(θ, 0) → Rx(θ, 1)
```

The `CZ` gates entangle position and momentum; the trainable single-qubit
rotations shape the learned energy surface.

### Dynamics by parameter-shift

The symplectic vector field comes directly from the circuit:

```
q̇ =  ∂H/∂p   ← parameter-shift (±π/2) on the p_in data gate
ṗ = −∂H/∂q   ← parameter-shift (±π/2) on the q_in data gate
```

No dissipation term is present, so the flow is purely conservative. Trained with
BFGS on the vector-field MSE (`train_qhnn`). The headline validation metric is the
**relative energy drift** over a held-out trajectory — for the nonlinear pendulum
it is a few percent, confirming the circuit learns an energy-conserving flow.

## 4.2 `DynamicQpHNN` — dissipative via MINL (`dissipative/quantum_phnn.py`)

The physically faithful dissipative model: friction is realised by *actual
mid-circuit measurement* of an ancilla (bath) qubit — measurement-induced
nonlinearity.

### Constructor

```python
DynamicQpHNN(n_steps=6, seed=42)
```

- 2 qubits: qubit 0 = system (`SYS_IDX = 0`), qubit 1 = ancilla/bath
  (`ANC_IDX = 1`), in Qiskit's LSB convention.
- `n_steps` discrete time steps per trajectory.

### One time step

The state is carried as a `Statevector` `sv`, and each step applies:

1. **Conservative (J):** `sv = sv.evolve(Rz(θ_J))` on the system qubit.
2. **Dissipative coupling (R):** `sv = sv.evolve(CRY(θ_R))` — a controlled
   rotation that entangles system and ancilla.
3. **MINL:** `bitstr, sv = sv.measure([ANC_IDX])` — a Born-rule measurement of the
   ancilla. The outcome is random and the state collapses. This is the
   irreversible, nonlinear operation.
4. **Feedforward:** `if "1" in bitstr: sv = sv.evolve(Rx(θ_kick))` — a conditional
   kick on the system.

The classical trajectory is read out via $\langle \sigma_x \rangle$ (position) and $\langle \sigma_y \rangle$ (momentum)
through the Isomorphic Mapping. Because step 3 is stochastic, trajectories are
averaged over many shots (`n_shots`). The diagnostic is the **energy monotonicity
fraction** — the fraction of steps along which the measured energy does not
increase — which approaches 1 for a genuinely dissipative channel.

> This is the model that most directly demonstrates the project's thesis:
> dissipation as a consequence of quantum measurement, not a hand-added damping
> term.

## 4.3 `VectorFieldQpHNN` — dissipative via learnable damping (`dissipative/quantum_phnn.py`)

The differentiable dissipative model: it reuses the conservative `QuantumHNN`
ansatz and adds a *classical scalar* damping `γ` to the momentum equation.

### Constructor

```python
VectorFieldQpHNN(n_layers=1, seed=42)
```

- Same 2-qubit $\langle ZZ \rangle$ energy ansatz as `QuantumHNN` (`4 * n_layers` circuit
  weights).
- Plus one learnable damping scalar `γ`.

### Dynamics

```
q̇ =  ∂H/∂p              (parameter-shift on p_in)
ṗ = −∂H/∂q − γ·p        (parameter-shift on q_in, plus classical damping)
```

Both $\theta$ (circuit) and `γ` (damping) are trained jointly by BFGS on the
vector-field MSE (`train_vector_field_qphnn`). Because everything is
differentiable, this variant trains more easily than the MINL variant; the
validation metric is the **relative error in the recovered damping coefficient**
$|\hat\gamma - \gamma|/\gamma$ (a few tens of percent at the demo budget) plus the energy
monotonicity of the learned flow.

### MINL vs. vector-field — which to use

| | `DynamicQpHNN` (MINL) | `VectorFieldQpHNN` |
|-|----------------------|--------------------|
| Dissipation | real ancilla measurement | learnable scalar γ |
| Physical faithfulness | high (CPTP channel) | lower (added term) |
| Trainability | harder (stochastic) | easier (differentiable) |
| Use when | demonstrating MINL | fitting a damping rate |

## 4.4 Datasets used

- `non_dissipative/data_generator.py` — `NonlinearPendulum` ($H = \tfrac{1}{2}p^{2} + (1 - \cos q)$),
  `HarmonicOscillator`.
- `dissipative/data_generator.py` — `DampedHarmonicOscillator` (`k = 1, m = 1,
  γ = 0.3`), `VanDerPolOscillator` (`μ = 0.5`).

Each returns a vector-field dataset (states + true derivatives) for training, and
the dissipative ones also carry the true damping `γ` for the recovery metric.

The next chapter generalises all of this to N-node networks.
