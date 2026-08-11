# Chapter 5 — Network Models

The centrepiece of the project: lifting the single-DOF quantum models to an N-node
coupled-oscillator network, where the coupling graph becomes the circuit's
entanglement graph. Two classes: `QGNNEnergy` (the energy surrogate) and
`NetworkQpHNN` (the full port-Hamiltonian model). Files in `network/`.

## 5.1 `QGNNEnergy` — topology-entangled energy (`network/qgnn_energy.py`)

A parameterized quantum circuit whose Pauli-Z expectation is a scalar energy
$H_\theta(x)$ for an N-node network. The defining feature: two-qubit entanglers are
placed **only on the edges of the coupling matrix**, so the circuit's wiring is
the network topology.

### Constructor

```python
QGNNEnergy(num_nodes, edges, n_layers=2, phasor=True,
           readout_weights=None, seed=42)
```

- `num_nodes` — N nodes = N system qubits.
- `edges` — undirected coupling edges `(i, j)`, `i < j`. Derive from a coupling
  matrix with `edges_from_coupling(K)`. Entanglers go on these edges only.
- `n_layers` — number of topology-entangled variational layers `L`. Choose
  $L \ge$ graph diameter so every node's energy depends on the whole connected
  network (for the small N = 4 demos, `L = 2` already covers the diameter).
- `phasor` — state encoding: `True` → $R_x(\varphi_i)\,R_y(\omega_i)$; `False` → $R_x(q_i)\,R_y(p_i)$.

> **Why `Ry` for momentum, not `Rz`.** `Rz` commutes with the diagonal Z-basis
> energy observable, which would make $\langle Z_i \rangle$ and $\langle Z_i Z_j \rangle$ independent of momentum
> — forcing $\partial H/\partial \omega_i = 0$ and rendering the momentum dynamics unlearnable. `Ry` is
> off-diagonal, so the energy genuinely depends on momentum. This was a real bug,
> caught and fixed; the constructor docstring documents it.

### The energy observable (Eq. 8)

```
Ĥ_θ  =  Σᵢ aᵢ ⟨Zᵢ⟩  +  Σ₍ᵢ,ⱼ₎∈E wᵢⱼ ⟨Zᵢ Zⱼ⟩
```

single-node terms plus edge terms over the coupling edges — the quantum analogue
of a classical GNN's node + edge energy decomposition. `readout_weights` sets
`{a, w}`; defaults are $a_i = 1$, $w_{ij} = 1$. A trainable classical energy **scale**
(the last weight) multiplies the bounded $\langle \cdot \rangle \in [-1, 1]$ so the energy can reach
physical magnitudes.

### Methods

- `energy(state, theta)` / `energy_batch(states, theta)` — the scalar energy.
- `conservative_field(state, theta)` — the symplectic gradient
  $[\partial H/\partial \omega_i \,;\, -\partial H/\partial \varphi_i]$ by parameter-shift. **All 4N shifted circuits are evaluated
  in one batched estimator call** — the full field costs a single job, not 4N.
- `grad_x1`, `grad_x2` — per-coordinate parameter-shift gradients.
- `init_weights` — random small variational angles (energy scale initialised to 1).

## 5.2 `NetworkQpHNN` — network port-Hamiltonian (`network/quantum_network_phnn.py`)

Wraps `QGNNEnergy` and turns its energy into the network port-Hamiltonian vector
field, with optional dissipation.

### Constructor

```python
NetworkQpHNN(num_nodes, edges, n_layers=2, dissipative=False,
             diss_mode='gamma', phasor=True, readout_weights=None, seed=42)
```

- `dissipative` — include an R-channel (per-node damping) or not.
- `diss_mode ∈ {'gamma', 'minl'}` — analytic damping vs. multi-ancilla MINL.
- Parameter vector: `params = [θ (QGNN weights) | γ_raw (N, if dissipative &
  mode='gamma')]`, with `γᵢ = softplus(γ_rawᵢ) ⪰ 0`.

### The vector field (Eqs. 10–11)

$$
\text{Conservative (}J\text{-channel):}\quad
\dot\varphi_i = \frac{\partial H}{\partial \omega_i}, \qquad
\dot\omega_i = -\frac{\partial H}{\partial \varphi_i}
$$

$$
\text{Dissipative (}\gamma\text{ mode):}\quad
\dot\omega_i \;\mathrel{-}=\; \gamma_i \left(\frac{\partial H}{\partial \omega_i}\right)
$$

A crucial design point, matching the classical port-Hamiltonian reference
implementation: **the R-channel damps
the gradient $\partial H/\partial \omega_i$, not the raw state $\omega_i$.** This is the exact analogue of
$\dot x = (J - R)\nabla H$ with `R = diag(γ)` on the momentum block, and it makes network
passivity hold by construction:

$$
\dot H = -\sum_i \gamma_i \left(\frac{\partial H}{\partial \omega_i}\right)^{2} \;\le\; 0
$$

for *any* weights. (For the true kinetic energy $\partial H/\partial \omega_i = \omega_i$, so it coincides
with the state-damping the Kuramoto generator applies.)

### `energy_rate` — the passivity check

`energy_rate(state, params)` computes the full chain-rule sum
$\dot H = \sum_i (\partial H/\partial \varphi_i)\dot\varphi_i + (\partial H/\partial \omega_i)\dot\omega_i$. Conservative (`γ = 0`): the two J-channel
terms cancel exactly ⇒ `Ḣ = 0`. Dissipative: $\dot H = -\sum_i \gamma_i (\partial H/\partial \omega_i)^{2} \le 0$.

> **Bug history worth knowing.** An earlier `energy_rate` summed only the momentum
> term and dropped the position term, which broke the exact cancellation and
> produced spurious non-zero (even positive) energy rates. The fix — the full
> chain-rule over *both* blocks — is what makes the conservative rate identically
> zero. If you modify the field, update `energy_rate` in lockstep.

### Multi-ancilla network MINL (`diss_mode='minl'`)

`run_minl_trajectory(params_minl, state0, rng)` realises dissipation the
physically faithful way, one ancilla per node. Layout: qubits $0 \dots N{-}1$ are system,
$N \dots 2N{-}1$ are ancillas (ancilla `i` paired with node `i`). Per Trotter step:

1. `U_J` — `Rz` per system qubit + `rzz` entanglers on the coupling edges.
2. `U_R` — `CRY` from each system qubit to its ancilla.
3. **Born-rule measure** each ancilla; conditional `Rx` kick on the node that
   read "1".
4. Read $\langle X_i \rangle$ as the position read-out $\hat q_i(t)$.

`params_minl = {'theta_J':(N,), 'theta_R':(N,), 'theta_k':(N,)}`. Returns a
`(minl_steps, N)` array of node positions over time. Averaged over shots the
per-node energy envelope decays — the network generalisation of single-qubit
MINL.

## 5.3 Topology from a coupling matrix

`edges_from_coupling(K, tol=1e-9)` turns a symmetric coupling matrix into the
undirected edge list the models consume. The network data generator
(`network/data_generator.py`) provides the matrices:

- `build_ring_coupling(N, strength)` — a cycle.
- `build_modular_coupling(clusters, nodes_per_cluster, intra, inter, seed)` —
  block-community structure.

## 5.4 Network datasets

- `gen_network_conservative(K, N, n_samples, seed)` — undamped Kuramoto on the
  N-torus (`γ = 0`, pure J-channel, energy conserved).
- `gen_network_dissipative(K, N, gamma_range, n_samples, seed)` — damped Kuramoto
  (`γ ∈ [0.15, 0.45]` per node, `R = diag(γ) ≻ 0`, energy decays).

Both return a `NetworkVectorFieldDataset` (states, true derivatives, `K`, `gamma`,
`H_true`, name, and the encoding scales). The generator applies a **uniform
symplectic angle scale** to both $\varphi$ and $\omega$ so the true field is the symplectic
gradient of a single Hamiltonian (a non-uniform scale would break the canonical
structure). Derivative targets are computed analytically from the Kuramoto RHS.

The next chapter covers training and the metrics that validate all of this.
