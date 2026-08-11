# Chapter 2 — Quantum Theory Background

This chapter develops the physics and quantum mechanics the codebase encodes. It
is self-contained; a one-paragraph recap of the classical side is given first.

## 2.1 Classical port-Hamiltonian recap

A port-Hamiltonian system evolves as

$$
\dot{x} = \big(J - R\big)\nabla H + G u
$$

with $x = (q, p)$ the state, $H(x)$ the energy, $J = -J^{\top}$ the conservative
routing, $R \succeq 0$ the dissipation, and $Gu$ external forcing. The power
balance

$$
\frac{dH}{dt} = -(\nabla H)^{\top} R \, \nabla H + y^{\top} u
$$

shows the $J$ part conserves energy, the $R$ part only removes it (passivity),
and the port supplies or extracts energy. The goal of this project is to realise
each of $H$, $J$, $R$, $G$ with quantum-circuit primitives.

## 2.2 Encoding a classical state into qubits

A single degree of freedom $(q, p)$ is encoded into a **2-qubit** circuit by data
rotation gates:

$$
\text{qubit 0 (position): } R_x(q)
\qquad
\text{qubit 1 (momentum): } R_y(p)
$$

The choice of axes is not arbitrary. The classical coordinates are later read
back from Pauli expectation values through the **Isomorphic Mapping**
(`common/isomorphic_mapping.py`):

$$
q \propto \langle \sigma_x \rangle
\qquad
p \propto \langle \sigma_y \rangle
$$

with linear scale factors $q_{\text{scale}}$, $p_{\text{scale}}$ chosen so the
classical range maps into $[-\pi, \pi]$ (angles) and back. `encode_q/encode_p` multiply by the scale;
`decode_q/decode_p` divide the measured expectation by it. This is the dictionary
that lets a quantum circuit stand in for a classical phase-space point.

## 2.3 The energy observable

The Hamiltonian is read from a Pauli-Z observable of a parameterized circuit. For
the single-DOF models the energy proxy is $\langle ZZ \rangle$ (the two-qubit Z⊗Z expectation);
for networks it is a graph-structured sum (Section 2.7). Training shapes the
circuit's variational angles $\theta$ so that this expectation reproduces the true
energy along trajectories.

Because a Pauli-Z expectation lies in $[-1, 1]$, the model also carries a
classical energy scale so the bounded observable can match physical energy
magnitudes (important for the network models — see Chapter 5).

## 2.4 The Isomorphic Hamiltonian Mapping (IHM)

The organising principle of the whole project is a structure-preserving map from
the classical port-Hamiltonian quadruple to quantum-circuit objects:

```
        classical                 quantum
  ─────────────────────   ─────────────────────────────
     H  (energy)      →   Pauli-Z observable  ⟨Ĥ_θ⟩
     J  (conservative) →   unitary evolution  U_J(θ)
     R  (dissipation) →   measurement-induced nonlinearity (MINL)
     G·u (forcing)    →   single-qubit drive gate
```

The map is *structural*: unitary evolution is norm-preserving, so the `J` channel
is automatically energy-conserving (the quantum echo of $J = -J^{\top}$); measurement
is irreversible, so the `R` channel can only remove energy (the quantum echo of
$R \succeq 0$). The guarantees carry over from the classical structure to the quantum
one. This is Theorem-level content in the manuscript (§03 theory).

## 2.5 Parameter-shift gradients

To learn dynamics we need gradients of circuit expectation values with respect to
the data-encoding angles (to get the vector field) and the variational weights
(to train). Quantum circuits admit *exact* analytic gradients via the
**parameter-shift rule** (`common/parameter_shift.py`):

$$
\frac{\partial \langle O \rangle}{\partial \theta}
= \tfrac{1}{2}\Big[ \langle O \rangle\big(\theta + \tfrac{\pi}{2}\big)
                   - \langle O \rangle\big(\theta - \tfrac{\pi}{2}\big) \Big]
$$

This is not a finite difference — for the rotation gates used here it is exact.
The symplectic vector field is obtained by shifting the *data* gates:

$$
\dot q = \frac{\partial H}{\partial p}
\quad\longrightarrow\quad \text{parameter-shift on the } p\text{-encoding gate}
$$

$$
\dot p = -\frac{\partial H}{\partial q}
\quad\longrightarrow\quad \text{parameter-shift on the } q\text{-encoding gate}
$$

so the conservative dynamics come straight out of the circuit with no numerical
differentiation.

## 2.6 Measurement-induced nonlinearity (MINL)

Dissipation is the physically interesting part. Unitary evolution cannot lose
energy; measurement can. MINL implements the `R` channel through a Born-rule
mid-circuit measurement of an ancilla ("bath") qubit. Per time step
(`DynamicQpHNN`, Chapter 4):

1. **Conservative step (J):** evolve the system qubit by $R_z(\theta_J)$.
2. **Dissipative coupling (R):** entangle system and ancilla with a controlled
   rotation $\mathrm{CRY}(\theta_R)$.
3. **MINL:** measure the ancilla (`bitstr, sv = sv.measure([anc])`). By the Born
   rule the outcome is random and the state collapses — this is the irreversible,
   nonlinear operation.
4. **Feedforward:** if the ancilla returned "1", apply a conditional kick
   $R_x(\theta_{\text{kick}})$ to the system.

Averaged over the Born-rule randomness (many shots), the system's energy decays
monotonically — a genuine dissipative channel. The key diagnostic is the
**energy monotonicity fraction**: the fraction of time steps along which the
measured energy does not increase.

### Why this is a completely-positive channel

Tracing out the measured ancilla realises a completely-positive trace-preserving
(CPTP) map on the system — the discrete-time analogue of a Lindblad dissipator.
This is what grounds MINL as a faithful `R`: it is the standard open-quantum-system
route to irreversibility, not an ad-hoc nonlinearity.

## 2.7 Networks: topology as entanglement

For an N-node network the encoding generalises node-by-node:

- **One system qubit per node.** Node `i`'s state $(\varphi_i, \omega_i)$ is encoded on qubit
  `i` by $R_x(\varphi_i)\,R_y(\omega_i)$. Momentum uses $R_y$, not $R_z$, because $R_z$ commutes
  with the diagonal Z-basis energy observable — which would force $\partial H/\partial \omega_i = 0$ and
  make the momentum dynamics unlearnable. $R_y$ is off-diagonal, so the energy
  genuinely depends on momentum.
- **Coupling graph = entanglement graph.** An edge $(i, j)$ in the coupling
  matrix $K$ places a two-qubit entangler between qubits `i` and `j`. The circuit
  wiring *is* the network topology.
- **Graph-structured energy readout:**

  $$
  \hat H_\theta = \sum_i a_i \langle Z_i \rangle
                 + \sum_{(i,j) \in E} w_{ij} \langle Z_i Z_j \rangle
  $$

  a sum of single-node terms and edge terms over the coupling edges — the quantum
  analogue of a GNN's node + edge energy decomposition.

The conservative flow uses the per-node symplectic gradient (parameter-shift on
each node's data gates, cost $4N$ circuit evaluations, batched into one call);
dissipation is realised either by an analytic per-node damping $\gamma_i$ or by a
**multi-ancilla MINL** (one ancilla per node). Chapter 5 documents both.

## 2.8 Summary

| Classical object | Quantum realisation | Module |
|------------------|---------------------|--------|
| state $(q,p)$ | $R_x(q)$, $R_y(p)$ encoding | `isomorphic_mapping.py` |
| energy $H$ | Pauli-Z expectation $\langle \hat H_\theta \rangle$ | `quantum_hnn.py`, `qgnn_energy.py` |
| conservative $J$ | unitary evolution | all models |
| dissipation $R$ | MINL (ancilla measurement) | `quantum_phnn.py`, `quantum_network_phnn.py` |
| gradient $\partial \langle O \rangle / \partial \theta$ | parameter-shift $\pm\pi/2$ | `parameter_shift.py` |
| coupling $K_{ij}$ | edge entangler i–j | `qgnn_energy.py` |

The next chapter maps these onto the actual code.
