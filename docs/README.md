# Quantum Port-Hamiltonian Neural Networks — Documentation

A guided tour of the `hnn-quantum` codebase: how classical Hamiltonian and
port-Hamiltonian dynamics are lifted onto quantum circuits, how dissipation is
realised through *measurement-induced nonlinearity*, and how a network of coupled
oscillators is modelled with a topology-entangled quantum graph neural network.

This is the quantum companion to `hnn-generic`. Where the generic project learns
oscillator dynamics with a classical neural network under a structural
port-Hamiltonian constraint, this project asks: **what is the quantum-circuit
analogue of that structure, and what does quantum measurement buy us that a
classical model cannot express?**

## Who this is for
- **Readers new to the subject** — start at Chapter 1, then Chapter 2. Assumes
  basic quantum-computing literacy (qubits, gates, measurement) and the classical
  port-Hamiltonian idea (a one-paragraph recap is given).
- **Users who want to run something** — Chapter 8 (Getting Started), then
  Chapter 7 (Experiments).
- **Developers extending the code** — Chapters 4–6 are the module reference.

## Table of contents

| # | Chapter | What it covers |
|---|---------|----------------|
| 1 | [Introduction & Motivation](01_introduction.md) | Why quantum HNNs; classical→quantum lift; measurement as nonlinearity |
| 2 | [Quantum Theory Background](02_theory.md) | Qubit encoding of (q,p); the Isomorphic Hamiltonian Mapping; parameter-shift gradients; MINL & the Lindblad picture |
| 3 | [Codebase Overview](03_codebase_overview.md) | Directory layout, the three model families, data flow |
| 4 | [Single-DOF Models](04_single_dof_models.md) | `QuantumHNN` (conservative), `DynamicQpHNN` & `VectorFieldQpHNN` (dissipative) |
| 5 | [Network Models](05_network_models.md) | `QGNNEnergy`, `NetworkQpHNN`, topology entanglement, network MINL |
| 6 | [Data, Training & Metrics](06_data_training_metrics.md) | Generators, parameter-shift trainers, energy/monotonicity metrics |
| 7 | [Experiments](07_experiments.md) | Pendulum, damped oscillator, conservative & dissipative networks |
| 8 | [Getting Started](08_getting_started.md) | Install, quickstart, a worked tutorial |

## Relationship to the manuscript

The `manuscript/latex/sections/` chapters (introduction, background, theory,
architecture, methods, experiments, discussion, conclusion) are the formal write-up.
These docs are the *implementation-facing* companion: they explain the same ideas
in code terms and point at the exact modules that realise each equation.

## Conventions
- **1-DOF systems** use 2 qubits (position and momentum). **N-node networks** use
  one system qubit per node (plus one ancilla per node for network MINL).
- Energy is read from Pauli-Z expectation values; the classical state is recovered
  from $\langle \sigma_x \rangle$ (position) and $\langle \sigma_y \rangle$ (momentum) via the Isomorphic Mapping — this is
  the codebase's *read-out* convention (`decode_q` $\leftarrow \langle \sigma_x \rangle$, `decode_p` $\leftarrow \langle \sigma_y \rangle$,
  applied after circuit evolution), which is distinct from the *encoding* axes
  (`Rx(q)` on the position qubit, `Ry(p)` on the momentum qubit).
- Code paths are relative to `hnn-quantum/codes/`.
- Built on **Qiskit 2.x** (`StatevectorEstimator`, `Statevector.measure`).

## One-paragraph summary
A classical port-Hamiltonian system $\dot x = (J - R)\nabla H + Gu$ is mapped to a quantum
circuit by the **Isomorphic Hamiltonian Mapping**: the energy `H` becomes a
Pauli-Z observable of a parameterized circuit, the conservative routing `J`
becomes unitary evolution, and the dissipation `R` becomes *measurement-induced
nonlinearity* (MINL) — a Born-rule mid-circuit measurement of an ancilla that
irreversibly removes energy, the quantum realisation of friction. For a network,
each node is a qubit, the coupling graph becomes the circuit's entanglement graph,
and the energy is read from a graph-structured observable. The result is a family
of quantum neural networks that learn conservative and dissipative dynamics while
respecting the underlying physical structure.
