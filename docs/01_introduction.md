# Chapter 1 — Introduction & Motivation

## 1.1 From classical to quantum Hamiltonian learning

The classical Hamiltonian Neural Network learns a scalar energy `H(x)` and
produces dynamics through the symplectic gradient `ẋ = J∇H`, guaranteeing energy
conservation by the skew-symmetry of `J`. The port-Hamiltonian extension adds
dissipation `R` and forcing $Gu$. The companion project `hnn-generic` implements
exactly this with a classical (graph) neural network.

This project asks the next question: **can we realise the same structure on a
quantum computer, and does quantum mechanics offer something a classical model
cannot?** The answer developed here is yes on both counts:

1. A parameterized quantum circuit can serve as the energy function — its
   Pauli-Z expectation value plays the role of `H`.
2. Unitary evolution is intrinsically energy-preserving, giving the conservative
   (`J`) part "for free."
3. **Measurement** — the one genuinely non-unitary, irreversible operation in
   quantum mechanics — provides a natural, physically-grounded model of
   dissipation that has no direct classical analogue.

## 1.2 Why quantum measurement is the interesting part

Unitary quantum evolution is reversible and conserves probability — it can model
the conservative part of the dynamics but *not* friction. Dissipation requires
irreversibility. In an open quantum system this comes from coupling to a bath and
tracing it out (the Lindblad master equation). This codebase realises that
mechanism concretely:

- Couple the "system" qubit to an **ancilla** qubit (the bath).
- **Measure** the ancilla. By the Born rule the measurement outcome is random,
  and the measurement *collapses* the state — an irreversible, nonlinear
  operation.
- Feed the outcome forward as a conditional kick on the system.

The net effect is that energy leaks from the system in a way that is
irreversible and monotone — exactly the signature of dissipation. We call this
**measurement-induced nonlinearity (MINL)**. It is the quantum heart of the
port-Hamiltonian `R` channel, and it is what distinguishes this work from simply
running a classical HNN on quantum hardware.

## 1.3 The three model families

The codebase builds up in complexity:

1. **Single-DOF conservative** (`QuantumHNN`) — a 2-qubit circuit encoding one
   position and one momentum, learning a conservative Hamiltonian (e.g. the
   nonlinear pendulum). Demonstrates energy-conserving learning on a quantum
   circuit.
2. **Single-DOF dissipative** — two variants:
   - `DynamicQpHNN` — realises dissipation through *actual mid-circuit
     measurement* of an ancilla (MINL, the physically faithful route).
   - `VectorFieldQpHNN` — realises dissipation through a learnable scalar damping
     `γ` added to the parameter-shift vector field (the differentiable route,
     easier to train).
3. **N-node networks** (`QGNNEnergy`, `NetworkQpHNN`) — the upgrade at the centre
   of this project. Each oscillator becomes a qubit; the coupling graph becomes
   the circuit's entanglement graph; the energy is read from a graph-structured
   observable; and dissipation is realised per node, either by an analytic
   damping term or by a multi-ancilla network MINL.

## 1.4 Networks of oscillators as quantum graphs

The scientific target is a *network* of coupled oscillators or phasors — the same
Kuramoto-type systems as in `hnn-generic`. The quantum realisation makes a
structural promise:

> **The circuit's two-qubit gate layout is the adjacency structure of the
> physical network.** An edge between nodes `i` and `j` in the coupling matrix `K`
> becomes a two-qubit entangler between qubits `i` and `j`. Quantum message
> passing over that entanglement graph is the quantum analogue of the classical
> GNN's message passing.

This means the model does not just *approximate* the network dynamics — its very
wiring *is* the network topology. Chapter 5 makes this precise.

## 1.5 What you can do with this codebase

- Encode a classical (q, p) state into a quantum circuit and read back the energy.
- Learn a conservative Hamiltonian on a quantum circuit and verify energy
  conservation (low relative energy drift over a trajectory).
- Learn dissipative dynamics two ways — measurement-induced (MINL) and
  vector-field damping — and verify passivity/monotone energy decay.
- Model an N-node coupled-oscillator network with a topology-entangled quantum
  GNN and recover the coupling structure.

## 1.6 Reading path

Chapter 2 develops the quantum theory (encoding, the isomorphic mapping,
parameter-shift gradients, MINL). If you already know quantum computing and want
to run something, skip to Chapter 8. Developers should read Chapters 4–6.
