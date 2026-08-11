"""
network/qgnn_energy.py
======================
Topology-Entangled Quantum Graph Neural Network (QGNN) energy surrogate.

Lifts the single-DOF 2-qubit Q-HNN energy ansatz to an N-node network:
one system qubit per oscillator/phasor node, parameterised two-qubit
entanglers placed ONLY on the edges of the coupling matrix K (quantum
message passing), and a graph-structured energy readout

    H_θ(x) = Σ_i a_i ⟨Z_i⟩ + Σ_{(i,j)∈E} w_ij ⟨Z_i Z_j⟩ .

Per-node symplectic gradients are obtained exactly by the parameter-shift
rule on the per-node data-encoding gates.

See theory/network_qphnn_theory.md §3–§4 for the derivation.

Qiskit 2.x API
--------------
- StatevectorEstimator.run([(circuit, observable, values)]) for energy.
- circuit.parameters is sorted alphabetically; we bind by an explicit
  Parameter→value map reduced to that sorted order.
"""

from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator


def edges_from_coupling(K: np.ndarray, tol: float = 1e-9) -> list[tuple[int, int]]:
    """
    Extract the undirected edge list (i<j) of a coupling matrix K.

    An edge (i,j) is present iff |K_ij| + |K_ji| > tol.
    """
    N = K.shape[0]
    edges: list[tuple[int, int]] = []
    for i in range(N):
        for j in range(i + 1, N):
            if abs(K[i, j]) + abs(K[j, i]) > tol:
                edges.append((i, j))
    return edges


class QGNNEnergy:
    """
    Topology-entangled QGNN scalar energy surrogate H_θ(x) for an N-node network.

    Parameters
    ----------
    num_nodes : int
        Number of network nodes N (= number of system qubits).
    edges : list[tuple[int,int]]
        Undirected coupling edges (i<j). Two-qubit entanglers are placed on
        these edges only — this hard-encodes the network topology into the
        ansatz. Use `edges_from_coupling(K)` to derive from a matrix.
    n_layers : int
        Number of topology-entangled variational layers L (Eq. 7).
        Take L >= graph diameter so every node's energy depends on the whole
        connected network.
    phasor : bool
        State encoding. True → phasor (φ,ω): R_x(φ_i) R_y(ω_i).
        False → harmonic (q,p): R_x(q_i) R_y(p_i).
        Momentum uses R_y (NOT R_z): R_z commutes with the diagonal Z-basis
        energy observable, which would force ∂H/∂ω_i = 0. R_y is non-diagonal,
        so the energy depends on momentum and the symplectic gradient is
        non-trivial.
    readout_weights : dict | None
        Optional {'a': (N,) node weights, 'w': (n_edges,) edge weights} for the
        energy observable Eq. (8). Defaults to a_i = 1, w_ij = 1.
    seed : int
        RNG seed for variational-weight initialisation.
    """

    def __init__(
        self,
        num_nodes: int,
        edges: list[tuple[int, int]],
        n_layers: int = 2,
        phasor: bool = True,
        readout_weights: dict | None = None,
        seed: int = 42,
    ):
        self.N = num_nodes
        self.edges = list(edges)
        self.n_layers = n_layers
        self.phasor = phasor
        self.seed = seed

        # Trainable weights per layer: 2 single-qubit angles per node (Ry, Rz)
        # + 1 ZZ angle per edge.  Plus ONE global classical energy scale s
        # (last entry of the weight vector) that multiplies the bounded
        # observable so the model can match the magnitude of the target field.
        self.n_node_weights = 2 * self.N * self.n_layers
        self.n_edge_weights = len(self.edges) * self.n_layers
        self.n_circuit_params = self.n_node_weights + self.n_edge_weights
        self.n_weights = self.n_circuit_params + 1   # +1 for energy scale s

        # ── Data-encoding parameters (one pair per node) ──────────────────
        # φ_i / q_i  → R_x ;   ω_i / p_i → R_y
        self._x1 = [Parameter(f"x1_{i}") for i in range(self.N)]  # φ or q
        self._x2 = [Parameter(f"x2_{i}") for i in range(self.N)]  # ω or p

        # ── Variational parameters (circuit only; scale is classical) ─────
        self._theta = ParameterVector("θ", length=self.n_circuit_params)

        # ── Energy observable  Σ a_i Z_i + Σ w_ij Z_iZ_j ──────────────────
        if readout_weights is None:
            a = np.ones(self.N)
            w = np.ones(len(self.edges))
        else:
            a = np.asarray(readout_weights.get("a", np.ones(self.N)), float)
            w = np.asarray(readout_weights.get("w", np.ones(len(self.edges))), float)
        self._a = a
        self._w = w
        self._obs = self._build_observable(a, w)

        self._estimator = StatevectorEstimator()
        self.circuit = self._build_ansatz()

        # Parameter order for positional binding. This MUST match the order the
        # StatevectorEstimator binds a flat value list to — namely
        # ``self.circuit.parameters``. Qiskit sorts ParameterVector elements
        # NUMERICALLY (θ[2] before θ[10]); a plain ``sorted(key=p.name)`` sorts
        # them lexicographically (θ[10] before θ[2]) and silently mis-binds the
        # θ weights. Use the circuit's own parameter view as the authority.
        self._all_params = list(self._x1) + list(self._x2) + list(self._theta)
        self._param_order = list(self.circuit.parameters)

        # Training state
        self.theta_opt: np.ndarray | None = None
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Observable construction (graph-structured energy readout, Eq. 8)
    # ------------------------------------------------------------------

    def _build_observable(self, a: np.ndarray, w: np.ndarray) -> SparsePauliOp:
        """
        Build  H_obs = Σ_i a_i Z_i + Σ_{(i,j)∈E} w_ij Z_iZ_j  as a SparsePauliOp.

        Qiskit uses little-endian Pauli strings: character position 0 (leftmost)
        is qubit N-1, rightmost is qubit 0. We build with `from_sparse_list`,
        which takes (pauli, qubit_indices, coeff) and handles the layout.
        """
        terms: list[tuple[str, list[int], float]] = []
        for i in range(self.N):
            if abs(a[i]) > 0:
                terms.append(("Z", [i], float(a[i])))
        for (idx, (i, j)) in enumerate(self.edges):
            if abs(w[idx]) > 0:
                terms.append(("ZZ", [i, j], float(w[idx])))
        return SparsePauliOp.from_sparse_list(terms, num_qubits=self.N)

    # ------------------------------------------------------------------
    # Circuit construction (topology-entangled QGNN, Eq. 6–7)
    # ------------------------------------------------------------------

    def _build_ansatz(self) -> QuantumCircuit:
        """
        Build the N-qubit topology-entangled energy ansatz.

        Node encoding (Eq. 6):   R_x(x1_i) R_y(x2_i) on qubit i.
        Per layer ℓ (Eq. 7):
            node update:   R_y(α_ℓ,i) R_z(δ_ℓ,i) on every qubit i
            edge message:  R_zz(β_ℓ,ij) on every edge (i,j) ∈ E

        Both data rotations R_x (position) and R_y (momentum) are NON-diagonal
        in the Z basis, so the diagonal energy observable Σ a_i Z_i + Σ w_ij Z_iZ_j
        depends on BOTH coordinates — a prerequisite for non-zero symplectic
        gradients ∂H/∂φ_i and ∂H/∂ω_i. (Encoding momentum via R_z would make the
        energy independent of ω, since R_z commutes with Z.)
        """
        qc = QuantumCircuit(self.N)

        # Data encoding
        for i in range(self.N):
            qc.rx(self._x1[i], i)
            qc.ry(self._x2[i], i)

        t = 0  # running index into self._theta
        for _ in range(self.n_layers):
            # Node-update sublayer: R_y, R_z per qubit
            for i in range(self.N):
                qc.ry(self._theta[t], i); t += 1
                qc.rz(self._theta[t], i); t += 1
            # Edge-message sublayer: parameterised ZZ on coupling edges only
            for (i, j) in self.edges:
                qc.rzz(self._theta[t], i, j); t += 1

        return qc

    # ------------------------------------------------------------------
    # Parameter binding
    # ------------------------------------------------------------------

    def _bind_values(self, state: np.ndarray, theta: np.ndarray) -> list[float]:
        """
        Build the positional value list matching circuit.parameters sorted order.

        state : (2N,) = [x1_0..x1_{N-1}, x2_0..x2_{N-1}]  (φ|q then ω|p)
        theta : (n_weights,) — only the first n_circuit_params bind to gates;
                the final entry (energy scale s) is applied classically.
        """
        pmap: dict[Parameter, float] = {}
        for i in range(self.N):
            pmap[self._x1[i]] = float(state[i])
            pmap[self._x2[i]] = float(state[self.N + i])
        for k, th in enumerate(self._theta):
            pmap[th] = float(theta[k])
        return [pmap[p] for p in self._param_order]

    @staticmethod
    def _scale(theta: np.ndarray) -> float:
        """Classical energy scale s = last weight (unbounds the ⟨Z⟩∈[-1,1])."""
        return float(theta[-1])

    # ------------------------------------------------------------------
    # Energy evaluation
    # ------------------------------------------------------------------

    def _raw_energy_batch(
        self, states: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Unscaled ⟨obs⟩ over a batch of states via a single estimator.run()."""
        pubs = [(self.circuit, self._obs, self._bind_values(s, theta))
                for s in states]
        res = self._estimator.run(pubs).result()
        return np.array([float(res[b].data.evs) for b in range(len(states))])

    def energy(self, state: np.ndarray, theta: np.ndarray) -> float:
        """
        Scalar network energy H_θ(x) = s · ⟨Σ a_i Z_i + Σ w_ij Z_iZ_j⟩,
        with s the trainable classical energy scale theta[-1].
        """
        raw = self._raw_energy_batch(state[None, :], theta)[0]
        return self._scale(theta) * raw

    def energy_batch(self, states: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Scaled energy over a batch of states. states:(B,2N) → (B,)."""
        return self._scale(theta) * self._raw_energy_batch(states, theta)

    # ------------------------------------------------------------------
    # Per-node symplectic gradients via parameter-shift (Eq. 9)
    # ------------------------------------------------------------------

    def conservative_field(
        self, state: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """
        Pure J-channel network vector field (Eq. 10):
            φ̇_i =  ∂H/∂ω_i ,   ω̇_i = − ∂H/∂φ_i .

        All 4N shifted circuits are evaluated in ONE batched estimator.run(),
        making the full field cost a single job instead of 4N jobs.

        Returns (2N,) = [φ̇_0..φ̇_{N-1}, ω̇_0..ω̇_{N-1}].
        """
        s = np.pi / 2.0
        shifted = []
        # order: for each node i → (x2+ , x2-) then (x1+ , x1-)
        for i in range(self.N):
            sp = state.copy(); sp[self.N + i] += s; shifted.append(sp)
            sm = state.copy(); sm[self.N + i] -= s; shifted.append(sm)
        for i in range(self.N):
            sp = state.copy(); sp[i] += s; shifted.append(sp)
            sm = state.copy(); sm[i] -= s; shifted.append(sm)
        raw = self._raw_energy_batch(np.array(shifted), theta)  # (4N,)
        scale = self._scale(theta)
        raw = raw * scale

        dphi = np.zeros(self.N)     # ∂H/∂ω_i
        domega = np.zeros(self.N)   # −∂H/∂φ_i
        for i in range(self.N):
            dphi[i] = 0.5 * (raw[2 * i] - raw[2 * i + 1])
        off = 2 * self.N
        for i in range(self.N):
            dHdphi = 0.5 * (raw[off + 2 * i] - raw[off + 2 * i + 1])
            domega[i] = -dHdphi
        return np.concatenate([dphi, domega])

    def grad_x2(self, state: np.ndarray, theta: np.ndarray, i: int) -> float:
        """∂H/∂ω_i via parameter-shift (single component; batched internally)."""
        s = np.pi / 2.0
        sp = state.copy(); sp[self.N + i] += s
        sm = state.copy(); sm[self.N + i] -= s
        raw = self._raw_energy_batch(np.array([sp, sm]), theta)
        return 0.5 * self._scale(theta) * (raw[0] - raw[1])

    def grad_x1(self, state: np.ndarray, theta: np.ndarray, i: int) -> float:
        """∂H/∂φ_i via parameter-shift (single component; batched internally)."""
        s = np.pi / 2.0
        sp = state.copy(); sp[i] += s
        sm = state.copy(); sm[i] -= s
        raw = self._raw_energy_batch(np.array([sp, sm]), theta)
        return 0.5 * self._scale(theta) * (raw[0] - raw[1])

    # ------------------------------------------------------------------
    # Initialisation & introspection
    # ------------------------------------------------------------------

    def init_weights(self) -> np.ndarray:
        """Small random circuit weights ∈ [-0.1, 0.1]; energy scale s init = 1."""
        rng = np.random.default_rng(self.seed)
        w = rng.uniform(-0.1, 0.1, self.n_weights)
        w[-1] = 1.0   # energy scale
        return w

    def circuit_diagram(self) -> str:
        return str(self.circuit.draw("text"))

    def __repr__(self) -> str:
        return (
            f"QGNNEnergy(N={self.N}, edges={len(self.edges)}, "
            f"L={self.n_layers}, n_weights={self.n_weights}, "
            f"phasor={self.phasor})"
        )
