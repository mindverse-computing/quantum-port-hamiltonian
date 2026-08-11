"""
network/quantum_network_phnn.py
================================
Network Quantum Port-Hamiltonian Neural Network (NetworkQpHNN).

Wraps the topology-entangled QGNNEnergy surrogate and assembles the network
port-Hamiltonian vector field  ẋ = (J − R)∇H  for an N-node system.

Two dissipation channels (theory §5):

  • Analytic-γ  (mode='gamma'):  ω̇_i = −∂H/∂φ_i − γ_i ω_i,  learnable γ ⪰ 0
       — exact parameter-shift gradients, BFGS-trainable, γ_i physically
         interpretable and validatable against ground truth.

  • Multi-ancilla MINL (mode='minl'):  one bath ancilla per node; per Trotter
       step: conservative block → CR_y entangle → Born-rule measure → conditional
       R_x kick. A genuine discrete-time network Lindblad (CPTP) channel.

The conservative regime (dissipative=False) is the pure J-channel: energy is
conserved on the N-torus.

See theory/network_qphnn_theory.md §1–§5.
"""

from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp

from .qgnn_energy import QGNNEnergy, edges_from_coupling


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus, enforces γ ⪰ 0."""
    return np.logaddexp(0.0, x)


class NetworkQpHNN:
    """
    Network Quantum Port-Hamiltonian Neural Network.

    Parameters
    ----------
    num_nodes : int
        Number of nodes N.
    edges : list[tuple[int,int]]
        Coupling edges (i<j). Use edges_from_coupling(K).
    n_layers : int
        QGNN entanglement layers.
    dissipative : bool
        If True, include an R-channel (per-node damping). If False, pure
        conservative J-channel (energy conserved).
    diss_mode : {'gamma','minl'}
        Dissipation implementation when dissipative=True.
    phasor : bool
        Phasor (φ,ω) vs harmonic (q,p) encoding.
    readout_weights : dict | None
        Passed to QGNNEnergy (graph-structured energy observable).
    seed : int
        RNG seed.

    Parameter vector
    ----------------
    The trainable vector is  params = [θ (QGNN weights) | γ_raw (N, if
    dissipative & mode='gamma')].  γ_i = softplus(γ_raw_i) ⪰ 0.
    """

    def __init__(
        self,
        num_nodes: int,
        edges: list[tuple[int, int]],
        n_layers: int = 2,
        dissipative: bool = False,
        diss_mode: str = "gamma",
        phasor: bool = True,
        readout_weights: dict | None = None,
        seed: int = 42,
    ):
        self.N = num_nodes
        self.edges = list(edges)
        self.dissipative = dissipative
        self.diss_mode = diss_mode
        self.phasor = phasor
        self.seed = seed

        self.qgnn = QGNNEnergy(
            num_nodes=num_nodes, edges=edges, n_layers=n_layers,
            phasor=phasor, readout_weights=readout_weights, seed=seed,
        )
        self.n_circuit_weights = self.qgnn.n_weights

        # Parameter layout
        if self.dissipative and self.diss_mode == "gamma":
            self.n_params = self.n_circuit_weights + self.N
        else:
            self.n_params = self.n_circuit_weights

        # MINL config (mode='minl')
        self.minl_steps = 6
        self.minl_shots = 20

        self.params_opt: np.ndarray | None = None
        self.loss_history: list[float] = []

    # ==================================================================
    #  Parameter split helpers
    # ==================================================================

    def split_params(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (theta, gamma) where gamma is softplus'd (or zeros)."""
        theta = params[: self.n_circuit_weights]
        if self.dissipative and self.diss_mode == "gamma":
            gamma = _softplus(params[self.n_circuit_weights:])
        else:
            gamma = np.zeros(self.N)
        return theta, gamma

    # ==================================================================
    #  Vector field (analytic channels: conservative + gamma dissipation)
    # ==================================================================

    def vector_field(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Network port-Hamiltonian vector field ẋ = (J−R)∇H.

        Conservative part (J-channel, Eq. 10):
            φ̇_i =  ∂H/∂ω_i ,   ω̇_i = − ∂H/∂φ_i
        Dissipative part (gamma mode, port-Hamiltonian R∇H form, Eq. 11):
            ω̇_i −= γ_i (∂H/∂ω_i)

        The R-channel damps the *gradient* ∂H/∂ω_i, not the raw state ω_i.
        This is the exact analogue of the classical phnn.py term
        ẋ = (J−R)∇H with R = diag(γ) on the momentum block, and it makes
        network passivity Ḣ = −Σ γ_i(∂H/∂ω_i)² ≤ 0 hold by construction.
        For the true kinetic energy ∂H/∂ω_i = ω_i, so this coincides with the
        state-damping −γ_i ω_i used by the Kuramoto data generator.

        state  : (2N,)   Returns (2N,) = [φ̇ ; ω̇].
        """
        theta, gamma = self.split_params(params)
        # Batched J-channel field: φ̇_i = ∂H/∂ω_i, ω̇_i^cons = −∂H/∂φ_i
        field = self.qgnn.conservative_field(state, theta)  # (2N,), one job
        phi_dot = field[: self.N]          # = ∂H/∂ω_i
        omega_dot = field[self.N:].copy()  # = −∂H/∂φ_i
        if self.dissipative and self.diss_mode == "gamma":
            # R-channel damps the gradient ∂H/∂ω_i (= φ̇_i here)
            omega_dot = omega_dot - gamma * phi_dot
        return np.concatenate([phi_dot, omega_dot])

    def energy(self, state: np.ndarray, params: np.ndarray) -> float:
        theta, _ = self.split_params(params)
        return self.qgnn.energy(state, theta)

    # ==================================================================
    #  Network passivity check:  Ḣ = Σ_i (∂H/∂ω_i) ω̇_i
    # ==================================================================

    def energy_rate(self, state: np.ndarray, params: np.ndarray) -> float:
        """
        Instantaneous energy rate Ḣ = (∇H)·ẋ along the learned field, computed
        as the full chain-rule sum over BOTH phase-space blocks:

            Ḣ = Σ_i (∂H/∂φ_i) φ̇_i + (∂H/∂ω_i) ω̇_i .

        Conservative (γ=0):  the two J-channel terms cancel exactly ⇒ Ḣ = 0.
        Dissipative (γ⪰0):   Ḣ = −Σ_i γ_i (∂H/∂ω_i)² ≤ 0 (network passivity).
        """
        theta, _ = self.split_params(params)
        # Batched conservative field gives ∂H/∂ω_i (=cons φ̇) and −∂H/∂φ_i.
        cons = self.qgnn.conservative_field(state, theta)
        dHdomega = cons[: self.N]        # ∂H/∂ω_i
        dHdphi = -cons[self.N:]          # ∂H/∂φ_i  (cons[N:] = −∂H/∂φ_i)
        field = self.vector_field(state, params)
        phi_dot, omega_dot = field[: self.N], field[self.N:]
        return float(np.sum(dHdphi * phi_dot + dHdomega * omega_dot))

    # ==================================================================
    #  Loss (vector-field MSE) — for BFGS training of analytic channels
    # ==================================================================

    def compute_loss(
        self,
        params: np.ndarray,
        states: np.ndarray,     # (B, 2N)
        d_states: np.ndarray,   # (B, 2N) true field
    ) -> float:
        n = len(states)
        loss = 0.0
        for b in range(n):
            pred = self.vector_field(states[b], params)
            loss += float(np.mean((pred - d_states[b]) ** 2))
        return loss / n

    # ==================================================================
    #  Trajectory rollout (Euler, learned analytic field)
    # ==================================================================

    def rollout(
        self, state0: np.ndarray, params: np.ndarray,
        dt: float = 0.05, n_steps: int = 100,
    ) -> np.ndarray:
        """Integrate the learned field. Returns (n_steps, 2N) trajectory."""
        traj = np.zeros((n_steps, 2 * self.N))
        traj[0] = state0
        for t in range(n_steps - 1):
            traj[t + 1] = traj[t] + dt * self.vector_field(traj[t], params)
        return traj

    def energy_along(self, traj: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Network energy H(t) along a trajectory (n_steps, 2N) → (n_steps,)."""
        theta, _ = self.split_params(params)
        return np.array([self.qgnn.energy(traj[t], theta)
                         for t in range(len(traj))])

    # ==================================================================
    #  Multi-ancilla MINL channel (mode='minl') — genuine CPTP dissipation
    # ==================================================================

    def _minl_operators(self, theta_J: np.ndarray, theta_R: np.ndarray,
                        theta_k: np.ndarray):
        """
        Pre-build the per-node gate operators for one Trotter block on the
        2N-qubit (N system + N ancilla) space.

        Qubit layout: system qubits 0..N-1, ancilla qubits N..2N-1.
        Ancilla i is paired with system node i.
        """
        nq = 2 * self.N
        # Conservative J-block: R_z(θ_J_i) on each system qubit + ZZ on edges
        qc_J = QuantumCircuit(nq)
        for i in range(self.N):
            qc_J.rz(float(theta_J[i]), i)
        for (idx, (i, j)) in enumerate(self.edges):
            # small Ising coupling shared parameter (reuse first edge angle set)
            qc_J.rzz(float(theta_J[self.N % len(theta_J)]), i, j)
        U_J = Operator(qc_J)

        # System-bath entangle: CR_y(θ_R_i) control=sys i, target=anc i
        qc_R = QuantumCircuit(nq)
        for i in range(self.N):
            qc_R.cry(float(theta_R[i]), i, self.N + i)
        U_R = Operator(qc_R)

        return U_J, U_R

    def run_minl_trajectory(
        self, params_minl: dict, state0: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        One multi-ancilla MINL trajectory.

        params_minl : {'theta_J':(N,), 'theta_R':(N,), 'theta_k':(N,)}
        state0      : (2N,) initial phase-space point (encodes system qubits).

        Returns (minl_steps, N) array of ⟨X_i⟩ (position read-out q̂_i(t)).
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)
        nq = 2 * self.N

        # Prepare initial system state via data encoding (ancillas in |0⟩).
        # Momentum uses R_y (not R_z): R_z commutes with the X/Y read-out and
        # the Z-basis dynamics would leave the momentum coordinate invisible —
        # the same diagonality bug fixed in QGNNEnergy._build_ansatz.
        qc0 = QuantumCircuit(nq)
        for i in range(self.N):
            qc0.rx(float(state0[i]), i)
            qc0.ry(float(state0[self.N + i]), i)
        sv = Statevector(qc0)

        U_J, U_R = self._minl_operators(
            params_minl["theta_J"], params_minl["theta_R"], params_minl["theta_k"]
        )
        # conditional kicks pre-built per node
        kick_ops = []
        for i in range(self.N):
            qk = QuantumCircuit(nq)
            qk.rx(float(params_minl["theta_k"][i]), i)
            kick_ops.append(Operator(qk))

        traj = np.zeros((self.minl_steps, self.N))
        for step in range(self.minl_steps):
            sv = sv.evolve(U_J)              # conservative
            sv = sv.evolve(U_R)              # system-bath entangle
            # per-node Born-rule measurement + feedforward
            for i in range(self.N):
                bit, sv = sv.measure([self.N + i])   # measure ancilla i
                if "1" in str(bit):
                    sv = sv.evolve(kick_ops[i])
            # read ⟨X_i⟩ on each system qubit
            for i in range(self.N):
                obs = SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=nq)
                traj[step, i] = float(sv.expectation_value(obs).real)
        return traj

    def predict_minl(
        self, params_minl: dict, state0: np.ndarray, n_shots: int | None = None,
    ) -> np.ndarray:
        """Ensemble-mean ⟨X_i⟩(t) over n_shots MINL trajectories → (steps, N)."""
        n_shots = n_shots or self.minl_shots
        rng = np.random.default_rng(self.seed + 999)
        acc = np.zeros((self.minl_steps, self.N))
        for s in range(n_shots):
            shot_rng = np.random.default_rng(rng.integers(0, 2**31) + s)
            acc += self.run_minl_trajectory(params_minl, state0, rng=shot_rng)
        return acc / n_shots

    # ==================================================================
    #  Initialisation
    # ==================================================================

    def init_params(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        theta = rng.uniform(-0.1, 0.1, self.n_circuit_weights)
        if self.dissipative and self.diss_mode == "gamma":
            # γ_raw init so softplus(γ_raw) ≈ 0.1
            gamma_raw = np.full(self.N, np.log(np.expm1(0.1)))
            return np.concatenate([theta, gamma_raw])
        return theta

    def __repr__(self) -> str:
        return (
            f"NetworkQpHNN(N={self.N}, edges={len(self.edges)}, "
            f"dissipative={self.dissipative}, mode={self.diss_mode}, "
            f"n_params={self.n_params})"
        )
