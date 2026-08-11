"""
dissipative/quantum_phnn.py  (v3 — symplectic rollout + passivity loss + corrected n_shots)
=============================================================
Quantum Port-Hamiltonian Neural Network (Q-pHNN) for open dissipative systems.

Two implementations:

v1 — DynamicQpHNN  (Qiskit Statevector MINL, COBYLA)
-----------------------------------------------------
Uses Qiskit 2.x Statevector engine — same as the production qrn_qiskit sample.
Measurement-Induced NonLinearity (MINL) is implemented via:
    bitstr, sv = sv.measure([site])     ← correct Qiskit 2.x Born-rule collapse

This is the canonical approach validated in qrn_qiskit/core/discrete_engine.py.
The AerSimulator + if_test approach is NOT used because:
  - Statevector.measure() gives exact Born-rule collapse without shot noise
  - It is numerically equivalent to the physical process
  - It runs on a laptop with no AerSimulator dependency

Per time step (discrete Trotter block):
  1. Evolve sys qubit: sv = sv.evolve(Rz(θ_J))         ← conservative J gate
  2. Entangle with anc: sv = sv.evolve(CRY(θ_R))        ← dissipative R coupling
  3. MINL: bitstr, sv = sv.measure([anc_idx])            ← Born-rule collapse
  4. if bitstr == "1": sv = sv.evolve(Rx(θ_kick))       ← classical feedforward

v2 — VectorFieldQpHNN  (StatevectorEstimator, BFGS + learned γ)
----------------------------------------------------------------
Uses the same 2-qubit energy ansatz as QuantumHNN.
Learns circuit weights θ + scalar damping coefficient γ jointly:
    q̇ =  ∂H/∂p       (parameter-shift on p_in data gate)
    ṗ = −∂H/∂q − γ·p  (parameter-shift on q_in gate + classical damping)

References
----------
qrn_qiskit/core/discrete_engine.py §§ run_trajectory_discrete, _qk_imports
Q-PortHamiltonian.ipynb §3–§5
"""

from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator


# ============================================================================
# v1 — DynamicQpHNN: Statevector MINL (correct Qiskit 2.x API)
# ============================================================================

class DynamicQpHNN:
    """
    Q-pHNN v1: Discrete-time port-Hamiltonian with Statevector MINL.

    Implements dissipation via Born-rule mid-circuit measurement using
    the correct Qiskit 2.x API:
        bitstr, sv = sv.measure([site])

    Architecture per time step
    --------------------------
    1. **Conservative (J)**:  sv = sv.evolve(Rz(θ_J)) on sys qubit
    2. **Dissipative (R)**:   sv = sv.evolve(CRY(θ_R)) — entangles sys+anc
    3. **MINL**:              bitstr, sv = sv.measure([1])  ← ancilla
    4. **Feedforward**:       if "1" in bitstr: sv = sv.evolve(Rx(θ_kick))

    Qubit layout: qubit 0 = sys, qubit 1 = anc (Qiskit LSB convention)

    Parameters
    ----------
    n_steps : int
        Number of discrete time steps per trajectory.
    seed : int
        RNG seed for measurement randomness reproducibility.
    """

    # Qiskit LSB: qubit 0 is the rightmost character in bitstrings
    SYS_IDX = 0   # system qubit
    ANC_IDX = 1   # ancilla (bath) qubit

    def __init__(self, n_steps: int = 6, seed: int = 42):
        self.n_steps = n_steps
        self.seed = seed

        # Pre-build static circuits for each gate (not parameterized — we
        # bind numerically and convert to Operator for each call)
        self._n_qubits = 2

    # ------------------------------------------------------------------
    # Gate builders (return Operator objects for sv.evolve())
    # ------------------------------------------------------------------

    @staticmethod
    def _rz_operator(theta: float, n_qubits: int = 2, target: int = 0) -> Operator:
        """Rz(θ) on `target` qubit embedded in n_qubits-qubit space."""
        qc = QuantumCircuit(n_qubits)
        qc.rz(theta, target)
        return Operator(qc)

    @staticmethod
    def _cry_operator(theta: float, control: int, target: int,
                      n_qubits: int = 2) -> Operator:
        """CRY(θ) gate: control=sys, target=anc."""
        qc = QuantumCircuit(n_qubits)
        qc.cry(theta, control, target)
        return Operator(qc)

    @staticmethod
    def _rx_operator(theta: float, n_qubits: int = 2, target: int = 0) -> Operator:
        """Rx(θ) on `target` qubit embedded in n_qubits-qubit space."""
        qc = QuantumCircuit(n_qubits)
        qc.rx(theta, target)
        return Operator(qc)

    # ------------------------------------------------------------------
    # Initial state preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _init_statevector(init_angle: float) -> Statevector:
        """
        Prepare initial state: Rx(init_angle)|0⟩ on sys, |0⟩ on anc.

        For init_angle = π/2: <σ_x> = sin(π/2) = 1 → q ≈ max
        For init_angle = 0:   state = |0⟩ → <σ_z> = 1, <σ_x> = 0
        """
        qc = QuantumCircuit(2)
        # Apply Hadamard so sys starts in |+⟩, mapping to <σ_x> = 1
        qc.rx(init_angle, DynamicQpHNN.SYS_IDX)
        return Statevector(qc)

    # ------------------------------------------------------------------
    # Expectation value extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_sigma_x(sv: Statevector) -> float:
        """
        <σ_x> on the sys qubit (qubit 0).
        Pauli string for qubit 0 of 2: 'IX' (Qiskit LSB: qubit 0 = rightmost)
        """
        obs = SparsePauliOp("IX")   # X on qubit 0, I on qubit 1
        return float(sv.expectation_value(obs).real)

    @staticmethod
    def _get_sigma_y(sv: Statevector) -> float:
        """<σ_y> on the sys qubit (qubit 0). Pauli string: 'IY'"""
        obs = SparsePauliOp("IY")
        return float(sv.expectation_value(obs).real)

    # ------------------------------------------------------------------
    # Single trajectory forward pass
    # ------------------------------------------------------------------

    def run_trajectory(
        self,
        params: np.ndarray,
        init_angle: float,
        rng: np.random.Generator | None = None,
    ) -> tuple[list[float], list[float]]:
        """
        Run one trajectory and return (<σ_x>, <σ_y>) at each time step.

        Parameters
        ----------
        params : np.ndarray
            [θ_J, θ_R, θ_kick]
        init_angle : float
            Initial state preparation angle for sys qubit.
        rng : numpy Generator, optional
            RNG for measurement randomness. If None, uses self.seed.

        Returns
        -------
        sigma_x_traj, sigma_y_traj : list[float]
            Expectation values at each of n_steps steps.
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)

        theta_J, theta_R, theta_kick = params

        # Pre-build gate operators (reused across steps)
        U_J    = self._rz_operator(theta_J)
        U_R    = self._cry_operator(theta_R, self.SYS_IDX, self.ANC_IDX)
        U_kick = self._rx_operator(theta_kick)
        U_reset_anc = self._rz_operator(0.0)   # identity (reset handled by measure)

        sv = self._init_statevector(init_angle)

        sigma_x_traj: list[float] = []
        sigma_y_traj: list[float] = []

        for _ in range(self.n_steps):
            # --- CONSERVATIVE DYNAMICS (J matrix): Rz on sys ---
            sv = sv.evolve(U_J)

            # --- DISSIPATIVE COUPLING (R matrix): CRY(sys→anc) ---
            sv = sv.evolve(U_R)

            # --- MINL: Born-rule projective measurement on ancilla ---
            # Qiskit 2.x API: (bitstring, new_statevector) = sv.measure([qubit_list])
            bitstr, sv = sv.measure([self.ANC_IDX])

            # --- FEEDFORWARD: conditional Rx kick if ancilla collapsed to |1⟩ ---
            # Qiskit LSB: measure([1]) → bitstr is "0" or "1" for single qubit
            if "1" in str(bitstr):
                sv = sv.evolve(U_kick)

            # Record observables: q ↔ <σ_x>, p ↔ <σ_y>
            sigma_x_traj.append(self._get_sigma_x(sv))
            sigma_y_traj.append(self._get_sigma_y(sv))

        return sigma_x_traj, sigma_y_traj

    # ------------------------------------------------------------------
    # Loss function
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        params: np.ndarray,
        target_q: list[float],
        init_angle: float,
        n_shots: int = 30,
        rng: np.random.Generator | None = None,
    ) -> float:
        """
        MSE between mean predicted <σ_x> and classical q(t).

        Averages `n_shots` independent trajectories to reduce MINL variance.

        Parameters
        ----------
        params : np.ndarray
            [θ_J, θ_R, θ_kick]
        target_q : list[float]
            True q values at steps [1, ..., n_steps].
        init_angle : float
            Initial state angle.
        n_shots : int
            Number of MINL trajectories to average (reduces variance).
        rng : np.random.Generator, optional
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)

        n = min(len(target_q), self.n_steps)
        accumulated_x = np.zeros(n)

        for shot in range(n_shots):
            shot_rng = np.random.default_rng(rng.integers(0, 2**31) + shot)
            sx_traj, _ = self.run_trajectory(params, init_angle, rng=shot_rng)
            accumulated_x += np.array(sx_traj[:n])

        pred_q = accumulated_x / n_shots
        loss = float(np.mean((pred_q - np.array(target_q[:n])) ** 2))
        return loss

    # ------------------------------------------------------------------
    # Prediction (evaluation mode)
    # ------------------------------------------------------------------

    def predict_trajectory(
        self,
        params: np.ndarray,
        init_angle: float,
        n_shots: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate predicted (q(t), p(t)) by averaging over MINL trajectories.

        Returns
        -------
        pred_q, pred_p : np.ndarray of shape (n_steps,)
        """
        rng = np.random.default_rng(self.seed + 999)
        accum_x = np.zeros(self.n_steps)
        accum_y = np.zeros(self.n_steps)

        for shot in range(n_shots):
            shot_rng = np.random.default_rng(rng.integers(0, 2**31) + shot)
            sx, sy = self.run_trajectory(params, init_angle, rng=shot_rng)
            accum_x += np.array(sx)
            accum_y += np.array(sy)

        return accum_x / n_shots, accum_y / n_shots

    # Training state
    params_opt: np.ndarray | None = None
    loss_history: list[float] = []

    def __repr__(self) -> str:
        trained = self.params_opt is not None
        return f"DynamicQpHNN(n_steps={self.n_steps}, trained={trained})"


# ============================================================================
# v2 — VectorFieldQpHNN: principled vector-field learning
# ============================================================================

class VectorFieldQpHNN:
    """
    Q-pHNN v2: Vector-field learning with parameter-shift gradients.

    Uses the same 2-qubit energy ansatz as QuantumHNN (non-dissipative).
    Trains circuit weights θ + scalar damping γ jointly:

        q̇ =  ∂H/∂p       (quantum parameter-shift on p_in data gate)
        ṗ = −∂H/∂q − γ·p  (parameter-shift on q_in + classical γ)

    Qiskit 2.x API
    --------------
    StatevectorEstimator is used with parameter binding via:
        pub = (circuit, observable, parameter_values_list)
    where parameter_values_list order matches circuit.parameters sorted order.

    Parameters
    ----------
    n_layers : int
        CZ entanglement layers (4 weights each).
    seed : int
        RNG seed for weight initialisation.
    """

    def __init__(self, n_layers: int = 1, seed: int = 42):
        self.n_layers = n_layers
        self.n_circuit_weights = 4 * n_layers
        # Full param vector: [θ₀…θ_{n-1}, s, γ]
        #   s = learnable energy scale (fixes gradient magnitude mismatch)
        #   γ = damping coefficient
        self.n_params = self.n_circuit_weights + 2  # +s +γ
        self.seed = seed

        # Build circuit parameters
        self._q_in = Parameter("q_in")
        self._p_in = Parameter("p_in")
        self._theta = ParameterVector("θ", length=self.n_circuit_weights)
        self._obs = SparsePauliOp("ZZ")
        self._estimator = StatevectorEstimator()

        self.circuit = self._build_ansatz()

        # Sorted parameter order (Qiskit sorts alphabetically by name):
        # q_in, p_in, θ[0]..θ[n-1]  →  ordered list for positional binding
        self._param_order = sorted(
            [self._q_in, self._p_in] + list(self._theta),
            key=lambda p: p.name,
        )

        # Training state
        self.params_opt: np.ndarray | None = None
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Ansatz
    # ------------------------------------------------------------------

    def _build_ansatz(self) -> QuantumCircuit:
        """
        2-qubit energy ansatz: data encoding + entanglement layers.

        Qubit 0 → position q (encoded via Rx(q_in))
        Qubit 1 → momentum p (encoded via Ry(p_in))
        Observable: ZZ = scalar energy proxy H(q, p)
        """
        qc = QuantumCircuit(2)
        qc.rx(self._q_in, 0)
        qc.ry(self._p_in, 1)

        for layer in range(self.n_layers):
            base = layer * 4
            qc.cz(0, 1)
            qc.ry(self._theta[base + 0], 0)
            qc.ry(self._theta[base + 1], 1)
            qc.cz(0, 1)
            qc.rx(self._theta[base + 2], 0)
            qc.rx(self._theta[base + 3], 1)

        return qc

    # ------------------------------------------------------------------
    # Parameter binding (Qiskit 2.x)
    # ------------------------------------------------------------------

    def _bind_values(
        self, q: float, p: float, theta: np.ndarray
    ) -> list[float]:
        """
        Build the parameter value list in sorted parameter name order.

        Qiskit sorts parameters alphabetically. For our circuit:
            q_in, p_in, θ[0], θ[1], ..., θ[n-1]
        Sorted: p_in < q_in < θ[0] < θ[1] ...
        (alphabetically: 'p' < 'q' < 'θ')

        We build a mapping dict and sort to the correct order.
        """
        param_map = {self._q_in: q, self._p_in: p}
        for i, th in enumerate(self._theta):
            param_map[th] = float(theta[i])

        return [param_map[p] for p in self._param_order]

    # ------------------------------------------------------------------
    # Energy evaluation
    # ------------------------------------------------------------------

    def _raw_zz(self, q: float, p: float, theta: np.ndarray) -> float:
        """Evaluate raw ⟨ZZ⟩ ∈ [−1,1] from the circuit."""
        vals = self._bind_values(q, p, theta)
        pub = (self.circuit, self._obs, vals)
        job = self._estimator.run([pub])
        return float(job.result()[0].data.evs)

    def energy(self, q: float, p: float, theta: np.ndarray, s: float = 1.0) -> float:
        """
        H_eff(q,p;θ,s) = s·⟨ZZ⟩(q,p;θ).
        s is the learnable energy scale extracted from the full params vector.
        """
        return s * self._raw_zz(q, p, theta)

    # ------------------------------------------------------------------
    # Port-Hamiltonian vector field (parameter-shift on data gates)
    # ------------------------------------------------------------------

    def q_dot(self, q: float, p: float, theta: np.ndarray, s: float = 1.0) -> float:
        """q̇ = s · ∂⟨ZZ⟩/∂p via parameter-shift on p_in data gate."""
        shift = np.pi / 2.0
        raw = 0.5 * (self._raw_zz(q, p + shift, theta) - self._raw_zz(q, p - shift, theta))
        return s * raw

    def p_dot_conservative(self, q: float, p: float, theta: np.ndarray, s: float = 1.0) -> float:
        """Conservative part: −s · ∂⟨ZZ⟩/∂q via parameter-shift on q_in gate."""
        shift = np.pi / 2.0
        raw = 0.5 * (self._raw_zz(q + shift, p, theta) - self._raw_zz(q - shift, p, theta))
        return -s * raw

    def p_dot(self, q: float, p: float, theta: np.ndarray, gamma: float, s: float = 1.0) -> float:
        """Full dissipative ṗ = −s·∂⟨ZZ⟩/∂q − γ·p."""
        return self.p_dot_conservative(q, p, theta, s) - gamma * p

    def vector_field(
        self, q: float, p: float, params: np.ndarray
    ) -> tuple[float, float]:
        """Return (q̇, ṗ) with params = [θ₀…θ_{n-1}, s, γ]."""
        theta = params[: self.n_circuit_weights]
        s     = float(params[-2])   # energy scale
        gamma = float(params[-1])   # damping
        return self.q_dot(q, p, theta, s), self.p_dot(q, p, theta, gamma, s)

    # ------------------------------------------------------------------
    # Loss function
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        params: np.ndarray,
        q_data: np.ndarray,
        p_data: np.ndarray,
        q_dot_true: np.ndarray,
        p_dot_true: np.ndarray,
    ) -> float:
        """MSE between predicted and true port-Hamiltonian vector field.
        params = [θ₀…θ_{n-1}, s, γ]
        """
        theta = params[: self.n_circuit_weights]
        s     = float(params[-2])
        gamma = float(params[-1])
        n = len(q_data)
        loss = 0.0
        for i in range(n):
            q_d = self.q_dot(q_data[i], p_data[i], theta, s)
            p_d = self.p_dot(q_data[i], p_data[i], theta, gamma, s)
            loss += (q_d - q_dot_true[i])**2 + (p_d - p_dot_true[i])**2
        return loss / n

    def compute_loss_with_passivity(
        self,
        params: np.ndarray,
        q_data: np.ndarray,
        p_data: np.ndarray,
        q_dot_true: np.ndarray,
        p_dot_true: np.ndarray,
        lambda_passivity: float = 0.1,
    ) -> float:
        """
        Vector-field MSE + passivity penalty.

        The passivity penalty penalises positive energy rates (Ḣ > 0) over
        the training states, ensuring the dissipative inequality Ḣ ≤ 0 is
        softly enforced across the training distribution:

            L_total = L_vf + λ · mean( relu(Ḣ_i) )

        where Ḣ_i = (∂H/∂q)(q̇) + (∂H/∂p)(ṗ) at each training point.

        Parameters
        ----------
        lambda_passivity : float
            Weight of the passivity penalty term. 0.1 is a good default.
        """
        vf_loss = self.compute_loss(params, q_data, p_data, q_dot_true, p_dot_true)

        theta = params[: self.n_circuit_weights]
        s     = float(params[-2])
        gamma = float(params[-1])

        # Compute Ḣ = (∂H/∂q)·q̇ + (∂H/∂p)·ṗ at each training point
        passivity_violation = 0.0
        n = len(q_data)
        shift = np.pi / 2.0
        for i in range(n):
            q_i, p_i = float(q_data[i]), float(p_data[i])
            dH_dp = s * 0.5 * (self._raw_zz(q_i, p_i + shift, theta) - self._raw_zz(q_i, p_i - shift, theta))
            dH_dq = s * 0.5 * (self._raw_zz(q_i + shift, p_i, theta) - self._raw_zz(q_i - shift, p_i, theta))
            q_d = self.q_dot(q_i, p_i, theta, s)
            p_d = self.p_dot(q_i, p_i, theta, gamma, s)
            h_dot = dH_dq * q_d + dH_dp * p_d
            passivity_violation += max(0.0, h_dot)

        passivity_penalty = lambda_passivity * passivity_violation / n
        return vf_loss + passivity_penalty

    # ------------------------------------------------------------------
    # Trajectory rollout (Euler integration)
    # ------------------------------------------------------------------

    def rollout(
        self,
        q0: float,
        p0: float,
        params: np.ndarray,
        dt: float = 0.05,
        n_steps: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate trajectory using learned vector field."""
        t_arr = np.arange(n_steps) * dt
        q_arr = np.zeros(n_steps)
        p_arr = np.zeros(n_steps)
        q_arr[0], p_arr[0] = q0, p0

        theta = params[: self.n_circuit_weights]
        gamma = params[-1]

        for i in range(n_steps - 1):
            dq = self.q_dot(q_arr[i], p_arr[i], theta)
            dp = self.p_dot(q_arr[i], p_arr[i], theta, gamma)
            q_arr[i+1] = q_arr[i] + dt * dq
            p_arr[i+1] = p_arr[i] + dt * dp

        return t_arr, q_arr, p_arr

    def symplectic_rollout(
        self,
        q0: float,
        p0: float,
        params: np.ndarray,
        dt: float = 0.05,
        n_steps: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Integrate dissipative trajectory using operator splitting + Störmer–Verlet.

        The total vector field is split into:
          1. Conservative part (J-channel): integrated with symplectic half-steps.
          2. Dissipative part (R-channel, −γp): integrated analytically via
             exponential: p → p · exp(−γ·dt/2) per half-step.

        Splitting scheme per step:
            p_{n+1/2} = p_n · exp(−γ·dt/2) + (dt/2) · p_dot_conservative(q_n, p_n)
            q_{n+1}   = q_n +  dt · q_dot(q_n, p_{n+1/2})
            p_{n+1}   = p_{n+1/2} · exp(−γ·dt/2) + (dt/2) · p_dot_conservative(q_{n+1}, p_{n+1/2})

        Returns (t, q, p) arrays.
        """
        t_arr = np.arange(n_steps) * dt
        q_arr = np.zeros(n_steps)
        p_arr = np.zeros(n_steps)
        q_arr[0], p_arr[0] = q0, p0

        theta = params[: self.n_circuit_weights]
        s     = float(params[-2])              # energy scale
        gamma = max(0.0, float(params[-1]))    # enforce γ ≥ 0
        decay_half = np.exp(-gamma * 0.5 * dt)

        for i in range(n_steps - 1):
            q, p = q_arr[i], p_arr[i]
            # Half-step: dissipative decay + conservative kick
            p_half = p * decay_half + 0.5 * dt * self.p_dot_conservative(q, p, theta, s)
            # Full-step position
            q_new = q + dt * self.q_dot(q, p_half, theta, s)
            # Half-step: conservative kick + dissipative decay
            p_new = (p_half + 0.5 * dt * self.p_dot_conservative(q_new, p_half, theta, s)) * decay_half
            q_arr[i + 1], p_arr[i + 1] = q_new, p_new

        return t_arr, q_arr, p_arr

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_params(self) -> np.ndarray:
        """Initialise full param vector [θ₀…θ_{n-1}, s, γ].

        θ: uniform in [-π/4, π/4] for non-trivial but bounded gradients.
        s: 2.0 — gives PSR gradients headroom to match damped oscillator
           which has q,p ∈ [-2,2] → |q̇|,|ṗ| ≤ 2.
        γ: 0.1 — small positive starting value.
        """
        rng = np.random.default_rng(self.seed)
        theta_init = rng.uniform(-np.pi / 4, np.pi / 4, self.n_circuit_weights)
        return np.concatenate([theta_init, [2.0, 0.1]])   # [θ, s, γ]

    def apply_sign_correction(
        self,
        params: np.ndarray,
        q_data: np.ndarray,
        p_data: np.ndarray,
        q_dot_true: np.ndarray,
        p_dot_true: np.ndarray,
    ) -> np.ndarray:
        """Negate energy scale s if vector field is anti-correlated with truth."""
        theta = params[: self.n_circuit_weights]
        s     = float(params[-2])
        gamma = float(params[-1])
        q_dots = [self.q_dot(q_data[i], p_data[i], theta, s) for i in range(len(q_data))]
        p_dots = [self.p_dot(q_data[i], p_data[i], theta, gamma, s) for i in range(len(p_data))]
        corr_q = float(np.corrcoef(q_dots, q_dot_true)[0, 1])
        corr_p = float(np.corrcoef(p_dots, p_dot_true)[0, 1])
        if 0.5 * (corr_q + corr_p) < 0:
            fixed = params.copy()
            fixed[-2] *= -1.0   # flip scale → flips all PSR gradients
            return fixed
        return params

    def __repr__(self) -> str:
        trained = self.params_opt is not None
        return (
            f"VectorFieldQpHNN(n_layers={self.n_layers}, "
            f"n_params={self.n_params}, trained={trained})"
        )

    def circuit_diagram(self) -> str:
        return str(self.circuit.draw("text"))
