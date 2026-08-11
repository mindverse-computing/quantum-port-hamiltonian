"""
non_dissipative/quantum_hnn.py  (v4 — energy scale + offset + sign-correction)
================================================================
Quantum Hamiltonian Neural Network (Q-HNN) for closed conservative systems.

Architecture
------------
A 2-qubit parameterized circuit represents the scalar energy manifold H_θ(q,p).
The symplectic gradient is extracted via Parameter-Shift on the DATA-ENCODING gates:

    q̇ = ∂H_eff/∂p  ≈  s · ½[⟨ZZ⟩(q, p+π/2; θ) − ⟨ZZ⟩(q, p−π/2; θ)]
    ṗ = −∂H_eff/∂q ≈ −s · ½[⟨ZZ⟩(q+π/2, p; θ) − ⟨ZZ⟩(q−π/2, p; θ)]

where H_eff(q,p;θ,s,b) = s·⟨ZZ⟩(q,p;θ) + b.

Key improvement (v4)
--------------------
Added two extra scalar parameters [s, b]:
  • s (energy scale):  allows the model to match the true gradient magnitude.
    ⟨ZZ⟩ ∈ [−1,1] but the pendulum energy gradient can exceed 1; without s the
    predicted frequency is systematically too low.
  • b (energy offset): allows setting the zero-point of H_eff independently of
    the circuit output, which matters for energy conservation diagnostics.
  • Post-training sign correction: if the overall PSR gradient correlation with
    the true vector field is negative, s is automatically negated to fix sign flips.

Qiskit 2.x API Notes
--------------------
• StatevectorEstimator.run() accepts a list of PUBs:
      pub = (circuit, observable, parameter_values)
  where parameter_values is a list ordered by circuit.parameters sorted order.
• circuit.parameters returns a ParameterView sorted alphabetically by name.
• We sort parameters explicitly to ensure correct binding order.

Circuit Layout
--------------
  Qubit 0 → position q:   Rx(q_in) → CZ → Ry(θ₀) → CZ → Rx(θ₂)
  Qubit 1 → momentum p:   Ry(p_in) → CZ → Ry(θ₁) → CZ → Rx(θ₃)
  Observable: ZZ  (scalar energy proxy H(q,p))
  Full parameter vector: [θ₀…θ_{n-1}, s, b]

References
----------
Q-Hamiltonian.ipynb §2 ("The Correct Method")
qrn_qiskit/core/discrete_engine.py (verified Qiskit 2.x API patterns)
"""

from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator


class QuantumHNN:
    """
    Quantum Hamiltonian Neural Network (conservative, non-dissipative).

    The full parameter vector is phi = [theta_0 ... theta_{n-1}, s, b] where
    s is a learnable energy scale and b is a learnable energy offset.
    The effective energy is H_eff(q,p) = s * <ZZ>(q,p;theta) + b.
    All symplectic gradients are computed via PSR on the data-encoding gates and
    multiplied by s, which corrects the frequency mismatch that occurs when
    <ZZ> ∈ [−1,1] but the true energy gradient has a larger magnitude.

    Parameters
    ----------
    n_layers : int
        Number of CZ-entanglement layers (4 trainable weights per layer).
    seed : int
        RNG seed for weight initialisation.
    """

    def __init__(self, n_layers: int = 1, seed: int = 42):
        self.n_layers = n_layers
        self.n_circuit_weights = 4 * n_layers  # θ parameters only
        self.n_weights = self.n_circuit_weights + 2  # + scale s + offset b
        self.seed = seed

        # Build circuit and observables
        self._q_in = Parameter("q_in")
        self._p_in = Parameter("p_in")
        self._theta = ParameterVector("θ", length=self.n_circuit_weights)
        self._obs = SparsePauliOp("ZZ")   # energy proxy H(q,p)
        self._estimator = StatevectorEstimator()

        self.circuit = self._build_ansatz()

        # Build the sorted parameter order (Qiskit sorts alphabetically).
        # For parameters named "q_in", "p_in", "θ[0]"..."θ[n-1]":
        #   Sorted order: p_in, q_in, θ[0], θ[1], ...
        # (alphabetically: 'p' < 'q' < 'θ')
        self._param_order = sorted(
            [self._q_in, self._p_in] + list(self._theta),
            key=lambda p: p.name,
        )

        # Training state
        self.theta_opt: np.ndarray | None = None
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def _build_ansatz(self) -> QuantumCircuit:
        """
        Build the 2-qubit parameterized energy ansatz.

        Qubit 0: position q  →  Rx(q_in)
        Qubit 1: momentum p  →  Ry(p_in)

        Per entanglement layer:
            CZ(0,1) → Ry(θ_2k, 0) → Ry(θ_2k+1, 1) → CZ(0,1)
            → Rx(θ_2k+2, 0) → Rx(θ_2k+3, 1)
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
    # Parameter binding helper
    # ------------------------------------------------------------------

    def _build_vals(
        self, q: float, p: float, theta: np.ndarray
    ) -> list[float]:
        """
        Build parameter value list in sorted-name order for Qiskit 2.x binding.

        circuit.parameters is sorted alphabetically. We map each Parameter
        object to its value and return a list matching that order.
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
        vals = self._build_vals(q, p, theta)
        pub = (self.circuit, self._obs, vals)
        job = self._estimator.run([pub])
        return float(job.result()[0].data.evs)

    def energy(self, q: float, p: float, phi: np.ndarray) -> float:
        """
        Evaluate H_eff(q,p;φ) = s·⟨ZZ⟩(q,p;θ) + b.

        Parameters
        ----------
        phi : np.ndarray
            Full parameter vector [θ₀…θ_{n-1}, s, b].
        """
        theta = phi[: self.n_circuit_weights]
        s, b  = float(phi[-2]), float(phi[-1])
        return s * self._raw_zz(q, p, theta) + b

    # ------------------------------------------------------------------
    # Symplectic gradient via parameter-shift on data-encoding gates
    # ------------------------------------------------------------------

    def q_dot(self, q: float, p: float, phi: np.ndarray) -> float:
        """q̇ = s · ∂⟨ZZ⟩/∂p via parameter-shift on p_in data gate."""
        theta = phi[: self.n_circuit_weights]
        s     = float(phi[-2])
        shift = np.pi / 2.0
        raw_grad = 0.5 * (
            self._raw_zz(q, p + shift, theta)
            - self._raw_zz(q, p - shift, theta)
        )
        return s * raw_grad

    def p_dot(self, q: float, p: float, phi: np.ndarray) -> float:
        """ṗ = −s · ∂⟨ZZ⟩/∂q via parameter-shift on q_in data gate."""
        theta = phi[: self.n_circuit_weights]
        s     = float(phi[-2])
        shift = np.pi / 2.0
        raw_grad = 0.5 * (
            self._raw_zz(q + shift, p, theta)
            - self._raw_zz(q - shift, p, theta)
        )
        return -s * raw_grad

    def vector_field(
        self, q: float, p: float, phi: np.ndarray
    ) -> tuple[float, float]:
        """Return (q̇, ṗ) at (q, p) given full parameter vector phi."""
        return self.q_dot(q, p, phi), self.p_dot(q, p, phi)

    # ------------------------------------------------------------------
    # Loss function (vector-field MSE, batched PSR)
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        phi: np.ndarray,
        q_data: np.ndarray,
        p_data: np.ndarray,
        q_dot_true: np.ndarray,
        p_dot_true: np.ndarray,
    ) -> float:
        """
        MSE loss between predicted and true vector field — batched PSR.

        phi = [θ₀…θ_{n-1}, s, b]  (full parameter vector incl. scale + offset).

        All 4·N shifted-circuit evaluations are submitted in a SINGLE
        estimator.run() call. The scale s multiplies all PSR gradients, so
        BFGS can freely adjust the gradient magnitude to match the true field.
        """
        n = len(q_data)
        shift = np.pi / 2.0
        theta = phi[: self.n_circuit_weights]
        s     = float(phi[-2])   # energy scale (signed — handles sign flips)
        # b (offset) cancels in the PSR difference, so not needed here

        # Build all 4N PUBs
        pubs = []
        for i in range(n):
            q_i, p_i = float(q_data[i]), float(p_data[i])
            pubs.append((self.circuit, self._obs, self._build_vals(q_i, p_i + shift, theta)))
            pubs.append((self.circuit, self._obs, self._build_vals(q_i, p_i - shift, theta)))
            pubs.append((self.circuit, self._obs, self._build_vals(q_i + shift, p_i, theta)))
            pubs.append((self.circuit, self._obs, self._build_vals(q_i - shift, p_i, theta)))

        job = self._estimator.run(pubs)
        evs = [float(job.result()[b].data.evs) for b in range(4 * n)]

        loss = 0.0
        for i in range(n):
            base = 4 * i
            q_dot_pred =  s * 0.5 * (evs[base]     - evs[base + 1])
            p_dot_pred = -s * 0.5 * (evs[base + 2] - evs[base + 3])
            loss += (q_dot_pred - q_dot_true[i]) ** 2 + (p_dot_pred - p_dot_true[i]) ** 2
        return loss / n

    # ------------------------------------------------------------------
    # Post-training sign correction
    # ------------------------------------------------------------------

    def apply_sign_correction(
        self,
        phi: np.ndarray,
        q_data: np.ndarray,
        p_data: np.ndarray,
        q_dot_true: np.ndarray,
        p_dot_true: np.ndarray,
    ) -> np.ndarray:
        """
        Check if the predicted vector field is globally anti-correlated with
        the true field. If so, negate the energy scale s (phi[-2]) to flip
        the sign of all predicted gradients.

        This handles the sign ambiguity where the circuit converges to −H
        instead of +H. Both are valid energy manifolds for the circuit, but
        only one produces the correct vector field direction.

        Returns the (possibly sign-corrected) phi vector.
        """
        q_dots, p_dots = [], []
        for i in range(len(q_data)):
            q_dots.append(self.q_dot(q_data[i], p_data[i], phi))
            p_dots.append(self.p_dot(q_data[i], p_data[i], phi))

        q_dots = np.array(q_dots)
        p_dots = np.array(p_dots)

        corr_q = np.corrcoef(q_dots, q_dot_true)[0, 1]
        corr_p = np.corrcoef(p_dots, p_dot_true)[0, 1]
        avg_corr = 0.5 * (corr_q + corr_p)

        if avg_corr < 0:
            phi_corrected = phi.copy()
            phi_corrected[-2] *= -1.0   # negate scale s → flips all gradients
            return phi_corrected
        return phi

    # ------------------------------------------------------------------
    # Energy conservation check (for evaluation)
    # ------------------------------------------------------------------

    def energy_along_trajectory(
        self, q_arr: np.ndarray, p_arr: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """
        Compute H(q(t), p(t)) along a rolled-out trajectory.

        Energy should be constant for a well-trained conservative Q-HNN.
        """
        return np.array([
            self.energy(q_arr[i], p_arr[i], theta)
            for i in range(len(q_arr))
        ])

    # ------------------------------------------------------------------
    # Trajectory rollout (Euler integration with learned vector field)
    # ------------------------------------------------------------------

    def rollout(
        self,
        q0: float,
        p0: float,
        theta: np.ndarray,
        dt: float = 0.05,
        n_steps: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Integrate trajectory using learned Q-HNN vector field (Euler method).

        Returns (t, q, p) arrays.
        """
        t_arr = np.arange(n_steps) * dt
        q_arr = np.zeros(n_steps)
        p_arr = np.zeros(n_steps)
        q_arr[0], p_arr[0] = q0, p0

        for i in range(n_steps - 1):
            dq, dp = self.vector_field(q_arr[i], p_arr[i], theta)
            q_arr[i+1] = q_arr[i] + dt * dq
            p_arr[i+1] = p_arr[i] + dt * dp

        return t_arr, q_arr, p_arr

    def symplectic_rollout(
        self,
        q0: float,
        p0: float,
        theta: np.ndarray,
        dt: float = 0.05,
        n_steps: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Integrate trajectory using Störmer–Verlet (leapfrog) symplectic integrator.

        Unlike first-order Euler, Störmer–Verlet is a symplectic method that
        preserves the symplectic 2-form dq∧dp to machine precision, keeping
        energy bounded over long rollouts rather than drifting linearly.

        Algorithm (velocity Verlet):
            p_{n+1/2} = p_n  + (dt/2) · ṗ(q_n,   p_n)
            q_{n+1}   = q_n  +  dt    · q̇(q_n,   p_{n+1/2})
            p_{n+1}   = p_{n+1/2} + (dt/2) · ṗ(q_{n+1}, p_{n+1/2})

        This costs 4N circuit evaluations per step (same as Euler) but achieves
        O(dt²) energy conservation vs O(dt) for Euler.

        Returns (t, q, p) arrays.
        """
        t_arr = np.arange(n_steps) * dt
        q_arr = np.zeros(n_steps)
        p_arr = np.zeros(n_steps)
        q_arr[0], p_arr[0] = q0, p0

        for i in range(n_steps - 1):
            q, p = q_arr[i], p_arr[i]
            # Half-step momentum update
            p_half = p + 0.5 * dt * self.p_dot(q, p, theta)
            # Full-step position update (using half-step momentum)
            q_new = q + dt * self.q_dot(q, p_half, theta)
            # Half-step momentum update (using new position)
            p_new = p_half + 0.5 * dt * self.p_dot(q_new, p_half, theta)
            q_arr[i + 1], p_arr[i + 1] = q_new, p_new

        return t_arr, q_arr, p_arr

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_weights(self) -> np.ndarray:
        """
        Initialise the full parameter vector phi = [θ₀…θ_{n-1}, s, b].

        Circuit weights θ: uniform in [−π/4, π/4] — small enough that the
        circuit starts near-identity but large enough for non-trivial gradients.
        Scale s: initialised to 2.0 — gives PSR gradients ≤ 1.0 headroom
        to match the pendulum's ∂H/∂p = p ≤ 1 and ∂H/∂q = sin(q) ≤ 1.
        Offset b: initialised to 0.0.
        """
        rng = np.random.default_rng(self.seed)
        theta = rng.uniform(-np.pi / 4, np.pi / 4, self.n_circuit_weights)
        s = np.array([2.0])   # energy scale
        b = np.array([0.0])   # energy offset
        return np.concatenate([theta, s, b])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        trained = self.theta_opt is not None
        return (
            f"QuantumHNN(n_layers={self.n_layers}, "
            f"n_weights={self.n_weights}, trained={trained})"
        )

    def circuit_diagram(self) -> str:
        """Return text representation of the circuit."""
        return str(self.circuit.draw("text"))
