"""
common/parameter_shift.py
=========================
Generalised parameter-shift rule for quantum circuits.

Two uses:
  1. Weight-gate shift  — gradient of circuit output w.r.t. trainable parameters θᵢ:
         ∂⟨O⟩/∂θᵢ = ½[⟨O⟩(θᵢ + π/2) − ⟨O⟩(θᵢ − π/2)]

  2. Data-encoding gate shift  — gradient of circuit output w.r.t. input data x:
         ∂⟨O⟩/∂xᵢ = ½[⟨O⟩(xᵢ + π/2) − ⟨O⟩(xᵢ − π/2)]

For the Q-HNN, the Symplectic gradient (Hamilton's equations) is extracted by
applying the data-encoding shift to the q_in and p_in parameters:
     q̇_pred = ∂H/∂p  ≡  data-shift on p_in
     ṗ_pred = −∂H/∂q ≡  −data-shift on q_in

Reference: Q-Hamiltonian.ipynb §2 ("The Correct Method") and Q-PortHamiltonian.ipynb §3
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Shift constant (works for standard generator gates: Rx, Ry, Rz, CRY, ...)
# ---------------------------------------------------------------------------
SHIFT = np.pi / 2.0


def parameter_shift_gradient(
    eval_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    idx: int,
    shift: float = SHIFT,
) -> float:
    """
    Compute the exact gradient of eval_fn w.r.t. params[idx] via parameter-shift.

    Parameters
    ----------
    eval_fn : Callable[[np.ndarray], float]
        Function that evaluates the quantum circuit and returns a scalar
        expectation value given a full parameter vector.
    params : np.ndarray
        Current parameter vector.
    idx : int
        Index of the parameter to differentiate.
    shift : float
        Shift constant (default π/2 for standard rotation gates).

    Returns
    -------
    float
        Exact gradient ∂⟨O⟩/∂params[idx].
    """
    p_plus = params.copy()
    p_plus[idx] += shift
    p_minus = params.copy()
    p_minus[idx] -= shift
    return 0.5 * (eval_fn(p_plus) - eval_fn(p_minus))


def full_gradient(
    eval_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    shift: float = SHIFT,
) -> np.ndarray:
    """
    Compute the full gradient vector via parameter-shift for all parameters.

    Parameters
    ----------
    eval_fn : Callable[[np.ndarray], float]
        Circuit evaluation function.
    params : np.ndarray
        Current parameter vector of length n.
    shift : float
        Shift constant.

    Returns
    -------
    np.ndarray
        Gradient vector of shape (n,).
    """
    grad = np.zeros_like(params)
    for i in range(len(params)):
        grad[i] = parameter_shift_gradient(eval_fn, params, i, shift)
    return grad


class DataEncodingShift:
    """
    Computes the symplectic gradient (Hamilton's equations) from a quantum
    circuit that encodes phase-space coordinates as rotation angles.

    The energy circuit evaluates H(q, p; θ) = ⟨ZZ⟩.
    The Hamiltonian vector field is:
        q̇ =  ∂H/∂p  ≈  ½[H(q, p+π/2; θ) − H(q, p−π/2; θ)]
        ṗ = −∂H/∂q  ≈ −½[H(q+π/2, p; θ) − H(q−π/2, p; θ)]

    Parameters
    ----------
    eval_energy : Callable[[float, float, np.ndarray], float]
        Function eval_energy(q, p, theta) → scalar H(q,p).
    shift : float
        Shift constant (default π/2).
    """

    def __init__(
        self,
        eval_energy: Callable[[float, float, np.ndarray], float],
        shift: float = SHIFT,
    ):
        self.eval_energy = eval_energy
        self.shift = shift

    def q_dot(
        self, q: float, p: float, theta: np.ndarray
    ) -> float:
        """Compute q̇ = ∂H/∂p via data-encoding parameter shift on p."""
        H_plus  = self.eval_energy(q, p + self.shift, theta)
        H_minus = self.eval_energy(q, p - self.shift, theta)
        return 0.5 * (H_plus - H_minus)

    def p_dot(
        self, q: float, p: float, theta: np.ndarray
    ) -> float:
        """Compute ṗ = −∂H/∂q via data-encoding parameter shift on q."""
        H_plus  = self.eval_energy(q + self.shift, p, theta)
        H_minus = self.eval_energy(q - self.shift, p, theta)
        return -0.5 * (H_plus - H_minus)

    def vector_field(
        self, q: float, p: float, theta: np.ndarray
    ) -> tuple[float, float]:
        """Return (q̇, ṗ) Hamiltonian vector field at (q, p)."""
        return self.q_dot(q, p, theta), self.p_dot(q, p, theta)


class DissipativeDataShift(DataEncodingShift):
    """
    Extends DataEncodingShift for port-Hamiltonian dissipative systems.

    The dissipative (R) contribution is modelled as a classical scalar
    damping coefficient γ added to the momentum equation:
        q̇ =  ∂H/∂p
        ṗ = −∂H/∂q − γ·p

    This allows learning H(q,p) on the quantum circuit while fitting the
    damping coefficient γ classically as an additional scalar parameter.
    """

    def p_dot_dissipative(
        self,
        q: float,
        p: float,
        theta: np.ndarray,
        gamma: float,
    ) -> float:
        """
        Compute ṗ = −∂H/∂q − γ·p (port-Hamiltonian dissipative flow).

        Parameters
        ----------
        q, p : float
            Phase-space coordinates.
        theta : np.ndarray
            Quantum circuit weight parameters.
        gamma : float
            Scalar damping coefficient (learned alongside θ).
        """
        conservative_p_dot = self.p_dot(q, p, theta)
        return conservative_p_dot - gamma * p

    def vector_field_dissipative(
        self,
        q: float,
        p: float,
        theta: np.ndarray,
        gamma: float,
    ) -> tuple[float, float]:
        """Return (q̇, ṗ) for the port-Hamiltonian dissipative vector field."""
        q_d = self.q_dot(q, p, theta)
        p_d = self.p_dot_dissipative(q, p, theta, gamma)
        return q_d, p_d
