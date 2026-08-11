"""
common/isomorphic_mapping.py
============================
Isomorphic Hamiltonian Mapping (IHM): encodes classical phase-space coordinates
(q, p) into qubit rotation angles for state preparation, and decodes Pauli
expectation values back to classical coordinates.

Mathematical basis:
    q(t) ∝ <ψ(t)|σ_x|ψ(t)>
    p(t) ∝ <ψ(t)|σ_y|ψ(t)>

The system qubit starts in |+⟩ (H|0⟩), so <σ_x> = 1, <σ_y> = 0.
Rx(angle) rotates in the YZ-plane, changing <σ_y>; Ry rotates in the XZ-plane.
For our IHM, we use the Bloch-sphere angle encoding:
    q_encoded = q_val  (directly as rotation angle, assuming data pre-normalised to [-π, π])
    p_encoded = p_val

Reference: Q-PortHamiltonian.ipynb §2 (Isomorphic Hamiltonian Mapping)
"""

from __future__ import annotations

import numpy as np


class IsomorphicMapping:
    """
    Encodes classical phase-space coordinates to qubit rotation angles
    and decodes Pauli expectations back to classical coordinates.

    Parameters
    ----------
    q_scale : float
        Scaling factor to map classical q range to [-π, π].
    p_scale : float
        Scaling factor to map classical p range to [-π, π].
    """

    def __init__(self, q_scale: float = 1.0, p_scale: float = 1.0):
        self.q_scale = q_scale
        self.p_scale = p_scale

    # ------------------------------------------------------------------
    # Encoding: classical → angle
    # ------------------------------------------------------------------

    def encode_q(self, q: float | np.ndarray) -> float | np.ndarray:
        """Map position coordinate to rotation angle (radians)."""
        return np.asarray(q) * self.q_scale

    def encode_p(self, p: float | np.ndarray) -> float | np.ndarray:
        """Map momentum coordinate to rotation angle (radians)."""
        return np.asarray(p) * self.p_scale

    def encode(
        self, q: float | np.ndarray, p: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Encode (q, p) pair to (angle_q, angle_p)."""
        return self.encode_q(q), self.encode_p(p)

    # ------------------------------------------------------------------
    # Decoding: Pauli expectation → classical
    # ------------------------------------------------------------------

    def decode_q(self, expval_x: float | np.ndarray) -> float | np.ndarray:
        """Recover position from <σ_x> expectation value."""
        return np.asarray(expval_x) / self.q_scale

    def decode_p(self, expval_y: float | np.ndarray) -> float | np.ndarray:
        """Recover momentum from <σ_y> expectation value."""
        return np.asarray(expval_y) / self.p_scale

    # ------------------------------------------------------------------
    # Normalisation helpers (call before encoding raw data)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_scales(
        q_data: np.ndarray,
        p_data: np.ndarray,
        target_range: float = np.pi / 2,
    ) -> tuple[float, float]:
        """
        Compute q_scale and p_scale so that max(|q|) maps to target_range.

        Parameters
        ----------
        q_data, p_data : np.ndarray
            Raw coordinate arrays.
        target_range : float
            Desired maximum encoded angle (default π/2).

        Returns
        -------
        q_scale, p_scale : float
        """
        q_max = float(np.max(np.abs(q_data))) or 1.0
        p_max = float(np.max(np.abs(p_data))) or 1.0
        return target_range / q_max, target_range / p_max

    # ------------------------------------------------------------------
    # State preparation angle for 1D initial condition
    # ------------------------------------------------------------------

    def initial_angle(self, q0: float) -> float:
        """
        Compute the Rx rotation angle to prepare the system qubit in a state
        where <σ_x> ≈ cos(angle_q0).

        For q0 near 0 the qubit starts close to |+⟩ (<σ_x> = 1).
        For q0 = π/2 the qubit is near |0⟩ (<σ_x> = 0).
        """
        return self.encode_q(q0)

    def __repr__(self) -> str:
        return (
            f"IsomorphicMapping(q_scale={self.q_scale:.4f}, "
            f"p_scale={self.p_scale:.4f})"
        )
