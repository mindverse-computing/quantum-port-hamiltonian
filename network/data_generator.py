"""
network/data_generator.py
=========================
Network data generators for the quantum trainer.

Ports the classical Kuramoto/phasor physics from hnn-generic/data/systems.py,
scaled to small networks (N = 3–6 nodes) and angle-encoded into the qubit
working range [-π/2, π/2] via IsomorphicMapping.

Two systems (mirroring the classical playground):
  • gen_network_conservative : γ = 0 Kuramoto phasors on the N-torus
                               (energy conserved, pure J-channel).
  • gen_network_dissipative  : γ > 0 damped Kuramoto network
                               (energy decays, R = diag(γ)).

Both return a NetworkVectorFieldDataset of random phase-space points and their
exact port-Hamiltonian vector field (φ̇, ω̇), plus the true coupling matrix K
and damping vector γ.

See theory/network_qphnn_theory.md §1 (Eqs. 1–4).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Coupling-matrix builders
# ---------------------------------------------------------------------------

def build_ring_coupling(N: int, strength: float = 1.0) -> np.ndarray:
    """Nearest-neighbour ring coupling (each node coupled to i±1 mod N)."""
    K = np.zeros((N, N))
    for i in range(N):
        j = (i + 1) % N
        K[i, j] = K[j, i] = strength
    return K


def build_modular_coupling(
    clusters: int = 2, nodes_per_cluster: int = 2,
    intra: float = 1.5, inter: float = 0.5, seed: int = 42,
) -> np.ndarray:
    """
    Small block-modular coupling: dense intra-cluster, sparse inter-cluster.
    A miniature of hnn-generic's _build_modular_coupling for N = clusters*npc.
    """
    rng = np.random.default_rng(seed)
    N = clusters * nodes_per_cluster
    K = np.zeros((N, N))
    for a in range(clusters):
        for b in range(clusters):
            si, ei = a * nodes_per_cluster, (a + 1) * nodes_per_cluster
            sj, ej = b * nodes_per_cluster, (b + 1) * nodes_per_cluster
            if a == b:
                blk = rng.normal(intra, 0.2, (nodes_per_cluster, nodes_per_cluster))
            else:
                blk = rng.normal(inter, 0.15, (nodes_per_cluster, nodes_per_cluster))
            K[si:ei, sj:ej] = blk
    K = (K + K.T) / 2
    np.fill_diagonal(K, 0.0)
    return K


def build_star_coupling(N: int, strength: float = 1.0) -> np.ndarray:
    """Hub-and-spoke (star) coupling: node 0 is a central hub coupled to every
    other node; the periphery is otherwise uncoupled.

    A generic stand-in for a *regulatory-hub* motif (one master regulator /
    relay driving many downstream units) — e.g. a transcription-factor hub in a
    gene network or a thalamic relay in a cortical network. Domain-agnostic:
    it is a coupled-oscillator topology, not omics- or BCI-specific data.
    """
    K = np.zeros((N, N))
    for j in range(1, N):
        K[0, j] = K[j, 0] = strength
    return K


def build_chain_coupling(N: int, strength: float = 1.0) -> np.ndarray:
    """Open linear chain (path) coupling: node i coupled to i+1 only.

    A generic stand-in for a *cascade* motif (feed-forward signalling / sensory
    pathway) where interaction is strictly nearest-neighbour along a line, with
    no wrap-around (unlike the ring). Domain-agnostic.
    """
    K = np.zeros((N, N))
    for i in range(N - 1):
        K[i, i + 1] = K[i + 1, i] = strength
    return K


# ---------------------------------------------------------------------------
# Kuramoto network vector field (Eq. 1–4)
# ---------------------------------------------------------------------------

def kuramoto_field(
    phi: np.ndarray, omega: np.ndarray, K: np.ndarray, gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Damped Kuramoto network vector field.

        φ̇_i = ω_i
        ω̇_i = −Σ_j K_ij sin(φ_i − φ_j) − γ_i ω_i

    phi, omega : (B, N)  batched states.
    Returns (dphi, domega) each (B, N).
    """
    B, N = phi.shape
    dphi = omega.copy()
    domega = np.zeros_like(omega)
    for i in range(N):
        coupling = np.zeros(B)
        for j in range(N):
            if i != j and K[i, j] != 0:
                coupling += K[i, j] * np.sin(phi[:, i] - phi[:, j])
        domega[:, i] = -coupling - gamma[i] * omega[:, i]
    return dphi, domega


def kuramoto_hamiltonian(
    phi: np.ndarray, omega: np.ndarray, K: np.ndarray,
) -> np.ndarray:
    """H = ½Σ ω_i² + ½Σ_{i≠j} K_ij[1 − cos(φ_i − φ_j)]  (Eq. 4).  (B,) out."""
    kinetic = 0.5 * (omega ** 2).sum(axis=1)
    dphi = phi[:, :, None] - phi[:, None, :]              # (B, N, N)
    potential = 0.5 * (K[None] * (1 - np.cos(dphi))).sum(axis=(1, 2))
    return kinetic + potential


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class NetworkVectorFieldDataset:
    """Random network phase-space points and their exact vector field."""
    states: np.ndarray      # (B, 2N) = [φ | ω]
    d_states: np.ndarray    # (B, 2N) = [φ̇ | ω̇]
    K: np.ndarray           # (N, N) true coupling
    gamma: np.ndarray       # (N,) true damping
    H_true: np.ndarray      # (B,) true Hamiltonian
    name: str = "network_dataset"
    q_scale: float = 1.0
    p_scale: float = 1.0

    @property
    def N(self) -> int:
        return self.K.shape[0]

    @property
    def n_samples(self) -> int:
        return len(self.states)

    def split(self, test_frac: float = 0.2, seed: int = 0):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(self.n_samples)
        n_test = max(1, int(self.n_samples * test_frac))
        te, tr = idx[:n_test], idx[n_test:]
        mk = lambda ii, nm: NetworkVectorFieldDataset(
            self.states[ii], self.d_states[ii], self.K, self.gamma,
            self.H_true[ii], nm, self.q_scale, self.p_scale)
        return mk(tr, self.name + "_train"), mk(te, self.name + "_test")


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _generate(
    K: np.ndarray, gamma: np.ndarray, n_samples: int, seed: int,
    phi_spread: float, omega_spread: float, name: str,
) -> NetworkVectorFieldDataset:
    """
    Sample random (φ, ω) and compute the exact Kuramoto vector field, then
    angle-scale φ, ω into the qubit working range [-π/2, π/2].

    The scaling is a linear coordinate change; the vector field transforms
    consistently (φ̇, ω̇ scaled by the same factors) so the learned quantum
    model sees a well-posed regression target on the encoded coordinates.
    """
    rng = np.random.default_rng(seed)
    N = K.shape[0]
    phi = rng.uniform(-phi_spread, phi_spread, (n_samples, N))
    omega = rng.uniform(-omega_spread, omega_spread, (n_samples, N))

    dphi, domega = kuramoto_field(phi, omega, K, gamma)
    H_true = kuramoto_hamiltonian(phi, omega, K)

    # UNIFORM angle scaling: one scale s for BOTH φ and ω so that the largest
    # coordinate maps into the qubit working range [-π/2, π/2]. A uniform scale
    # is REQUIRED to preserve the canonical symplectic structure — with
    # different q/p scales the true field is no longer the symplectic gradient
    # of a single H, and the parameter-shift model cannot fit it consistently.
    target = np.pi / 2
    scale = target / max(phi_spread, omega_spread, 1e-9)

    phi_e, omega_e = phi * scale, omega * scale
    dphi_e, domega_e = dphi * scale, domega * scale

    states = np.hstack([phi_e, omega_e])
    d_states = np.hstack([dphi_e, domega_e])
    return NetworkVectorFieldDataset(
        states, d_states, K, gamma, H_true, name, scale, scale)


def gen_network_conservative(
    K: np.ndarray | None = None, N: int = 4,
    n_samples: int = 60, seed: int = 0,
) -> NetworkVectorFieldDataset:
    """
    Conservative Kuramoto phasor network (γ = 0) on the N-torus.
    Pure J-channel; energy is conserved.
    """
    if K is None:
        K = build_ring_coupling(N, strength=1.0)
    N = K.shape[0]
    gamma = np.zeros(N)
    return _generate(K, gamma, n_samples, seed,
                     phi_spread=np.pi / 2, omega_spread=1.0,
                     name=f"network_conservative_N{N}")


def gen_network_dissipative(
    K: np.ndarray | None = None, N: int = 4,
    gamma_range: tuple[float, float] = (0.15, 0.45),
    n_samples: int = 60, seed: int = 0,
) -> NetworkVectorFieldDataset:
    """
    Dissipative damped Kuramoto network (γ > 0).
    R = diag(γ) ≻ 0; energy decays monotonically.
    """
    if K is None:
        K = build_ring_coupling(N, strength=1.0)
    N = K.shape[0]
    rng = np.random.default_rng(seed + 100)
    gamma = rng.uniform(gamma_range[0], gamma_range[1], N)
    return _generate(K, gamma, n_samples, seed,
                     phi_spread=np.pi / 2, omega_spread=1.0,
                     name=f"network_dissipative_N{N}")
