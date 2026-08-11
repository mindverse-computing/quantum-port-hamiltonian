"""
tests/test_qgnn.py
==================
Unit tests for the topology-entangled QGNN energy surrogate.

Checks:
  1. energy() returns a finite scalar and respects the topology (absent edges
     ⇒ no ZZ gate on that pair).
  2. Per-node parameter-shift gradients match central finite differences.
  3. The energy is invariant under a graph automorphism (permutation
     equivariance) when node parameters and readout weights respect the symmetry.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.qgnn_energy import QGNNEnergy, edges_from_coupling


def test_energy_finite_and_topology():
    # Ring on N=3 (a triangle): edges (0,1),(1,2),(0,2)
    K = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    edges = edges_from_coupling(K)
    assert edges == [(0, 1), (0, 2), (1, 2)]
    m = QGNNEnergy(num_nodes=3, edges=edges, n_layers=2, seed=0)
    th = m.init_weights()
    state = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.2])  # [φ_0..2, ω_0..2]
    e = m.energy(state, th)
    assert np.isfinite(e)
    # Energy magnitude bounded by Σ|a_i| + Σ|w_ij| = 3 + 3 = 6
    assert abs(e) <= 6.0 + 1e-9

    # Topology: a path graph 0-1-2 has NO (0,2) edge → fewer edge weights
    Kpath = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    ep = edges_from_coupling(Kpath)
    assert ep == [(0, 1), (1, 2)]
    mp = QGNNEnergy(num_nodes=3, edges=ep, n_layers=2, seed=0)
    assert mp.n_edge_weights == 2 * 2  # 2 edges × 2 layers


def test_parameter_shift_matches_finite_difference():
    K = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)  # path 0-1-2
    edges = edges_from_coupling(K)
    m = QGNNEnergy(num_nodes=3, edges=edges, n_layers=2, seed=7)
    th = m.init_weights()
    state = np.array([0.25, 0.4, -0.15, -0.3, 0.2, 0.35])

    eps = 1e-5
    for i in range(3):
        # ∂H/∂ω_i
        sp = state.copy(); sp[3 + i] += eps
        sm = state.copy(); sm[3 + i] -= eps
        fd = (m.energy(sp, th) - m.energy(sm, th)) / (2 * eps)
        ps = m.grad_x2(state, th, i)
        assert abs(fd - ps) < 1e-4, f"grad_x2 node {i}: fd={fd} ps={ps}"

        # ∂H/∂φ_i
        sp = state.copy(); sp[i] += eps
        sm = state.copy(); sm[i] -= eps
        fd = (m.energy(sp, th) - m.energy(sm, th)) / (2 * eps)
        ps = m.grad_x1(state, th, i)
        assert abs(fd - ps) < 1e-4, f"grad_x1 node {i}: fd={fd} ps={ps}"


def test_permutation_equivariance():
    """
    Triangle graph with all-equal edges is symmetric under swapping nodes 0<->1.
    If node params and readout are homogeneous, swapping the (φ,ω) of nodes 0,1
    must leave the energy invariant.
    """
    K = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    edges = edges_from_coupling(K)
    m = QGNNEnergy(num_nodes=3, edges=edges, n_layers=2, seed=3)

    # Make variational weights homogeneous across the 0<->1 symmetry:
    # simplest sufficient condition — identical params on all nodes/edges.
    th = np.full(m.n_weights, 0.17)

    state = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.2])
    # Permute nodes 0<->1: swap φ_0<->φ_1 and ω_0<->ω_1
    perm = state.copy()
    perm[[0, 1]] = state[[1, 0]]
    perm[[3, 4]] = state[[4, 3]]

    e0 = m.energy(state, th)
    e1 = m.energy(perm, th)
    assert abs(e0 - e1) < 1e-9, f"permutation broke invariance: {e0} vs {e1}"


if __name__ == "__main__":
    test_energy_finite_and_topology()
    test_parameter_shift_matches_finite_difference()
    test_permutation_equivariance()
    print("ALL QGNN TESTS PASSED")
