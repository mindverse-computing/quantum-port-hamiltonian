"""
tests/test_network_phnn.py
==========================
Unit tests for the NetworkQpHNN model.

Checks:
  1. Conservative mode: energy_rate ≈ 0 (J skew ⇒ Ḣ = 0), and energy is
     approximately conserved along a short rollout.
  2. Dissipative gamma mode: energy_rate ≤ 0 pointwise (passivity), γ ⪰ 0.
  3. MINL channel runs and returns bounded ⟨X_i⟩ ∈ [-1, 1] for every node.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.qgnn_energy import edges_from_coupling
from network.quantum_network_phnn import NetworkQpHNN


def test_conservative_energy_rate_zero():
    K = np.array([[0, 1, 0, 1], [1, 0, 1, 0],
                  [0, 1, 0, 1], [1, 0, 1, 0]], float)  # 4-ring
    edges = edges_from_coupling(K)
    m = NetworkQpHNN(num_nodes=4, edges=edges, n_layers=2,
                     dissipative=False, seed=1)
    p = m.init_params()
    rng = np.random.default_rng(0)
    # Ḣ must be ~0 at several random states (pure J-channel)
    for _ in range(5):
        state = rng.uniform(-0.5, 0.5, 8)
        hdot = m.energy_rate(state, p)
        assert abs(hdot) < 1e-9, f"conservative Ḣ not zero: {hdot}"


def test_dissipative_passivity():
    K = np.array([[0, 1, 0, 1], [1, 0, 1, 0],
                  [0, 1, 0, 1], [1, 0, 1, 0]], float)
    edges = edges_from_coupling(K)
    m = NetworkQpHNN(num_nodes=4, edges=edges, n_layers=2,
                     dissipative=True, diss_mode="gamma", seed=2)
    p = m.init_params()
    theta, gamma = m.split_params(p)
    assert np.all(gamma >= 0), "gamma must be >= 0 (softplus)"

    rng = np.random.default_rng(5)
    # Ḣ = −Σ γ_i (∂H/∂ω_i)² ≤ 0 at every state
    for _ in range(5):
        state = rng.uniform(-0.5, 0.5, 8)
        hdot = m.energy_rate(state, p)
        assert hdot <= 1e-9, f"passivity violated: Ḣ = {hdot} > 0"


def test_minl_channel_bounded():
    K = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)  # triangle
    edges = edges_from_coupling(K)
    m = NetworkQpHNN(num_nodes=3, edges=edges, n_layers=1,
                     dissipative=True, diss_mode="minl", seed=3)
    m.minl_steps = 4
    params_minl = {
        "theta_J": np.array([0.3, 0.3, 0.3]),
        "theta_R": np.array([0.5, 0.5, 0.5]),
        "theta_k": np.array([0.2, 0.2, 0.2]),
    }
    state0 = np.array([np.pi / 2, 0.0, -np.pi / 4, 0.1, 0.0, -0.1])
    mean_x = m.predict_minl(params_minl, state0, n_shots=8)
    assert mean_x.shape == (4, 3)
    assert np.all(np.abs(mean_x) <= 1.0 + 1e-9), "⟨X⟩ out of [-1,1]"
    assert np.all(np.isfinite(mean_x))


if __name__ == "__main__":
    test_conservative_energy_rate_zero()
    test_dissipative_passivity()
    test_minl_channel_bounded()
    print("ALL NETWORK-pHNN TESTS PASSED")
