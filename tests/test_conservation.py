"""
Energy conservation for the conservative model.

The manuscript's claim is structural rather than statistical: because the
conservative flow is carried by a unitary, energy is conserved *by construction*
at any parameter values, trained or not. These tests assert that at the level
where it can actually fail — the circuit, the symplectic integrator, and the
distinction between the two error sources.
"""
import numpy as np
import pytest

from common.metrics import compute_energy_conservation
from non_dissipative.quantum_hnn import QuantumHNN
from non_dissipative.data_generator import NonlinearPendulum


def test_conservation_is_structural_not_trained():
    """dH/dt vanishes along the model's own field, at *random* parameters.

    This is the manuscript's structural claim stated exactly. Energy conservation
    is a property of the skew coupling between H and its symplectic gradients:

        dH/dt = ∂H/∂q · q̇ + ∂H/∂p · ṗ
              = ∂H/∂q · (+∂H/∂p) + ∂H/∂p · (-∂H/∂q) = 0

    It therefore holds identically, for any parameter values, trained or not.
    Checked here to ~1e-10 across a phase-space grid at four random parameter
    draws.

    Note what this does NOT claim: a rollout still accumulates *integrator*
    error, which is a property of the discrete time step and not of the circuit.
    That separate quantity is what `test_symplectic_integrator_beats_euler`
    measures.
    """
    rng = np.random.default_rng(3)
    model = QuantumHNN(n_layers=1, seed=1)
    h = 1e-5

    for draw in range(4):
        phi = model.init_weights()
        phi[: model.n_circuit_weights] = rng.uniform(-np.pi, np.pi, model.n_circuit_weights)

        worst = 0.0
        for q in np.linspace(-1.2, 1.2, 7):
            for p in np.linspace(-1.0, 1.0, 7):
                qdot, pdot = model.vector_field(q, p, phi)
                dH_dq = (model.energy(q + h, p, phi) - model.energy(q - h, p, phi)) / (2 * h)
                dH_dp = (model.energy(q, p + h, phi) - model.energy(q, p - h, phi)) / (2 * h)
                worst = max(worst, abs(dH_dq * qdot + dH_dp * pdot))

        assert worst < 1e-8, f"draw {draw}: max |dH/dt| = {worst:.2e} — not structural"


def test_symplectic_integrator_beats_euler():
    """Störmer–Verlet keeps the orbit bounded where explicit Euler does not.

    The manuscript reports 1.35% against 12.3% for the trained model; this
    asserts the qualitative ordering that claim depends on, which must hold for
    any symplectic-vs-explicit pair.
    """
    model = QuantumHNN(n_layers=1, seed=1)
    phi = model.init_weights()

    sv = model.symplectic_rollout(0.8, 0.0, phi, dt=0.05, n_steps=300)
    H_sv = np.array([model.energy(q, p, phi) for q, p in zip(sv[0], sv[1])])
    drift_sv, _ = compute_energy_conservation(H_sv)

    eu = model.rollout(0.8, 0.0, phi, dt=0.05, n_steps=300)
    H_eu = np.array([model.energy(q, p, phi) for q, p in zip(eu[0], eu[1])])
    drift_eu, _ = compute_energy_conservation(H_eu)

    assert drift_sv < drift_eu, f"symplectic {drift_sv:.4f} not better than Euler {drift_eu:.4f}"


def test_energy_is_a_state_function():
    """H depends on (q, p) alone — the same point returns the same energy."""
    model = QuantumHNN(n_layers=1, seed=1)
    phi = model.init_weights()

    first = model.energy(0.3, 0.7, phi)
    _ = model.symplectic_rollout(-1.0, 0.5, phi, dt=0.05, n_steps=50)
    second = model.energy(0.3, 0.7, phi)

    assert first == pytest.approx(second, abs=1e-12)


def test_ground_truth_pendulum_conserves_energy():
    """The reference system itself conserves energy under its own integrator.

    Guards the baseline: a drifting reference would make every model comparison
    meaningless.
    """
    sysm = NonlinearPendulum()
    t, q, p = sysm.integrate(0.8, 0.0, dt=0.01, n_steps=1000)

    # guard the unpacking order itself: t is the monotonic one
    assert np.all(np.diff(t) > 0), "integrate() returns (t, q, p) — order changed"

    H = sysm.hamiltonian(q, p)
    drift, _ = compute_energy_conservation(H)
    assert drift < 1e-6, f"reference pendulum drifts {drift:.2e}"
