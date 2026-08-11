"""
Parameter-shift rule.

The manuscript's central computational claim about gradients is that the
symplectic vector field is obtained *exactly* from two circuit evaluations per
component, with no finite-difference truncation error. These tests hold the rule
to that standard: agreement with a high-accuracy finite difference, exactness on
the trigonometric form the rule assumes, and the antisymmetry that makes the
field symplectic rather than merely smooth.
"""
import numpy as np
import pytest

from common.parameter_shift import parameter_shift_gradient
from non_dissipative.quantum_hnn import QuantumHNN


def test_matches_central_difference_on_gate_angles():
    """Parameter-shift and a fine central difference agree on the circuit weights.

    The parameter vector is [θ₀…θ_{n-1}, s, b]: the leading entries are gate
    rotation angles, and the trailing two are the classical energy scale and
    offset. The shift rule applies to the gate angles only — it assumes the
    expectation is sinusoidal in the parameter, which holds for a rotation angle
    and not for a multiplicative scale.
    """
    model = QuantumHNN(n_layers=1, seed=0)
    theta = model.init_weights()

    def energy_of(params):
        return model.energy(0.4, -0.3, params)

    h = 1e-5
    for idx in range(model.n_circuit_weights):
        ps = parameter_shift_gradient(energy_of, theta, idx)
        fwd, bwd = theta.copy(), theta.copy()
        fwd[idx] += h
        bwd[idx] -= h
        fd = (energy_of(fwd) - energy_of(bwd)) / (2 * h)
        assert ps == pytest.approx(fd, abs=1e-4), f"param {idx}: shift={ps} fd={fd}"


def test_shift_rule_does_not_apply_to_the_energy_scale():
    """The scale s enters linearly, so the shift rule must NOT be used on it.

    Guards a real trap: applying the rule to index -2 returns a value that looks
    plausible but is not dH/ds. The scale's derivative is the raw expectation,
    available in closed form.
    """
    model = QuantumHNN(n_layers=1, seed=0)
    phi = model.init_weights()
    s_idx = len(phi) - 2

    def energy_of(params):
        return model.energy(0.4, -0.3, params)

    h = 1e-6
    fwd, bwd = phi.copy(), phi.copy()
    fwd[s_idx] += h
    bwd[s_idx] -= h
    true_dH_ds = (energy_of(fwd) - energy_of(bwd)) / (2 * h)

    misapplied = parameter_shift_gradient(energy_of, phi, s_idx)
    assert misapplied != pytest.approx(true_dH_ds, abs=1e-3)

    raw_zz = model._raw_zz(0.4, -0.3, phi[: model.n_circuit_weights])
    assert true_dH_ds == pytest.approx(raw_zz, abs=1e-5)


def test_exact_on_pure_sinusoid():
    """On f(x) = a*cos(x + phi) the rule is exact, not approximate.

    This is the property that distinguishes it from finite differences: the
    error is zero by construction, not O(h^2).
    """
    a, phi = 1.7, 0.63

    def f(params):
        return a * np.cos(params[0] + phi)

    for x in (-1.2, 0.0, 0.5, 2.9):
        got = parameter_shift_gradient(f, np.array([x]), 0)
        exact = -a * np.sin(x + phi)
        assert got == pytest.approx(exact, abs=1e-12)


def test_symplectic_sign_convention():
    """q̇ = +∂H/∂p and ṗ = -∂H/∂q — the sign that makes the flow symplectic."""
    model = QuantumHNN(n_layers=1, seed=0)
    theta = model.init_weights()
    q, p = 0.55, -0.2

    qdot, pdot = model.vector_field(q, p, theta)

    h = 1e-5
    dH_dp = (model.energy(q, p + h, theta) - model.energy(q, p - h, theta)) / (2 * h)
    dH_dq = (model.energy(q + h, p, theta) - model.energy(q - h, p, theta)) / (2 * h)

    assert qdot == pytest.approx(dH_dp, abs=1e-4)
    assert pdot == pytest.approx(-dH_dq, abs=1e-4)


def test_two_evaluations_suffice():
    """The rule consumes exactly two circuit evaluations per parameter."""
    calls = []

    def counting_energy(params):
        calls.append(params.copy())
        return float(np.cos(params[0]))

    parameter_shift_gradient(counting_energy, np.array([0.3]), 0)
    assert len(calls) == 2

    shifted = sorted(c[0] for c in calls)
    assert shifted[0] == pytest.approx(0.3 - np.pi / 2)
    assert shifted[1] == pytest.approx(0.3 + np.pi / 2)
