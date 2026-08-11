"""
Measurement-induced dissipation.

The manuscript's central claim is that damping is produced by Born-rule
measurement — a genuine CPTP channel — and not by a non-unitary term or a
classical damping coefficient. These tests hold that claim where it is
falsifiable: passivity of the dissipative flow, the monotone-decrease property
the MINL circuit is built to realise, and the control that the effect disappears
when the dissipative kick is switched off.
"""
import numpy as np
import pytest

from common.metrics import compute_energy_monotone_fraction, compute_energy_conservation
from dissipative.quantum_phnn import DynamicQpHNN, VectorFieldQpHNN
from dissipative.data_generator import DampedHarmonicOscillator


def _phase_space_energy(sx, sy):
    """H = ½(⟨σx⟩² + ⟨σy⟩²) — the phase-space energy of the encoded state.

    ``run_trajectory`` returns the Bloch components the encoding uses as (q, p),
    so the oscillator energy is their squared radius.
    """
    sx, sy = np.asarray(sx), np.asarray(sy)
    return 0.5 * (sx**2 + sy**2)


def test_minl_energy_decreases_monotonically():
    """Energy decreases at (nearly) every measurement round.

    The manuscript reports f_mono = 100% over 30 independent runs. Born-rule
    measurement is stochastic, so this asserts the property over several
    independent trajectories rather than a single lucky one.
    """
    model = DynamicQpHNN(n_steps=6, seed=7)
    params = np.array([-0.048, 0.624, 0.474])       # [θ_J, θ_R, θ_kick]

    fracs = []
    for _ in range(60):
        sx, sy = model.run_trajectory(params, 1.0)
        fracs.append(compute_energy_monotone_fraction(_phase_space_energy(sx, sy)))

    # Individual trajectories are stochastic: a measurement outcome can transiently
    # raise the energy, so a single run may dip below 1.0. Measured over 200
    # trajectories, 93% are perfectly monotone and the mean fraction is 0.971.
    # The threshold and sample size are set from that distribution — a 12-sample
    # mean falls below 0.9 about 3% of the time, which would make this test flaky.
    assert np.mean(fracs) >= 0.90, (
        f"mean monotone fraction {np.mean(fracs):.3f} over {len(fracs)} trajectories"
    )
    assert np.mean(np.array(fracs) == 1.0) >= 0.75, (
        f"only {np.mean(np.array(fracs) == 1.0):.0%} of trajectories fully monotone"
    )


def test_run_trajectory_ignores_its_rng_argument():
    """Known limitation, pinned by source inspection rather than by sampling.

    ``run_trajectory`` accepts an ``rng`` argument and binds it, but the
    Born-rule draw is performed by Qiskit's ``Statevector.measure()``, which
    takes no generator and uses its own global stream. The bound ``rng`` is
    never consumed.

    The consequence for users: an ensemble built by looping ``run_trajectory``
    with different ``rng`` values is neither independently seeded nor
    reproducible. ``predict_trajectory(n_shots=...)`` is the supported averaging
    path.

    Asserted against the source, because the observable symptom is
    state-dependent: whether two calls agree depends on where the global stream
    happens to be, so a sampling-based check would be flaky in either direction.
    """
    import inspect

    src = inspect.getsource(DynamicQpHNN.run_trajectory)
    body = src[src.index("rng = np.random.default_rng"):]

    assert "sv.measure(" in body, "measurement path changed — re-check this guard"
    assert "rng.random" not in body and "rng.integers" not in body and "rng=rng" not in body, (
        "run_trajectory now draws from its rng — the generator appears to be "
        "wired through; replace this guard with a reproducibility test"
    )


def test_dissipation_scales_with_the_system_bath_coupling():
    """Dose-response on θ_R: stronger coupling removes more energy, in the mean.

    θ_R sets the CRY rotation that entangles the system with its bath ancilla, so
    it controls how much of the state is exposed to the measurement. This is the
    knob that carries the dissipation.

    Averaged, because the channel is genuinely stochastic at strong coupling: at
    θ_R = 1.2 individual trajectories range over [-0.33, 1.00] and about 3% show
    a net energy *gain*, which is the Born rule doing its job rather than a bug.
    Weak coupling is effectively deterministic (zero spread at θ_R <= 0.3). A
    single-trajectory comparison here would be flaky by construction.
    """
    model = DynamicQpHNN(n_steps=6, seed=7)
    base = np.array([-0.048, 0.624, 0.474])          # [θ_J, θ_R, θ_kick]
    n_rep = 40

    means = []
    for theta_R in (0.0, 0.3, 0.624, 1.2):
        params = base.copy()
        params[1] = theta_R
        decays = []
        for _ in range(n_rep):
            sx, sy = model.run_trajectory(params, 1.0)
            H = _phase_space_energy(sx, sy)
            decays.append((H[0] - H[-1]) / abs(H[0]))
        means.append(float(np.mean(decays)))

    assert means[0] == pytest.approx(0.0, abs=1e-9), (
        f"θ_R=0 leaves the ancilla uncoupled, so energy must be conserved; "
        f"got mean decay {means[0]:.4f}"
    )
    assert all(b > a for a, b in zip(means, means[1:])), (
        f"mean decay not monotone in θ_R: {np.round(means, 4)}"
    )


def test_no_coupling_means_no_dissipation():
    """The null control that matters: θ_R = 0 conserves energy exactly.

    With the system-bath rotation switched off, the ancilla is still measured
    every step — the measurement schedule and circuit depth are unchanged — but
    the measurement carries no information about the system. Energy is then
    conserved to machine precision, which shows the loss comes from the designed
    channel rather than from the act of measuring.
    """
    model = DynamicQpHNN(n_steps=6, seed=7)
    params = np.array([-0.048, 0.0, 0.474])

    sx, sy = model.run_trajectory(params, 1.0)
    H = _phase_space_energy(sx, sy)

    assert H.max() - H.min() < 1e-9, f"uncoupled run varies by {H.max() - H.min():.2e}"


def test_dissipative_term_is_the_only_energy_sink():
    """dH/dt = -γ·p·(∂H/∂p): the R-channel is the sole non-conservative term.

    This is the port-Hamiltonian split stated as an identity. The conservative
    part contributes exactly zero to the energy rate (verified separately in
    `test_conservative_limit_recovers_conservation`), so whatever energy the
    model loses or gains comes only from the γ term.

    Note what this does NOT assert: that dH/dt <= 0 pointwise at arbitrary
    parameters. Passivity requires p and ∂H/∂p to share a sign, which a *trained*
    model satisfies over its operating region but an untrained circuit does not.
    Passivity is therefore a fitted property here, not a structural guarantee —
    the manuscript reports it as a measured metric, and the trainer carries an
    explicit passivity penalty (`compute_loss_with_passivity`) for exactly this
    reason.
    """
    model = VectorFieldQpHNN(n_layers=1, seed=3)
    params = model.init_params()
    theta = params[: model.n_circuit_weights]
    s = float(params[-2])
    gamma = 0.3
    h = 1e-5

    worst = 0.0
    for q in np.linspace(-1.5, 1.5, 9):
        for p in np.linspace(-1.5, 1.5, 9):
            qdot = model.q_dot(q, p, theta, s=s)
            pdot = model.p_dot(q, p, theta, gamma, s=s)
            dH_dq = (model.energy(q + h, p, theta, s=s) - model.energy(q - h, p, theta, s=s)) / (2 * h)
            dH_dp = (model.energy(q, p + h, theta, s=s) - model.energy(q, p - h, theta, s=s)) / (2 * h)

            rate = dH_dq * qdot + dH_dp * pdot
            predicted = -gamma * p * dH_dp          # the identity
            worst = max(worst, abs(rate - predicted))

    assert worst < 1e-6, f"energy rate departs from -γ·p·∂H/∂p by {worst:.2e}"


def test_passivity_holds_where_the_damping_opposes_momentum():
    """Where p and ∂H/∂p share a sign, the flow is strictly passive."""
    model = VectorFieldQpHNN(n_layers=1, seed=3)
    params = model.init_params()
    theta = params[: model.n_circuit_weights]
    s = float(params[-2])
    gamma = 0.3
    h = 1e-5

    checked = 0
    for q in np.linspace(-1.5, 1.5, 9):
        for p in np.linspace(-1.5, 1.5, 9):
            dH_dp = (model.energy(q, p + h, theta, s=s) - model.energy(q, p - h, theta, s=s)) / (2 * h)
            if p * dH_dp <= 0:
                continue                              # outside the passive region
            qdot = model.q_dot(q, p, theta, s=s)
            pdot = model.p_dot(q, p, theta, gamma, s=s)
            dH_dq = (model.energy(q + h, p, theta, s=s) - model.energy(q - h, p, theta, s=s)) / (2 * h)
            assert dH_dq * qdot + dH_dp * pdot <= 1e-9
            checked += 1

    assert checked >= 15, f"only {checked} passive-region points sampled"


def test_conservative_limit_recovers_conservation():
    """With the R-channel off, the dissipative model conserves energy exactly.

    The two halves of the port-Hamiltonian split must be separable: switching off
    dissipation leaves the skew interconnection untouched, and dH/dt returns to
    zero to machine precision.
    """
    model = VectorFieldQpHNN(n_layers=1, seed=3)
    params = model.init_params()
    theta = params[: model.n_circuit_weights]
    s = float(params[-2])
    h = 1e-5

    worst = 0.0
    for q in np.linspace(-1.2, 1.2, 7):
        for p in np.linspace(-1.2, 1.2, 7):
            qdot = model.q_dot(q, p, theta, s=s)
            pdot = model.p_dot_conservative(q, p, theta, s=s)     # R-channel off
            dH_dq = (model.energy(q + h, p, theta, s=s) - model.energy(q - h, p, theta, s=s)) / (2 * h)
            dH_dp = (model.energy(q, p + h, theta, s=s) - model.energy(q, p - h, theta, s=s)) / (2 * h)
            worst = max(worst, abs(dH_dq * qdot + dH_dp * pdot))

    assert worst < 1e-6, f"conservative limit leaks energy: max |dH/dt| = {worst:.2e}"


def test_reference_damped_oscillator_loses_energy():
    """The ground-truth damped system dissipates, as the model must learn to."""
    sysm = DampedHarmonicOscillator(gamma=0.3)
    traj = sysm.integrate(1.5, 0.0, dt=0.01, n_steps=800)
    H = sysm.hamiltonian(traj.q, traj.p)

    assert H[-1] < H[0], "reference oscillator does not lose energy"
    assert compute_energy_monotone_fraction(H) > 0.95
