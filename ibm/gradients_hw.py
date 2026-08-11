"""
ibm/gradients_hw.py
===================
Symplectic gradients (Hamilton's equations) on IBM hardware.

The Q-HNN's central claim is not that a 2-qubit circuit can output a number
resembling an energy -- it is that the *parameter-shift rule applied to the
data-encoding gates* returns the Hamiltonian vector field:

    q̇ = ∂H/∂p  =  s · ½[⟨ZZ⟩(q, p+π/2) − ⟨ZZ⟩(q, p−π/2)]
    ṗ = −∂H/∂q = −s · ½[⟨ZZ⟩(q+π/2, p) − ⟨ZZ⟩(q−π/2, p)]

(convention taken verbatim from ``non_dissipative.QuantumHNN.q_dot`` /
``.p_dot`` and ``common.parameter_shift.DataEncodingShift``; nothing is
reinvented here.)

Two design choices matter for the hardware run:

1. **One PUB, not N.**  Every shifted evaluation is the *same* circuit with
   different parameter values, so the whole sweep ships as a single PUB
   carrying a ``(4N, n_params)`` parameter array.  Per-PUB overhead dominates
   the QPU charge at this circuit size; this collapses it to one.

2. **Transpile once, bind after.**  The parameterised circuit is transpiled a
   single time and the bound values are supplied to the ISA circuit.  If each
   of the 4N angle sets were transpiled independently, ``optimization_level=3``
   would fold particular angles away and the 4N circuits would no longer be the
   same experiment -- the shift-rule difference would then mix a genuine
   gradient with a transpiler artefact.  One ISA circuit means the ± members of
   each shift pair see literally the same gate sequence and the same layout,
   so coherent errors partially cancel in the difference.

Shot noise is propagated the way the shift rule composes it: each expectation
carries an estimator standard error, the difference adds them in quadrature,
and the ½·s prefactor scales the result.
"""
from __future__ import annotations

import numpy as np

SHIFT = np.pi / 2.0


# ---- shift-rule bookkeeping -------------------------------------------------

def shifted_points(q: np.ndarray, p: np.ndarray, shift: float = SHIFT):
    """
    Expand N phase-space points into the 4N shifted evaluations the rule needs.

    Row order per point i is fixed and relied on downstream:
        4i+0 : (q,        p+shift)   ->  q̇ plus
        4i+1 : (q,        p−shift)   ->  q̇ minus
        4i+2 : (q+shift,  p      )   ->  ṗ plus
        4i+3 : (q−shift,  p      )   ->  ṗ minus

    This is the same ordering ``QuantumHNN.compute_loss`` uses for its batched
    PSR, so hardware and simulator PUB lists are index-compatible.
    """
    q = np.asarray(q, float)
    p = np.asarray(p, float)
    qs = np.empty(4 * q.size)
    ps = np.empty(4 * q.size)
    for i in range(q.size):
        qs[4 * i + 0], ps[4 * i + 0] = q[i], p[i] + shift
        qs[4 * i + 1], ps[4 * i + 1] = q[i], p[i] - shift
        qs[4 * i + 2], ps[4 * i + 2] = q[i] + shift, p[i]
        qs[4 * i + 3], ps[4 * i + 3] = q[i] - shift, p[i]
    return qs, ps


def param_array(circuit, q_vals, p_vals, theta) -> np.ndarray:
    """
    Build the ``(M, n_params)`` value array in *this circuit's* parameter order.

    Qiskit orders ``circuit.parameters`` by name, and transpilation preserves
    the parameter set, so the order is read off the circuit rather than
    assumed.
    """
    names = [pp.name for pp in circuit.parameters]
    theta = np.asarray(theta, float)
    rows = []
    for qv, pv in zip(np.asarray(q_vals, float), np.asarray(p_vals, float)):
        lookup = {"q_in": float(qv), "p_in": float(pv)}
        for i in range(theta.size):
            lookup[f"θ[{i}]"] = float(theta[i])
        missing = [n for n in names if n not in lookup]
        if missing:
            raise KeyError(f"circuit has unbound parameters not supplied: {missing}")
        rows.append([lookup[n] for n in names])
    return np.asarray(rows, float)


def assemble_gradients(evs, stds, s: float) -> dict:
    """
    Fold a length-4N expectation sweep into (q̇, ṗ) with propagated 1σ bars.

        q̇_i = + s·½(ev[4i]   − ev[4i+1])
        ṗ_i = − s·½(ev[4i+2] − ev[4i+3])
        σ    = |s|·½·sqrt(σ₊² + σ₋²)

    ``stds`` entries may be None (a backend that reports no standard error);
    the corresponding uncertainty comes back as NaN rather than a fabricated 0.
    """
    evs = np.asarray(evs, float)
    n = evs.size // 4
    sd = np.array([np.nan if x is None else float(x) for x in stds], float)

    q_dot = s * 0.5 * (evs[0::4] - evs[1::4])
    p_dot = -s * 0.5 * (evs[2::4] - evs[3::4])
    q_dot_sigma = abs(s) * 0.5 * np.sqrt(sd[0::4] ** 2 + sd[1::4] ** 2)
    p_dot_sigma = abs(s) * 0.5 * np.sqrt(sd[2::4] ** 2 + sd[3::4] ** 2)
    return {
        "n_points": int(n),
        "q_dot": q_dot,
        "p_dot": p_dot,
        "q_dot_sigma": q_dot_sigma,
        "p_dot_sigma": p_dot_sigma,
    }


def component_errors(hw: np.ndarray, exact: np.ndarray, sigma: np.ndarray) -> dict:
    """
    Per-component deviation of a hardware vector-field component from exact.

    ``mean_rel_err`` is reported but is not the headline: the pendulum vector
    field passes through zero inside the validation domain, so the relative
    error diverges there.  Median relative error, MAE and RMSE are the
    interpretable summaries.

    ``frac_within_1sigma`` / ``frac_within_2sigma`` answer the actual question:
    is the hardware field consistent with exact *given shot noise*, or is there
    a systematic bias on top?  ``mean_pull`` is the signed bias in units of the
    shot-noise bar; ``rms_pull`` near 1 means shot-noise-limited, ≫1 means a
    coherent/device error dominates.
    """
    hw = np.asarray(hw, float)
    ex = np.asarray(exact, float)
    sg = np.asarray(sigma, float)
    d = hw - ex
    denom = np.maximum(np.abs(ex), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        pull = d / sg
    good = np.isfinite(pull)
    return {
        "n": int(hw.size),
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "max_abs_err": float(np.max(np.abs(d))),
        "bias": float(np.mean(d)),
        "median_rel_err": float(np.median(np.abs(d) / denom)),
        "mean_rel_err": float(np.mean(np.abs(d) / denom)),
        "pearson_r": float(np.corrcoef(hw, ex)[0, 1]) if hw.size > 1 else None,
        "median_sigma": float(np.nanmedian(sg)),
        "mean_pull": float(np.mean(pull[good])) if good.any() else None,
        "rms_pull": float(np.sqrt(np.mean(pull[good] ** 2))) if good.any() else None,
        "frac_within_1sigma": float(np.mean(np.abs(pull[good]) <= 1.0)) if good.any() else None,
        "frac_within_2sigma": float(np.mean(np.abs(pull[good]) <= 2.0)) if good.any() else None,
    }
