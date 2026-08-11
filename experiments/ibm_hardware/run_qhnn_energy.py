"""
experiments/ibm_hardware/run_qhnn_energy.py
===========================================
Q-HNN energy landscape on IBM hardware, plus a three-rung mitigation ladder.

The experiment is a phase-space sweep of the trained 2-qubit Q-HNN energy

    H(q, p) = s * <ZZ>(q, p; theta*) + b

over the manuscript's training box, q in [-1.6, 1.6], p in [-1.4, 1.4].  Each
grid point is one bound circuit; the whole grid ships as PUBs inside a SINGLE
EstimatorV2 job, because per-job overhead -- not per-PUB work -- dominates the
QPU charge at this circuit size.

The same PUB list is then re-run at three distinct mitigation settings so that
the benefit of error mitigation is *measured* on this observable rather than
assumed:

    rung 0  resilience_level=0, DD off, twirling off   (raw device)
    rung 1  resilience_level=1, DD on,  twirling on    (TREX readout mitigation)
    rung 2  resilience_level=2, DD on,  twirling on    (ZNE on top of rung 1)

Every run is scored pointwise against a local statevector reference.  H passes
through zero inside this box, so relative error is reported as a *median* and
the mean relative error is only meaningful with the near-zero caveat attached.

Nothing here prints a credential.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]      # -> codes/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from ibm.hardware_harness import (
    RunRecord, error_metrics, exact_expectations, run_estimator,
)

# Manuscript-trained Q-HNN parameters.
THETA_STAR = [1.723, -1.582, 1.170, 1.594]
SCALE_S = 1.335
OFFSET_B = 0.0

# Training box (manuscript), and grid resolution.
Q_RANGE = (-1.6, 1.6)
P_RANGE = (-1.4, 1.4)
N_Q = 6
N_P = 6

# The mitigation ladder.  Keys are the rung labels used in filenames.
LADDER = {
    "r0_raw":  dict(resilience_level=0, dynamical_decoupling=False, twirling=False),
    "r1_trex": dict(resilience_level=1, dynamical_decoupling=True,  twirling=True),
    "r2_zne":  dict(resilience_level=2, dynamical_decoupling=True,  twirling=True),
}


def build_grid(n_q: int = N_Q, n_p: int = N_P):
    """Return (qs, ps, points) for the phase-space sweep, row-major in q."""
    qs = np.linspace(*Q_RANGE, n_q)
    ps = np.linspace(*P_RANGE, n_p)
    points = [(float(q), float(p)) for q in qs for p in ps]
    return qs, ps, points


def bind_circuits(model, points, theta=THETA_STAR):
    """One bound circuit per phase-space point, at the trained weights."""
    pmap = {p.name: p for p in model.circuit.parameters}
    out = []
    for q, p in points:
        vals = {pmap["q_in"]: q, pmap["p_in"]: p}
        for i in range(model.n_circuit_weights):
            vals[pmap[f"\u03b8[{i}]"]] = theta[i]
        out.append(model.circuit.assign_parameters(vals))
    return out


def energy(zz, s: float = SCALE_S, b: float = OFFSET_B):
    """H = s * <ZZ> + b, elementwise."""
    return (s * np.asarray(zz, float) + b).tolist()


def reference_block(points, circuits, observables):
    """Exact statevector <ZZ> and H at every grid point -- the scoring truth."""
    zz = exact_expectations(circuits, observables)
    return {
        "q_range": list(Q_RANGE), "p_range": list(P_RANGE),
        "n_q": N_Q, "n_p": N_P,
        "grid_q": [q for q, _ in points],
        "grid_p": [p for _, p in points],
        "theta_star": list(THETA_STAR),
        "scale_s": SCALE_S, "offset_b": OFFSET_B,
        "exact_zz": zz,
        "exact_H": energy(zz),
    }


def score(rec: RunRecord, ref: dict) -> RunRecord:
    """Attach the exact reference and the pointwise error analysis to *rec*."""
    zz_hw = rec.observables["evs"]
    h_hw = energy(zz_hw)
    rec.reference = ref
    rec.observables["H"] = h_hw
    rec.error_analysis = {
        "zz": error_metrics(zz_hw, ref["exact_zz"]),
        "H": error_metrics(h_hw, ref["exact_H"]),
        "note": ("H crosses zero inside the training box, so mean_rel_err is "
                 "inflated by near-zero denominators; use median_rel_err, MAE "
                 "and RMSE."),
    }
    return rec


def energy_quantile_subset(ref: dict, k: int = 12) -> list[int]:
    """
    Indices of *k* grid points spaced evenly across the exact energy range.

    The mitigation ladder is priced per PUB on hardware, so rungs 0 and 2 run
    on a subset of the full grid.  Picking the subset by energy quantile --
    rather than by a corner of the box -- keeps the deep well, the flat region
    and the near-zero crossing all represented, which is what the raw-vs-
    mitigated comparison has to span.  Both extremes are always included.
    """
    h = np.asarray(ref["exact_H"], float)
    order = np.argsort(h)
    picks = np.linspace(0, len(order) - 1, k).round().astype(int)
    return sorted(int(order[i]) for i in dict.fromkeys(picks.tolist()))


def subset_reference(ref: dict, idx: list[int]) -> dict:
    """Restrict a reference block to *idx*, recording the parent grid indices."""
    sub = dict(ref)
    for key in ("grid_q", "grid_p", "exact_zz", "exact_H"):
        sub[key] = [ref[key][i] for i in idx]
    sub["subset_of_full_grid"] = True
    sub["subset_indices"] = list(idx)
    sub["n_points"] = len(idx)
    return sub


def run_rung(backend, circuits, observables, ref, rung: str, *,
             shots: int, mode: str, name_prefix: str = "qhnn_energy"):
    """Submit one rung of the ladder as a single job and score it."""
    cfg = LADDER[rung]
    rec, job = run_estimator(
        backend, circuits, observables,
        name=f"{name_prefix}_{rung}", shots=shots, mode=mode, **cfg,
    )
    score(rec, ref)
    rec.notes.append(f"mitigation rung {rung}: {cfg}")
    return rec, job
