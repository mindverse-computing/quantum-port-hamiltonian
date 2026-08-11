"""
experiments/ibm_hardware/run_gradients.py
=========================================
Plan steps 6-7: symplectic gradients on IBM hardware, and the accuracy of the
resulting vector field against the exact statevector field.

Usage
-----
    python experiments/ibm_hardware/run_gradients.py --mode fake      # dry run, no QPU
    python experiments/ibm_hardware/run_gradients.py --mode hardware --shots 1024

The dry run writes the exact JSON the hardware run will write, so the schema is
settled before any QPU time is spent.

Evaluation set
--------------
The manuscript's validation split, reproduced bit-for-bit from the training
script's config (``experiments/qhnn_pendulum/run.py``):
NonlinearPendulum(q_range=π/2, p_range=1) → generate(n_samples=200, seed=42)
→ train_test_split(test_fraction=0.20, seed=42) → 40 validation points.
``--n-points`` truncates that split (leading indices, order untouched) when the
QPU budget will not carry all 40.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]      # -> codes/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ibm.hardware_harness import (                                 # noqa: E402
    RunRecord, calibration_snapshot, job_cost, record_cost,
    success_proxy, transpile_report, used_qubits, budget_remaining,
)
from ibm.gradients_hw import (                                     # noqa: E402
    SHIFT, assemble_gradients, component_errors, param_array, shifted_points,
)
from non_dissipative import NonlinearPendulum, QuantumHNN          # noqa: E402

# Manuscript-trained parameters (report/qhnn_pendulum/report.md, main.tex §7).
THETA_STAR = np.array([1.723, -1.582, 1.170, 1.594])
S_STAR = 1.335
B_STAR = 0.0
PHI_STAR = np.concatenate([THETA_STAR, [S_STAR, B_STAR]])


def validation_split(n_points: int | None = None):
    """The manuscript's 40-point validation split (optionally truncated)."""
    pend = NonlinearPendulum(q_range=np.pi / 2, p_range=1.0)
    _, val = pend.generate(n_samples=200, seed=42).train_test_split(
        test_fraction=0.20, seed=42)
    q, p, qd, pd = val.q, val.p, val.q_dot, val.p_dot
    if n_points is not None:
        q, p, qd, pd = q[:n_points], p[:n_points], qd[:n_points], pd[:n_points]
    return q, p, qd, pd


def statevector_sweep(circuit, obs, values) -> np.ndarray:
    """Noiseless ⟨ZZ⟩ for a parameter-value array -- the exact reference."""
    from qiskit.primitives import StatevectorEstimator
    res = StatevectorEstimator().run([(circuit, obs, values)]).result()
    return np.asarray(res[0].data.evs, float).ravel()


def verify_convention(model, q, p) -> float:
    """
    Assert the array-bound shift rule reproduces ``QuantumHNN.q_dot``/``p_dot``.

    A silent parameter-ordering slip would look like a hardware error, so the
    binding is checked against the model's own methods before submission.
    """
    qs, ps = shifted_points(q, p)
    vals = param_array(model.circuit, qs, ps, THETA_STAR)
    evs = statevector_sweep(model.circuit, model._obs, vals)
    g = assemble_gradients(evs, [0.0] * evs.size, S_STAR)
    ref_q = np.array([model.q_dot(q[i], p[i], PHI_STAR) for i in range(q.size)])
    ref_p = np.array([model.p_dot(q[i], p[i], PHI_STAR) for i in range(q.size)])
    return float(max(np.max(np.abs(g["q_dot"] - ref_q)),
                     np.max(np.abs(g["p_dot"] - ref_p))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fake", "hardware"], default="fake")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--n-points", type=int, default=40)
    ap.add_argument("--resilience", type=int, default=1)
    ap.add_argument("--name", default="qhnn_gradients_hw")
    ap.add_argument("--max-qpu-seconds", type=float, default=9.0,
                    help="cancel the job if IBM's pre-execution estimate exceeds this")
    args = ap.parse_args()

    t0 = time.time()
    q, p, qd_true, pd_true = validation_split(args.n_points)
    model = QuantumHNN(n_layers=1, seed=42)

    dev = verify_convention(model, q, p)
    assert dev < 1e-9, f"shift-rule binding disagrees with QuantumHNN by {dev:g}"

    qs, ps = shifted_points(q, p)
    logical_vals = param_array(model.circuit, qs, ps, THETA_STAR)
    exact_evs = statevector_sweep(model.circuit, model._obs, logical_vals)
    exact = assemble_gradients(exact_evs, [0.0] * exact_evs.size, S_STAR)

    # ---- backend ------------------------------------------------------------
    if args.mode == "fake":
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        backend = FakeMarrakesh()
    else:
        backend = _hardware_backend()
    print(f"[backend] {backend.name}  mode={args.mode}", flush=True)

    # ---- transpile ONCE, bind after ----------------------------------------
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                      seed_transpiler=42)
    isa = pm.run(model.circuit)
    obs_isa = model._obs.apply_layout(isa.layout)
    isa_vals = param_array(isa, qs, ps, THETA_STAR)

    treport = transpile_report(model.circuit, isa)
    treport["n_shifted_evaluations"] = int(isa_vals.shape[0])
    treport["n_pubs"] = 1
    treport["binding"] = "single transpiled ISA circuit, parameter array"
    snap = calibration_snapshot(backend, used_qubits(isa))

    # ---- submit -------------------------------------------------------------
    from qiskit_ibm_runtime import EstimatorV2
    est = EstimatorV2(mode=backend)
    est.options.default_shots = args.shots
    est.options.resilience_level = args.resilience
    est.options.dynamical_decoupling.enable = True
    est.options.twirling.enable_gates = True
    est.options.twirling.enable_measure = True

    job = est.run([(isa, obs_isa, isa_vals)])
    job_id = job.job_id()
    print(f"[job] {job_id}", flush=True)

    est_qpu = None
    try:
        est_qpu = float(job.usage_estimation.get("quantum_seconds"))
        print(f"[estimate] quantum_seconds={est_qpu:.2f}", flush=True)
    except Exception as e:                                          # noqa: BLE001
        print(f"[estimate] unavailable ({type(e).__name__})", flush=True)
    if args.mode == "hardware" and est_qpu is not None and est_qpu > args.max_qpu_seconds:
        job.cancel()
        print(f"[ABORT] estimate {est_qpu:.2f}s exceeds cap {args.max_qpu_seconds}s "
              f"-- job cancelled before execution", flush=True)
        return 2

    result = job.result()
    evs = np.asarray(result[0].data.evs, float).ravel()
    try:
        stds = np.asarray(result[0].data.stds, float).ravel().tolist()
    except Exception:                                               # noqa: BLE001
        stds = [None] * evs.size

    hw = assemble_gradients(evs, stds, S_STAR)

    # ---- error analysis -----------------------------------------------------
    err = {
        "q_dot_vs_exact": component_errors(hw["q_dot"], exact["q_dot"], hw["q_dot_sigma"]),
        "p_dot_vs_exact": component_errors(hw["p_dot"], exact["p_dot"], hw["p_dot_sigma"]),
        "q_dot_vs_truth": component_errors(hw["q_dot"], qd_true, hw["q_dot_sigma"]),
        "p_dot_vs_truth": component_errors(hw["p_dot"], pd_true, hw["p_dot_sigma"]),
        "exact_vs_truth": {
            "q_dot_mse": float(np.mean((exact["q_dot"] - qd_true) ** 2)),
            "p_dot_mse": float(np.mean((exact["p_dot"] - pd_true) ** 2)),
        },
        "raw_zz_vs_exact": {
            "mae": float(np.mean(np.abs(evs - exact_evs))),
            "rmse": float(np.sqrt(np.mean((evs - exact_evs) ** 2))),
            "bias": float(np.mean(evs - exact_evs)),
            "contraction_slope": float(np.polyfit(exact_evs, evs, 1)[0]),
        },
    }

    rec = RunRecord(
        name=args.name, backend=backend.name, mode=args.mode,
        shots=args.shots, n_pubs=1, job_id=job_id,
        resilience_level=args.resilience, optimization_level=3,
        dynamical_decoupling=True, twirling=True,
        transpile=treport, calibration=snap,
        success_proxy=success_proxy(snap, treport["isa_2q"]),
    )
    rec.observables = {
        "shifted_zz_evs": evs.tolist(),
        "shifted_zz_stds": [None if s is None else float(s) for s in stds],
        "shift_row_order": "per point i: 4i=(q,p+π/2) 4i+1=(q,p−π/2) "
                           "4i+2=(q+π/2,p) 4i+3=(q−π/2,p)",
        "q_dot_hw": hw["q_dot"].tolist(),
        "p_dot_hw": hw["p_dot"].tolist(),
        "q_dot_sigma": hw["q_dot_sigma"].tolist(),
        "p_dot_sigma": hw["p_dot_sigma"].tolist(),
    }
    rec.reference = {
        "split": "NonlinearPendulum(q_range=pi/2,p_range=1).generate(200,seed=42)"
                 ".train_test_split(0.20,seed=42) -> validation",
        "n_points": int(q.size),
        "n_points_in_full_split": 40,
        "theta_star": THETA_STAR.tolist(), "s_star": S_STAR, "b_star": B_STAR,
        "shift": float(SHIFT),
        "q": q.tolist(), "p": p.tolist(),
        "q_dot_exact_statevector": exact["q_dot"].tolist(),
        "p_dot_exact_statevector": exact["p_dot"].tolist(),
        "shifted_zz_exact": exact_evs.tolist(),
        "q_dot_ground_truth": qd_true.tolist(),
        "p_dot_ground_truth": pd_true.tolist(),
        "binding_check_max_abs_dev_vs_model_methods": dev,
    }
    rec.error_analysis = err
    rec.wall_seconds = time.time() - t0
    rec.qpu_seconds = job_cost(job)
    rec.notes = [
        "Parameter-shift rule applied to the data-encoding gates (q_in, p_in), "
        "convention copied from QuantumHNN.q_dot/p_dot and common/parameter_shift.py.",
        "All 4N shifted evaluations ride one PUB on one transpiled ISA circuit; "
        "the +/- members of each shift pair therefore share gate sequence and layout.",
        "Uncertainties are estimator standard errors combined in quadrature and "
        "scaled by |s|/2, the prefactor the shift rule applies.",
        "mean_rel_err is reported but not interpretable here: the pendulum vector "
        "field crosses zero inside the validation domain.",
    ]
    if est_qpu is not None:
        rec.notes.append(f"IBM pre-execution estimate: {est_qpu:.2f} quantum_seconds.")
    if args.mode == "hardware" and rec.qpu_seconds is not None:
        record_cost(args.name, job_id, backend.name, rec.qpu_seconds,
                    args.shots, 1, note="symplectic gradients, plan steps 6-7")

    path = rec.save(f"ibm_{args.name}.json")
    print(f"[saved] {path}", flush=True)
    print(json.dumps({
        "job_id": job_id,
        "qpu_seconds": rec.qpu_seconds,
        "budget_remaining": budget_remaining() if args.mode == "hardware" else None,
        "q_dot": {k: err["q_dot_vs_exact"][k] for k in
                  ("mae", "rmse", "bias", "median_rel_err", "rms_pull",
                   "frac_within_1sigma", "frac_within_2sigma", "median_sigma")},
        "p_dot": {k: err["p_dot_vs_exact"][k] for k in
                  ("mae", "rmse", "bias", "median_rel_err", "rms_pull",
                   "frac_within_1sigma", "frac_within_2sigma", "median_sigma")},
        "raw_zz": err["raw_zz_vs_exact"],
    }, indent=2))
    return 0


def _hardware_backend():
    """Least-busy operational QPU, credentials via the host SDK (never printed)."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    tok = _cred("IBM_QUANTUM_TOKEN")
    inst = _cred("IBM_QUANTUM_INSTANCE")
    service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                   token=tok, instance=inst)
    return service.least_busy(operational=True, simulator=False, min_num_qubits=2)


def _cred(name: str) -> str:
    import builtins
    h = getattr(builtins, "host", None)
    if h is None:
        raise RuntimeError("host SDK unavailable; run this module from the "
                           "python tool where `host` is injected")
    return h.credentials.get(name)["value"].strip()


if __name__ == "__main__":
    raise SystemExit(main())
