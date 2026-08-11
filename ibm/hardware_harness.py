"""
ibm/hardware_harness.py
=======================
Submission harness for the Q-pHNN hardware campaign.

Three jobs, in order of importance:

1. **Batch.** At this circuit size the per-job overhead dominates the QPU
   charge, so every experiment submits *one* job holding many PUBs rather than
   many small jobs. A 40-point energy sweep costs about what a single point
   costs.
2. **Snapshot the device.** Heron calibration drifts hour to hour, so a result
   without the calibration state that produced it is not reproducible. Every
   run records T1/T2, readout error, and per-CZ error for the physical qubits
   actually used, read at submission time.
3. **Account for cost.** The Open plan grants 600 s of QPU time per month.
   Every run records ``qpu_charge_time_seconds`` and appends to a ledger, so
   the remaining budget is always known before the next submission.

Results land in ``codes/results/ibm_<name>.json``. That file is the single
source of truth for every number quoted in the manuscript — see
``check_numbers.py``.

Nothing here prints a credential.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LEDGER = RESULTS_DIR / "ibm_qpu_ledger.json"

# IBM Open plan monthly allowance, seconds of QPU time.
OPEN_PLAN_MONTHLY_SECONDS = 600


# ---- device snapshot --------------------------------------------------------

def calibration_snapshot(backend, qubits: list[int]) -> dict:
    """
    T1/T2, readout error and per-CZ error for *qubits*, read now.

    Heron calibration moves hour to hour; a measurement is only interpretable
    against the calibration that produced it. Missing entries come back as
    ``None`` rather than raising, since not every property exists on every
    backend generation.
    """
    snap: dict = {
        "backend": backend.name,
        "num_qubits": backend.num_qubits,
        "basis_gates": sorted(backend.operation_names),
        "read_at": datetime.now(timezone.utc).isoformat(),
        "qubits": {},
        "two_qubit": {},
    }
    try:
        props = backend.properties()
    except Exception:                                          # noqa: BLE001
        props = None
    if props is None:
        snap["note"] = "backend exposes no properties() (simulator or fake backend)"
        return snap

    for q in qubits:
        entry = {}
        for key, fn in (("t1_us", props.t1), ("t2_us", props.t2)):
            try:
                entry[key] = float(fn(q)) * 1e6
            except Exception:                                  # noqa: BLE001
                entry[key] = None
        try:
            entry["readout_error"] = float(props.readout_error(q))
        except Exception:                                      # noqa: BLE001
            entry["readout_error"] = None
        try:
            entry["frequency_ghz"] = float(props.frequency(q)) / 1e9
        except Exception:                                      # noqa: BLE001
            entry["frequency_ghz"] = None
        snap["qubits"][str(q)] = entry

    # per-pair two-qubit error, for the pairs among `qubits` the device couples
    cmap = getattr(backend, "coupling_map", None)
    pairs = []
    if cmap is not None:
        qs = set(qubits)
        pairs = [tuple(e) for e in cmap.get_edges() if e[0] in qs and e[1] in qs]
    twoq_names = [g for g in ("cz", "ecr", "cx") if g in backend.operation_names]
    for a, b in pairs:
        for gate in twoq_names:
            try:
                snap["two_qubit"][f"{gate}_{a}_{b}"] = float(props.gate_error(gate, [a, b]))
            except Exception:                                  # noqa: BLE001
                continue
    return snap


def used_qubits(isa_circuit) -> list[int]:
    """Physical qubits an ISA circuit actually touches."""
    idx = set()
    for inst in isa_circuit.data:
        for q in inst.qubits:
            idx.add(isa_circuit.find_bit(q).index)
    return sorted(idx)


def transpile_report(logical, isa) -> dict:
    """Depth and gate counts before/after transpilation -- the error budget."""
    def _twoq(c):
        return sum(1 for i in c.data if i.operation.num_qubits == 2)
    return {
        "logical_depth": int(logical.depth()),
        "logical_2q": int(_twoq(logical)),
        "isa_depth": int(isa.depth()),
        "isa_2q": int(_twoq(isa)),
        "isa_size": int(isa.size()),
        "physical_qubits": used_qubits(isa),
        "n_physical_qubits": len(used_qubits(isa)),
    }


def success_proxy(snapshot: dict, isa_2q: int) -> float | None:
    """
    Crude circuit-success proxy: (1 - median 2q error) ** (2q gate count).

    A budget indicator, not a fidelity prediction -- it ignores idle error,
    crosstalk and readout, and assumes every 2q gate carries the median error.
    Returns None when the backend exposes no 2q error data.
    """
    errs = [v for v in snapshot.get("two_qubit", {}).values() if v is not None]
    if not errs or isa_2q == 0:
        return None
    return float((1.0 - float(np.median(errs))) ** isa_2q)


# ---- cost ledger ------------------------------------------------------------

def _read_ledger() -> dict:
    if LEDGER.is_file():
        return json.loads(LEDGER.read_text())
    return {"plan": "open", "monthly_seconds": OPEN_PLAN_MONTHLY_SECONDS, "jobs": []}


def record_cost(name: str, job_id: str, backend: str, seconds: float,
                shots: int, n_pubs: int, note: str = "") -> dict:
    """Append one job to the QPU ledger and return the running total."""
    led = _read_ledger()
    led["jobs"].append({
        "name": name, "job_id": job_id, "backend": backend,
        "qpu_seconds": seconds, "shots": shots, "n_pubs": n_pubs,
        "submitted": datetime.now(timezone.utc).isoformat(), "note": note,
    })
    spent = sum(j["qpu_seconds"] or 0 for j in led["jobs"])
    led["spent_seconds"] = spent
    led["remaining_seconds"] = led["monthly_seconds"] - spent
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=2))
    return led


def budget_remaining() -> float:
    led = _read_ledger()
    return led.get("remaining_seconds",
                   led["monthly_seconds"] - sum(j["qpu_seconds"] or 0 for j in led["jobs"]))


def job_cost(job) -> float | None:
    """QPU seconds IBM charged for *job*, or None if not yet reported."""
    try:
        return float(job.metrics().get("usage", {}).get("qpu_charge_time_seconds"))
    except Exception:                                          # noqa: BLE001
        return None


# ---- result envelope --------------------------------------------------------

@dataclass
class RunRecord:
    """
    One hardware experiment, serialised whole.

    Every manuscript number must be readable out of this record; anything
    quoted but absent here is untraceable by construction.
    """
    name: str
    backend: str
    mode: str                       # "hardware" | "fake" | "statevector"
    shots: int
    n_pubs: int
    job_id: str | None = None
    qpu_seconds: float | None = None
    resilience_level: int | None = None
    optimization_level: int | None = None
    dynamical_decoupling: bool | None = None
    twirling: bool | None = None
    mitigation: dict | None = None
    transpile: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    success_proxy: float | None = None
    observables: dict = field(default_factory=dict)
    reference: dict = field(default_factory=dict)
    error_analysis: dict = field(default_factory=dict)
    wall_seconds: float | None = None
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: list = field(default_factory=list)

    def save(self, filename: str | None = None) -> Path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / (filename or f"ibm_{self.name}.json")
        path.write_text(json.dumps(asdict(self), indent=2, default=_jsonable))
        return path


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---- estimator submission ---------------------------------------------------

def run_estimator(backend, circuits, observables, *, name: str,
                  shots: int = 4096, optimization_level: int = 3,
                  resilience_level: int = 1, dynamical_decoupling: bool = True,
                  twirling: bool = True, mode: str = "hardware",
                  dd_sequence: str | None = None,
                  num_randomizations: int | None = None,
                  twirling_strategy: str | None = None,
                  zne_noise_factors: list | None = None,
                  zne_amplifier: str | None = None,
                  measure_mitigation: bool | None = None,
                  wait: bool = True) -> tuple[RunRecord, object]:
    """
    Transpile and submit *many* (circuit, observable) pairs as ONE job.

    ``circuits`` and ``observables`` are equal-length sequences; each pair
    becomes one PUB. Batching is the whole point -- per-job overhead, not
    per-PUB work, dominates the charge for circuits this small.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2

    assert len(circuits) == len(observables), "one observable per circuit"
    t0 = time.time()
    pm = generate_preset_pass_manager(optimization_level=optimization_level,
                                      backend=backend)
    isas = [pm.run(c) for c in circuits]
    pubs = [(isa, ob.apply_layout(isa.layout)) for isa, ob in zip(isas, observables)]

    qubits = sorted({q for isa in isas for q in used_qubits(isa)})
    snap = calibration_snapshot(backend, qubits)
    treport = transpile_report(circuits[0], isas[0])
    treport["max_isa_depth"] = max(int(i.depth()) for i in isas)
    treport["total_isa_2q"] = sum(
        sum(1 for x in i.data if x.operation.num_qubits == 2) for i in isas)

    est = EstimatorV2(mode=backend)
    opts = est.options
    # Record what was actually APPLIED, not what was requested. A silently
    # swallowed option would otherwise let a record claim mitigation the job
    # never received -- the provenance failure this guard exists to prevent.
    mitigation: dict = {"requested": {}, "applied": {}, "failed": {}}

    def _set(path: str, value):
        mitigation["requested"][path] = value
        obj = opts
        parts = path.split(".")
        try:
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], value)
            mitigation["applied"][path] = getattr(obj, parts[-1])
        except Exception as exc:                               # noqa: BLE001
            mitigation["failed"][path] = f"{type(exc).__name__}: {exc}"

    _set("default_shots", shots)
    _set("resilience_level", resilience_level)
    _set("dynamical_decoupling.enable", bool(dynamical_decoupling))
    if dd_sequence is not None:
        _set("dynamical_decoupling.sequence_type", dd_sequence)
    _set("twirling.enable_gates", bool(twirling))
    _set("twirling.enable_measure", bool(twirling))
    if num_randomizations is not None:
        _set("twirling.num_randomizations", num_randomizations)
    if twirling_strategy is not None:
        _set("twirling.strategy", twirling_strategy)
    if zne_noise_factors is not None:
        _set("resilience.zne_mitigation", True)
        _set("resilience.zne.noise_factors", list(zne_noise_factors))
    if zne_amplifier is not None:
        _set("resilience.zne.amplifier", zne_amplifier)
    if measure_mitigation is not None:
        _set("resilience.measure_mitigation", bool(measure_mitigation))

    if mitigation["failed"]:
        raise RuntimeError(
            "Estimator options rejected -- refusing to spend QPU on a run whose "
            f"mitigation provenance would be wrong: {mitigation['failed']}")

    job = est.run(pubs)
    rec = RunRecord(
        name=name, backend=getattr(backend, "name", str(backend)), mode=mode,
        shots=shots, n_pubs=len(pubs), job_id=getattr(job, "job_id", lambda: None)(),
        resilience_level=resilience_level, optimization_level=optimization_level,
        dynamical_decoupling=dynamical_decoupling, twirling=twirling,
        mitigation=mitigation,
        transpile=treport, calibration=snap,
        success_proxy=success_proxy(snap, treport["isa_2q"]),
    )
    if not wait:
        return rec, job

    result = job.result()
    evs = [float(np.asarray(result[i].data.evs).ravel()[0]) for i in range(len(pubs))]
    stds = []
    for i in range(len(pubs)):
        try:
            stds.append(float(np.asarray(result[i].data.stds).ravel()[0]))
        except Exception:                                      # noqa: BLE001
            stds.append(None)
    rec.observables = {"evs": evs, "stds": stds}
    rec.wall_seconds = time.time() - t0
    rec.qpu_seconds = job_cost(job)
    if mode == "hardware" and rec.qpu_seconds is not None:
        record_cost(name, rec.job_id, rec.backend, rec.qpu_seconds, shots, len(pubs))
    return rec, job


def run_sampler(backend, circuits, *, name: str, shots: int = 4096,
                optimization_level: int = 1, mode: str = "hardware",
                dynamical_decoupling: bool = True, twirling: bool = True,
                wait: bool = True) -> tuple[RunRecord, object]:
    """Same contract as :func:`run_estimator`, for count-based circuits."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    t0 = time.time()
    pm = generate_preset_pass_manager(optimization_level=optimization_level,
                                      backend=backend)
    isas = [pm.run(c) for c in circuits]
    qubits = sorted({q for isa in isas for q in used_qubits(isa)})
    snap = calibration_snapshot(backend, qubits)
    treport = transpile_report(circuits[0], isas[0])
    treport["max_isa_depth"] = max(int(i.depth()) for i in isas)

    sampler = SamplerV2(mode=backend)
    try:
        sampler.options.default_shots = shots
        sampler.options.dynamical_decoupling.enable = bool(dynamical_decoupling)
        sampler.options.twirling.enable_measure = bool(twirling)
    except Exception:                                          # noqa: BLE001
        pass

    job = sampler.run(isas, shots=shots)
    rec = RunRecord(
        name=name, backend=getattr(backend, "name", str(backend)), mode=mode,
        shots=shots, n_pubs=len(isas), job_id=getattr(job, "job_id", lambda: None)(),
        optimization_level=optimization_level, transpile=treport, calibration=snap,
        dynamical_decoupling=dynamical_decoupling, twirling=twirling,
        success_proxy=success_proxy(snap, treport["isa_2q"]),
    )
    if not wait:
        return rec, job

    result = job.result()
    counts = []
    for i in range(len(isas)):
        data = result[i].data
        creg = next(iter(data.__dict__.values())) if hasattr(data, "__dict__") else None
        counts.append(creg.get_counts() if creg is not None else {})
    rec.observables = {"counts": counts}
    rec.wall_seconds = time.time() - t0
    rec.qpu_seconds = job_cost(job)
    if mode == "hardware" and rec.qpu_seconds is not None:
        record_cost(name, rec.job_id, rec.backend, rec.qpu_seconds, shots, len(isas))
    return rec, job


# ---- offline references -----------------------------------------------------

def exact_expectations(circuits, observables) -> list[float]:
    """Ideal <O> via local statevector -- the reference every run is scored against."""
    from qiskit.primitives import StatevectorEstimator
    est = StatevectorEstimator()
    res = est.run([(c, o) for c, o in zip(circuits, observables)]).result()
    return [float(np.asarray(res[i].data.evs).ravel()[0]) for i in range(len(circuits))]


def error_metrics(hardware: list[float], exact: list[float]) -> dict:
    """Deviation of a hardware sweep from its exact reference."""
    h, e = np.asarray(hardware, float), np.asarray(exact, float)
    d = h - e
    denom = np.maximum(np.abs(e), 1e-12)
    return {
        "n": int(h.size),
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "max_abs_err": float(np.max(np.abs(d))),
        "mean_rel_err": float(np.mean(np.abs(d) / denom)),
        "median_rel_err": float(np.median(np.abs(d) / denom)),
        "bias": float(np.mean(d)),
        "pearson_r": float(np.corrcoef(h, e)[0, 1]) if h.size > 1 else None,
    }
