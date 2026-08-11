"""
ibm/preflight.py
================
Staged IBM Quantum readiness check. Each stage isolates one failure mode, so a
red line tells you exactly what to fix instead of "something went wrong".

    1. offline   Bell + QGNN energy on the local statevector (no token/network)
    2. noisy     same circuits on a fake IBM device (transpile + primitives)
    3. iam       raw IBM Cloud IAM exchange -- is the API key itself valid?
    4. service   QiskitRuntimeService + backend listing -- is the instance bound?
    5. live      submit the Bell circuit to real hardware (opt-in: --live)

Credentials come from the process environment (IBM_QUANTUM_TOKEN /
IBM_QUANTUM_INSTANCE) or an explicit --dotenv path. Nothing is ever printed.

Run::

    python -m ibm.preflight              # stages 1-4, no hardware time spent
    python -m ibm.preflight --live       # also submit a real Bell job
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

IAM_URL = "https://iam.cloud.ibm.com/identity/token"
_OK, _NO = "PASS", "FAIL"


def _line(stage: str, ok: bool, msg: str = "") -> bool:
    print(f"[{_OK if ok else _NO}] {stage:<9} {msg}")
    return ok


# ---- stage 1: offline -------------------------------------------------------

def stage_offline() -> bool:
    from qiskit.primitives import StatevectorSampler

    from .runner import estimate_energy_local
    from .sample_circuit import bell_circuit
    from network.qgnn_energy import QGNNEnergy

    res = StatevectorSampler().run([(bell_circuit(),)], shots=4096).result()
    counts = res[0].data.c.get_counts()
    frac = (counts.get("00", 0) + counts.get("11", 0)) / max(sum(counts.values()), 1)
    ok = _line("offline", frac > 0.98, f"Bell P(00)+P(11) = {frac:.3f}")

    rng = np.random.default_rng(7)
    worst = 0.0
    for N, L in [(3, 1), (4, 2), (5, 2)]:
        edges = [(i, i + 1) for i in range(N - 1)] + ([(N - 1, 0)] if N > 2 else [])
        m = QGNNEnergy(N, edges, n_layers=L)
        theta = m.init_weights()
        state = rng.normal(size=2 * N)      # state is 2N: (q, p) halves
        worst = max(worst, abs(float(m.energy(state, theta))
                               - float(estimate_energy_local(m, state, theta))))
    return _line("offline", worst < 1e-9,
                 f"QGNN energy vs network reference, worst |diff| = {worst:.2e}") and ok


# ---- stage 2: noisy fake device --------------------------------------------

def stage_noisy() -> bool:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2, SamplerV2
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane

    from .sample_circuit import bell_circuit
    from network.qgnn_energy import QGNNEnergy

    backend = FakeBrisbane()
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa = pm.run(bell_circuit())
    counts = SamplerV2(mode=backend).run([isa], shots=4096).result()[0].data.c.get_counts()
    frac = (counts.get("00", 0) + counts.get("11", 0)) / max(sum(counts.values()), 1)
    ok = _line("noisy", frac > 0.80, f"Bell on {backend.name}: P(00)+P(11) = {frac:.3f}")

    rng = np.random.default_rng(7)
    N, L = 4, 2
    edges = [(i, i + 1) for i in range(N - 1)] + [(N - 1, 0)]
    m = QGNNEnergy(N, edges, n_layers=L)
    theta, state = m.init_weights(), rng.normal(size=2 * N)
    ideal = float(m.energy(state, theta))
    bound = m.circuit.assign_parameters(m._bind_values(state, theta))
    isa = generate_preset_pass_manager(optimization_level=3, backend=backend).run(bound)
    twoq = sum(1 for i in isa.data if i.operation.num_qubits == 2)
    ev = EstimatorV2(mode=backend).run([(isa, m._obs.apply_layout(isa.layout))]).result()
    noisy = float(QGNNEnergy._scale(theta) * float(ev[0].data.evs))
    rel = abs(noisy - ideal) / max(abs(ideal), 1e-12)
    return _line("noisy", np.isfinite(noisy),
                 f"QGNN N={N}: depth={isa.depth()} 2q={twoq} "
                 f"ideal={ideal:+.4f} noisy={noisy:+.4f} rel_err={rel:.1%}") and ok


# ---- credentials ------------------------------------------------------------

def _creds(dotenv_path: str | None):
    from .connection import load_credentials
    try:
        return load_credentials(dotenv_path)
    except Exception as e:                                    # noqa: BLE001
        _line("creds", False, f"{type(e).__name__}: {e}")
        return None


# ---- stage 3: IAM -----------------------------------------------------------

def stage_iam(token: str) -> bool:
    """Ask IBM Cloud IAM directly whether the API key exists. Authoritative."""
    body = urllib.parse.urlencode(
        {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": token}).encode()
    req = urllib.request.Request(
        IAM_URL, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Accept": "application/json"})
    try:
        payload = json.loads(urllib.request.urlopen(req, timeout=30).read())
        return _line("iam", True,
                     f"API key valid (access token expires in {payload.get('expires_in')}s)")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            j = json.loads(e.read())
            detail = f"{j.get('errorCode', '')} {j.get('errorMessage', '')}".strip()
        except Exception:                                     # noqa: BLE001
            pass
        hint = ("  -> the key is not an IBM Cloud API key, was deleted, or belongs "
                "to another account. Mint a fresh one in the IBM Quantum Platform "
                "dashboard (top-right account menu -> API keys)."
                if e.code == 400 else "")
        return _line("iam", False, f"HTTP {e.code} {detail}{hint}")
    except Exception as e:                                    # noqa: BLE001
        return _line("iam", False, f"{type(e).__name__}: {e}")


# ---- stage 4: service -------------------------------------------------------

def stage_service(creds: dict):
    from qiskit_ibm_runtime import QiskitRuntimeService
    kwargs = {"channel": creds["channel"], "token": creds["token"]}
    if creds["instance"]:
        kwargs["instance"] = creds["instance"]
    try:
        service = QiskitRuntimeService(**kwargs)
        backends = service.backends(operational=True, simulator=False)
        _line("service", bool(backends), f"{len(backends)} operational device(s)")
        for b in backends:
            print(f"             {b.name:18s} nq={b.num_qubits:<4d} "
                  f"queue={b.status().pending_jobs}")
        return service if backends else None
    except Exception as e:                                    # noqa: BLE001
        _line("service", False, f"{type(e).__name__}: {str(e)[:200]}")
        return None


# ---- stage 5: live ----------------------------------------------------------

def stage_live(service, backend_name: str | None, shots: int) -> bool:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    from .connection import pick_backend
    from .sample_circuit import bell_circuit, _counts_summary

    backend = pick_backend(service, min_qubits=2, name=backend_name)
    print(f"[ .. ] live      backend = {backend.name}, submitting Bell x{shots} ...")
    isa = generate_preset_pass_manager(optimization_level=1, backend=backend).run(bell_circuit())
    job = SamplerV2(mode=backend).run([isa], shots=shots)
    print(f"[ .. ] live      job id = {job.job_id()} — waiting (queue depth applies) ...")
    counts = job.result()[0].data.c.get_counts()
    frac = (counts.get("00", 0) + counts.get("11", 0)) / max(sum(counts.values()), 1)
    print(f"             counts: {_counts_summary(counts)}")
    return _line("live", frac > 0.80, f"{backend.name}: P(00)+P(11) = {frac:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Staged IBM Quantum readiness check.")
    ap.add_argument("--live", action="store_true", help="submit a real Bell job (uses quota)")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--dotenv", default=None)
    args = ap.parse_args()

    print("=" * 68)
    ok_offline = stage_offline()
    ok_noisy = stage_noisy()

    creds = _creds(args.dotenv)
    if not creds:
        print("=" * 68)
        print("Stopped: no credentials. Set IBM_QUANTUM_TOKEN / IBM_QUANTUM_INSTANCE.")
        raise SystemExit(1)

    inst = creds["instance"] or ""
    _line("creds", True,
          f"token len={len(creds['token'])}, channel={creds['channel']}, "
          f"instance={'CRN' if inst.startswith('crn:') else (inst or 'none')}")

    if not stage_iam(creds["token"]):
        print("=" * 68)
        print("Stopped at IAM: the API key is rejected by IBM Cloud itself, so the")
        print("instance/channel/backend settings cannot be tested yet. Mint a new key.")
        raise SystemExit(2)

    service = stage_service(creds)
    if service and args.live:
        stage_live(service, args.backend, args.shots)

    print("=" * 68)
    print(f"offline={_OK if ok_offline else _NO}  noisy={_OK if ok_noisy else _NO}  "
          f"service={_OK if service else _NO}")


if __name__ == "__main__":
    main()
