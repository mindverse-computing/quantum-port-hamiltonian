"""
ibm/sample_circuit.py
=====================
Minimal end-to-end connectivity test for IBM Quantum: build a 2-qubit Bell
state, transpile it to the target backend's ISA, submit it with SamplerV2, and
report the measured counts. A healthy device returns ~50/50 on ``00`` and
``11`` with a little noise — this is the "does my token/instance/backend work?"
smoke test before spending time on the real Q-pHNN circuits.

Run::

    python -m ibm.sample_circuit                      # least-busy real device
    python -m ibm.sample_circuit --backend ibm_fez    # a named backend
    python -m ibm.sample_circuit --simulator          # local Aer, no token/network

The ``--simulator`` path runs entirely locally (no credentials, no network) so
the pipeline — build → transpile → sample → parse — can be validated offline.
"""
from __future__ import annotations

import argparse

from qiskit import QuantumCircuit


def bell_circuit() -> QuantumCircuit:
    """The canonical 2-qubit Bell pair with measurement."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _counts_summary(counts: dict) -> str:
    total = sum(counts.values()) or 1
    rows = sorted(counts.items(), key=lambda kv: -kv[1])
    return "  ".join(f"{k}:{v} ({100*v/total:.1f}%)" for k, v in rows)


def run_local(shots: int = 4096) -> dict:
    """Local statevector sampling — no IBM account, no network. Sanity path."""
    from qiskit.primitives import StatevectorSampler
    qc = bell_circuit()
    res = StatevectorSampler().run([(qc,)], shots=shots).result()
    counts = res[0].data.c.get_counts()
    return counts


def run_ibm(backend_name: str | None = None, shots: int = 4096,
            dotenv_path: str | None = None) -> dict:
    """
    Submit the Bell circuit to an IBM Quantum backend via SamplerV2.

    Requires a valid token (see ibm/connection.py). Transpiles to the backend
    ISA first — Runtime primitives require ISA circuits.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    from .connection import get_service, pick_backend

    service = get_service(dotenv_path)
    backend = pick_backend(service, min_qubits=2, name=backend_name)
    print(f"[ibm] backend = {backend.name}  ({backend.num_qubits} qubits)")

    qc = bell_circuit()
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa = pm.run(qc)
    print(f"[ibm] transpiled depth = {isa.depth()}  (logical depth {qc.depth()})")

    sampler = SamplerV2(mode=backend)
    job = sampler.run([isa], shots=shots)
    print(f"[ibm] job id = {job.job_id()}  — submitted, waiting for result ...")
    result = job.result()
    counts = result[0].data.c.get_counts()
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="IBM Quantum connectivity smoke test (Bell state).")
    ap.add_argument("--backend", default=None, help="named backend, else least-busy real device")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--simulator", action="store_true",
                    help="run locally (no token / no network) for an offline pipeline check")
    ap.add_argument("--dotenv", default=None, help="explicit path to a .env file")
    args = ap.parse_args()

    if args.simulator:
        print("[local] StatevectorSampler (no IBM account)")
        counts = run_local(args.shots)
    else:
        counts = run_ibm(args.backend, args.shots, args.dotenv)

    print("[result] Bell-state counts:")
    print("   " + _counts_summary(counts))
    # a healthy Bell run concentrates on 00 and 11
    good = counts.get("00", 0) + counts.get("11", 0)
    frac = good / (sum(counts.values()) or 1)
    print(f"[result] P(00)+P(11) = {frac:.3f}  "
          f"({'PASS' if frac > 0.8 else 'noisy/unexpected — check backend'})")


if __name__ == "__main__":
    main()
