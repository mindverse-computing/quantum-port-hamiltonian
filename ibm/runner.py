"""
ibm/runner.py
=============
Run the network Q-GNN-pHNN **energy** circuit on IBM Quantum hardware.

Reuses the existing Qiskit builder ``network.qgnn_energy.QGNNEnergy`` — the same
parameterised ansatz and ``SparsePauliOp`` energy observable the reference and
the CUDA-Q parity gate use — and evaluates the graph energy
``H(x) = s (Σ_i a_i⟨Z_i⟩ + Σ_ij w_ij⟨Z_iZ_j⟩)`` on a real device via the
Runtime ``EstimatorV2`` primitive (expectation values, with error mitigation).

Pipeline (matches ibm/sample_circuit.py, extended for an observable):
    build QGNNEnergy → bind a node state → transpile circuit + observable to the
    backend ISA → EstimatorV2.run → parse ⟨H_obs⟩ → apply the classical scale s.

The transpile step reports the transpiled 2-qubit depth so the device error
budget is visible before submitting (see experiments/IBM-Quantum-plan.md §4).

Offline validation: ``estimate_energy_local`` runs the identical build →
(ideal) estimate → scale path with the local ``StatevectorEstimator`` — no token,
no network — and must agree with ``network``'s own ``QGNNEnergy.energy``.
"""
from __future__ import annotations

import numpy as np

from network.qgnn_energy import QGNNEnergy


# ---- shared: bind a node state into circuit parameter values ----------------

def _bound_circuit(model: QGNNEnergy, state: np.ndarray, theta: np.ndarray):
    """Return a fully-bound (parameter-free) circuit for one node state."""
    values = model._bind_values(state, theta)          # list[float] in circuit-parameter order
    return model.circuit.assign_parameters(values)


def _scale(model: QGNNEnergy, theta: np.ndarray) -> float:
    """The classical read-out scale s appended to theta (last element)."""
    return QGNNEnergy._scale(theta)


# ---- offline reference (no IBM account) -------------------------------------

def estimate_energy_local(model: QGNNEnergy, state: np.ndarray,
                          theta: np.ndarray) -> float:
    """Ideal ⟨H⟩ via the local StatevectorEstimator — the offline check."""
    from qiskit.primitives import StatevectorEstimator
    bound = _bound_circuit(model, state, theta)
    est = StatevectorEstimator()
    raw = est.run([(bound, model._obs)]).result()[0].data.evs
    return float(_scale(model, theta) * float(raw))


# ---- IBM hardware -----------------------------------------------------------

def estimate_energy_ibm(model: QGNNEnergy, state: np.ndarray, theta: np.ndarray,
                        backend_name: str | None = None,
                        optimization_level: int = 3,
                        resilience_level: int = 1,
                        dotenv_path: str | None = None) -> dict:
    """
    Evaluate the graph energy on an IBM backend via EstimatorV2.

    Returns a dict with the scaled energy, the raw expectation, the chosen
    backend, the job id, and the transpiled 2-qubit depth (error-budget proxy).
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2

    from .connection import get_service, pick_backend

    service = get_service(dotenv_path)
    backend = pick_backend(service, min_qubits=model.N, name=backend_name)

    bound = _bound_circuit(model, state, theta)
    pm = generate_preset_pass_manager(optimization_level=optimization_level,
                                      backend=backend)
    isa = pm.run(bound)
    # the observable must be mapped onto the transpiled layout
    isa_obs = model._obs.apply_layout(isa.layout)
    twoq = sum(n for g, n in isa.count_ops().items()
               if g in ("cz", "cx", "ecr", "rzz"))

    estimator = EstimatorV2(mode=backend)
    estimator.options.resilience_level = resilience_level
    job = estimator.run([(isa, isa_obs)])
    raw = float(job.result()[0].data.evs)
    return {
        "energy": float(_scale(model, theta) * raw),
        "raw_expectation": raw,
        "backend": backend.name,
        "job_id": job.job_id(),
        "transpiled_depth": isa.depth(),
        "transpiled_2q_gates": int(twoq),
        "num_qubits": model.N,
    }


# ---- CLI --------------------------------------------------------------------

def _demo_model(N: int, n_layers: int, seed: int):
    """A ring-coupled QGNN + a random node state + init weights — a self-contained demo."""
    edges = [(i, (i + 1) % N) for i in range(N)]
    model = QGNNEnergy(N, edges, n_layers=n_layers, phasor=True)
    theta = model.init_weights()
    rng = np.random.default_rng(seed)
    state = rng.normal(scale=0.3, size=2 * N)
    return model, state, theta


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run the QGNN energy circuit on IBM Quantum.")
    ap.add_argument("--N", type=int, default=4, help="network nodes / qubits")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--backend", default=None, help="named backend, else least-busy")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", type=int, default=3, help="transpiler optimization_level")
    ap.add_argument("--resilience", type=int, default=1, help="EstimatorV2 resilience_level")
    ap.add_argument("--simulator", action="store_true",
                    help="local StatevectorEstimator (no token / no network)")
    ap.add_argument("--dotenv", default=None)
    args = ap.parse_args()

    model, state, theta = _demo_model(args.N, args.layers, args.seed)
    print(f"[qgnn] N={args.N} layers={args.layers} edges={len(model.edges)}")

    if args.simulator:
        E = estimate_energy_local(model, state, theta)
        print(f"[local] ideal graph energy H = {E:+.6f}")
    else:
        out = estimate_energy_ibm(model, state, theta, args.backend,
                                  args.opt, args.resilience, args.dotenv)
        print(f"[ibm] backend={out['backend']} job={out['job_id']}")
        print(f"[ibm] transpiled depth={out['transpiled_depth']} "
              f"2q-gates={out['transpiled_2q_gates']}")
        print(f"[ibm] graph energy H = {out['energy']:+.6f} "
              f"(raw ⟨H_obs⟩={out['raw_expectation']:+.6f})")


if __name__ == "__main__":
    main()
