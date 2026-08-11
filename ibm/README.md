# `ibm/` — IBM Quantum hardware path (Qiskit-native)

The CUDA-Q port (`../../cudaq`) runs the network model on an NVIDIA GPU up to the
single-device simulation ceiling. **IBM Quantum is not a CUDA-Q backend** —
CUDA-Q's hardware targets are IonQ, Quantinuum, IQM, OQC, and Quantum Circuits —
so real-hardware execution goes through Qiskit / `qiskit-ibm-runtime`. This
subpackage adds that path to the Qiskit reference codebase, reusing the existing
circuit builders in `../network` (same ansatz, same observable the CUDA-Q parity
gate checks against).

## Layout

- `connection.py` — load `IBM_QUANTUM_TOKEN` / `INSTANCE` / `CHANNEL` from a
  git-ignored `.env` (or the session environment) and open a
  `QiskitRuntimeService`; `pick_backend()` resolves a named or least-busy device.
- `sample_circuit.py` — a 2-qubit Bell-state connectivity smoke test. Run this
  first to confirm the token/instance/backend work before spending time on the
  model circuits.
- `runner.py` — transpile + submit the QGNN **energy** circuit via `EstimatorV2`
  (expectation values with error mitigation), returning the graph energy plus
  the transpiled 2-qubit depth (error-budget proxy).

## Credentials

Copy `../.env.example` to `../.env` (repo-root, git-ignored) and fill in your
token. Never commit `.env`. The loader also reads the process environment, so a
session-injected `IBM_QUANTUM_TOKEN` works without a file.

## Usage

```bash
# from the codes/ directory, with it on PYTHONPATH
export PYTHONPATH="$PWD"

# 1) offline pipeline check — no token, no network (local statevector)
python -m ibm.sample_circuit --simulator
python -m ibm.runner --simulator --N 4 --layers 2

# 2) connectivity smoke test on a real device (needs .env)
python -m ibm.sample_circuit                 # least-busy real backend
python -m ibm.sample_circuit --backend ibm_fez

# 3) run the QGNN energy circuit on hardware
python -m ibm.runner --N 4 --layers 2 --backend ibm_fez --opt 3 --resilience 1
```

## Validation status

The build → transpile → (ISA-observable) → estimate → parse pipeline is validated
offline two ways:

1. `ibm.runner`'s local statevector path reproduces `network.QGNNEnergy.energy()`
   **exactly** (|diff| = 0) — the circuit binding and observable are provably
   correct.
2. The full transpile → `EstimatorV2` path runs end-to-end on a **fake IBM
   backend** with realistic noise (ideal H vs. noisy-device H differ only by the
   expected device noise).

Only the live token + network submission is untested here (it needs your
credentials). The large-N hardware campaign — topologies at N=20…150,
transpilation protocol, ancilla-reuse for the MINL channel, and quantum error
analysis — is specified in
[`../../cudaq/experiments/IBM-Quantum-plan.md`](../../cudaq/experiments/IBM-Quantum-plan.md).
