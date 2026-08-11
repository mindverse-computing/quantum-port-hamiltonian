<div align="center">

## What is in this repository

|                      |                                                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `common/`          | **shared machinery** — the parameter-shift rule, the isomorphic Hamiltonian mapping, physics metrics, plotting |
| `non_dissipative/` | **Q-HNN** — the conservative single-oscillator model                                                           |
| `dissipative/`     | **Q-pHNN** — the dissipative model, damping realised by mid-circuit measurement                                |
| `network/`         | **QGNN** — topology-entangled circuits lifting both models to $N$-node networks                              |
| `ibm/`             | **hardware access** — connection, execution harness, staged preflight, smoke test                              |
| `experiments/`     | one directory per experiment: the code that produces the numbers, and the numbers themselves                          |
| `tests/`           | unit tests over the network and QGNN models                                                                           |
| `docs/`            | nine chapters: theory, models, training, experiments, next steps                                                      |
| `legacy/`          | retired scripts and early feasibility notes, kept rather than deleted                                                 |

The split is between **library** and **experiments**. The five core packages
import nothing from `experiments/`; every experiment imports the library.

## The construction

A port-Hamiltonian system splits its dynamics into a skew-symmetric
interconnection that conserves energy and a positive-semidefinite term that
dissipates it:

$$
\dot{\mathbf{x}} = \big(\mathbf{J}(\mathbf{x}) - \mathbf{R}(\mathbf{x})\big)\nabla H(\mathbf{x}) + \mathbf{G}(\mathbf{x})\mathbf{u}
$$

Both halves are realised on a quantum circuit, and the second is the reason for
the approach. The conservative flow is carried by a **unitary**: because
$U^\dagger U = I$ exactly, the skew structure holds by circuit construction
rather than by penalty, and energy conservation is not a term in a loss that
training trades away.

The energy is read out as a scaled expectation of a Pauli observable,

$$
H_\theta(q, p) = s \cdot \langle ZZ \rangle(q, p; \theta) + b
$$

and its symplectic gradients come from the **parameter-shift rule** applied to
the data-encoding gates, giving exact analytic derivatives from two circuit
evaluations each — no finite differences, no automatic differentiation:

$$
\dot q = \frac{\partial H}{\partial p} = \tfrac{1}{2}\big[H(q, p{+}\tfrac{\pi}{2}) - H(q, p{-}\tfrac{\pi}{2})\big]
$$

Dissipation is where a quantum circuit stops being a re-implementation of a
classical model. A bath ancilla is attached to each node; per step the circuit
applies a controlled system–bath rotation, **measures the ancilla**, and applies
a feed-forward kick conditioned on that outcome. Each step is a genuine
discrete-time CPTP map — a Lindblad channel — so the damping is produced by
Born-rule measurement, with no non-unitary term in the Hamiltonian and no
classical damping coefficient anywhere in the model. This is the
measurement-induced nonlinearity (MINL) the manuscript is about.

## Installation

Requires Python 3.10 or newer.

```bash
git clone <repo> && cd codes
pip install -e .
```

The core install carries qiskit, scipy, numpy and matplotlib — enough to import
the package and run every simulation experiment. Hardware access is an extra,
because the IBM runtime and the credential loader are imported only by `ibm/`:

```bash
pip install -e ".[ibm]"     # qiskit-ibm-runtime + python-dotenv
pip install -e ".[dev]"     # pytest
pytest
```

The editable install puts the core packages on the import path, so runners work
from any working directory.

## Quick start

```python
import numpy as np
from non_dissipative.quantum_hnn import QuantumHNN

model = QuantumHNN(n_layers=1)
theta = np.array([1.723, -1.582, 1.170, 1.594])   # the trained parameters

qdot, pdot = model.vector_field(0.8, 0.0, theta)   # parameter-shift, exact
print(round(qdot, 4), round(pdot, 4))
# -0.0103 -0.781
```

`vector_field` returns the symplectic pair: `qdot` is $\partial H/\partial p$ and
`pdot` is $-\partial H/\partial q$, each from two circuit evaluations. At the top
of the swing the position derivative is near zero and the momentum derivative
carries the restoring force, as a pendulum requires. Rolled out with a
Störmer–Verlet integrator, this field keeps the orbit closed — the relative
energy drift over 300 steps is 1.35%, against 12.3% for the same field
integrated with first-order Euler.

## Experiments

Seven suites under `experiments/`, each self-contained: its runner, its report,
its figures, its records.

| Experiment                | What it measures                                | Where it runs         |
| ------------------------- | ----------------------------------------------- | --------------------- |
| `qhnn_pendulum/`        | energy conservation on the nonlinear pendulum   | simulation            |
| `qphnn_damped/`         | energy monotonicity under the MINL channel      | simulation            |
| `layer_ablation/`       | expressibility against circuit depth            | simulation            |
| `network_conservative/` | exact conservation lifted to$N$-node networks | simulation            |
| `network_dissipative/`  | measurement-induced dissipation on networks     | simulation            |
| `network_scaling/`      | dissipation across topology and size            | simulation            |
| `ibm_hardware/`         | the same observables on a real QPU              | **IBM Quantum** |

```bash
python experiments/qhnn_pendulum/run.py
python experiments/network_conservative/run.py --nodes 3 --epochs 3
```

Each runner writes into its own `figures/` and `results/`. **Runners overwrite
their own figures** — the versions here are from the published runs, and
re-running with different settings replaces them.

## IBM Quantum

Credentials go in a git-ignored `.env` (see `.env.example`); never commit it.

```
IBM_QUANTUM_TOKEN=...
IBM_QUANTUM_INSTANCE=...
```

Check readiness before spending device time. The preflight is staged so a red
line names its own cause rather than reporting that something went wrong:

```bash
python -m ibm.preflight          # 1 offline · 2 noisy · 3 iam · 4 service
python -m ibm.preflight --live   # 5 live: also submit one Bell circuit
```

Stages 1–2 need no token and no network: they run the Bell circuit and the QGNN
energy on the local statevector, then on a fake IBM device with a real noise
model. Stages 3–4 check that the API key exchanges for an IAM token and that the
instance binds to operational backends. Stage 5 is opt-in and is the only stage
that consumes QPU time.

```bash
python experiments/ibm_hardware/run_gradients.py --mode fake       # dry run
python experiments/ibm_hardware/run_gradients.py --mode hardware   # spends QPU time
```

`run_gradients.py` has a `--mode fake` path that writes the same JSON schema the
hardware run writes, so the analysis is settled before any device time is spent.
`run_qhnn_energy.py` is a module of building blocks — grid construction, circuit
binding, scoring, one function per mitigation rung — rather than a command-line
script; it is driven from a session that composes those pieces.

Dry-running first matters on the free tier: the allowance is 10 minutes of QPU
per month, and one mis-specified job can consume a large fraction of it.

## Result integrity

Hardware records carry their own provenance — the IBM job ID, the device
calibration snapshot taken at submission, transpiled circuit statistics, raw
expectation values, and the exact statevector reference they are scored against.

`check_numbers.py` verifies that every numeric literal quoted in the manuscript
matches its archived value:

```bash
python check_numbers.py
```

101 checks. One is expected to fail: an invariant asserting the QPU account is
not overspent against its monthly allowance. That is a true finding about the
account, left failing rather than relaxed — the guard is only useful if it is
allowed to report bad news.

`experiments/ibm_hardware/make_figures.py` regenerates the manuscript's eight
hardware figures from those records alone. No value is typed into the figure
code.

## Scope

This repository models classical dynamical systems on quantum hardware. It makes
no claim that the systems studied — pendulums, damped oscillators, phasor
networks — are quantum systems, and no claim of quantum advantage: at the sizes
reached here a classical symplectic integrator is faster and more accurate. What
the circuit provides is structural, not computational: conservation that holds by
unitarity rather than by penalty, and dissipation that is a physical measurement
channel rather than a matrix chosen to be positive-semidefinite.

## Licence

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Manuscript sources
and figures are not licensed for redistribution here.
