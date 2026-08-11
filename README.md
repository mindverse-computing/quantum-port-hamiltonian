<div align="center">

# Quantum Port Hamiltonian

**Conservation carried by a unitary, dissipation carried by a measurement.**

[![DOI](https://zenodo.org/badge/DOI/0.5281/zenodo.21894846.svg)](https://doi.org/0.5281/zenodo.21894846)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/downloads/)
[![Framework: Qiskit](https://img.shields.io/badge/Framework-Qiskit-6929c4.svg)](https://www.ibm.com/quantum/qiskit)
[![Hardware: IBM Quantum](https://img.shields.io/badge/hardware-IBM%20Quantum-052fad.svg)](https://quantum.ibm.com/)
[![Stewarded by Mindverse Computing](https://img.shields.io/badge/stewarded%20by-Mindverse%20Computing-8a90ff.svg)](https://github.com/mindverse-computing)

</div>

---

## The construction

A port-Hamiltonian system separates what routes energy from what removes it:

$$
\dot{x} = \big(J(x) - R(x)\big)\nabla H(x) + G(x)u
$$

with $J = -J^{\top}$ skew and $R \succeq 0$ positive-semidefinite. The split is
the content of the model: $J$ conserves, $R$ dissipates, and the power balance
$\dot H = -(\nabla H)^{\top} R \nabla H + y^{\top}u$ follows from the algebra
rather than from a fitted constraint.

This package realises both halves on a quantum circuit. The energy is a scaled
Pauli expectation of a parameterised circuit,

$$
H_\theta(q, p) = s \cdot \langle ZZ \rangle(q, p; \theta) + b
$$

and the symplectic gradients come from the **parameter-shift rule** applied to
the data-encoding gates, exactly and in two circuit evaluations each:

$$
\frac{\partial \langle O \rangle}{\partial \theta}
= \tfrac{1}{2}\Big[\langle O \rangle\big(\theta + \tfrac{\pi}{2}\big)
                 - \langle O \rangle\big(\theta - \tfrac{\pi}{2}\big)\Big]
$$

One consequence is load-bearing and easy to miss. Taking $\dot q = +\partial
H/\partial p$ and $\dot p = -\partial H/\partial q$ from the *same* circuit makes
the energy rate vanish identically:

$$
\frac{dH}{dt}
= \frac{\partial H}{\partial q}\dot q + \frac{\partial H}{\partial p}\dot p
= \frac{\partial H}{\partial q}\frac{\partial H}{\partial p}
- \frac{\partial H}{\partial p}\frac{\partial H}{\partial q} = 0
$$

Conservation is therefore structural, not trained — it holds at random parameter
values, and the test suite checks it to $10^{-11}$ on an untrained circuit. It is
not a penalty term that optimisation can trade away.

## Dissipation as measurement

Unitary evolution cannot lose energy. Measurement can, and that is the whole
argument for putting this model on a quantum device rather than writing it
classically.

Each node carries a bath ancilla. Per step the circuit applies a conservative
rotation $R_z(\theta_J)$, entangles system and bath with $\mathrm{CRY}(\theta_R)$,
**measures the ancilla**, and applies a feed-forward kick $R_x(\theta_{\text{kick}})$
conditioned on the outcome. Tracing out the measured ancilla leaves a
completely-positive trace-preserving map on the system — a discrete-time Lindblad
channel:

$$
\rho \;\longmapsto\; \sum_k K_k \rho K_k^{\dagger},
\qquad \sum_k K_k^{\dagger}K_k = I
$$

So the $R$ channel is a physical irreversibility, not a non-unitary term inserted
into a Hamiltonian and not a damping coefficient multiplied onto a velocity. The
control that establishes this is $\theta_R = 0$: the ancilla is still measured
every step, the circuit depth is unchanged, but the measurement carries no
information about the system and the energy is then conserved exactly.

## Installation

Requires Python 3.10 or newer.

```bash
git clone <repo>/quantum-port-hamiltonian && cd codes
pip install -e .
```

The core install carries qiskit, scipy, numpy and matplotlib — enough to import
the whole package and run every simulation experiment. Hardware access is an
extra, because the IBM runtime and the credential loader are imported only by
`ibm/`:

```bash
pip install -e ".[ibm]"     # qiskit-ibm-runtime + python-dotenv
pip install -e ".[dev]"     # pytest
pytest
```

## Quick start

```python
import numpy as np
from non_dissipative.quantum_hnn import QuantumHNN

model = QuantumHNN(n_layers=1)
theta = np.array([1.723, -1.582, 1.170, 1.594])   # trained parameters

qdot, pdot = model.vector_field(0.8, 0.0, theta)   # parameter-shift, exact

print(round(qdot, 4), round(pdot, 4))
# -0.0103 -0.781
```

`vector_field` returns $(\dot q, \dot p)$ from four circuit evaluations. At the
top of the pendulum's swing the position derivative is near zero and the momentum
derivative carries the restoring force. Evaluating $dH/dt$ at that point returns
$-5.9 \times 10^{-12}$: the flow is conservative to machine precision, before any
integrator is chosen. Rolled out with Störmer–Verlet the orbit closes to 1.35%
relative energy drift over 300 steps, against 12.3% for the same field under
explicit Euler.

## Package layout

| Module | Contents |
|---|---|
| `common` | the parameter-shift rule, the isomorphic mapping between classical $(q,p)$ and qubit angles, physics metrics, plotting |
| `non_dissipative` | **Q-HNN** — the conservative single-oscillator model and its pendulum reference systems |
| `dissipative` | **Q-pHNN** — `DynamicQpHNN` (MINL, mid-circuit measurement) and `VectorFieldQpHNN` (analytic $\gamma$ channel) |
| `network` | **QGNN** — topology-entangled circuits lifting both models to $N$ nodes, with the graph-structured energy $\hat H_\theta = \sum_i a_i \langle Z_i \rangle + \sum_{(i,j) \in E} w_{ij} \langle Z_i Z_j \rangle$ |
| `ibm` | connection, execution harness, staged preflight, hardware parameter-shift, smoke test |
| `tests` | 23 tests over conservation, dissipation and the shift rule |

The coupling graph *is* the entanglement graph: an edge $(i,j)$ in the coupling
matrix places a two-qubit entangler between qubits $i$ and $j$.

## Experiments

Seven suites under `experiments/`, each self-contained — its runner, its report,
its figures, its records.

| Suite | What it measures | Where it runs |
|---|---|---|
| `qhnn_pendulum/` | energy conservation on the nonlinear pendulum | simulation |
| `qphnn_damped/` | energy monotonicity under the MINL channel | simulation |
| `layer_ablation/` | expressibility against circuit depth | simulation |
| `network_conservative/` | conservation lifted to $N$-node networks | simulation |
| `network_dissipative/` | measurement-induced dissipation on networks | simulation |
| `network_scaling/` | dissipation across topology and size | simulation |
| `ibm_hardware/` | the same observables on a real QPU | **IBM Quantum** |

```bash
python experiments/qhnn_pendulum/run.py
python experiments/network_conservative/run.py --nodes 3 --epochs 3
```

Runners overwrite their own figures; the versions here are from the published
runs.

## Hardware

Credentials go in a git-ignored `.env` (see `.env.example`). Check readiness
before spending device time — the preflight is staged so a red line names its own
cause:

```bash
python -m ibm.preflight          # 1 offline · 2 noisy · 3 iam · 4 service
python -m ibm.preflight --live   # 5 live: also submit one Bell circuit
```

Stages 1–2 need no token and no network. Stage 5 is opt-in and is the only stage
that consumes QPU time. `run_gradients.py --mode fake` writes the same JSON
schema the hardware run writes, so the analysis is settled before any device time
is spent — which matters on the free tier, where the allowance is 10 minutes of
QPU per month.

The campaign reported in the manuscript ran 12 jobs on a 156-qubit Heron
processor. Two results are worth stating plainly because they are negative. The
error budget is **readout-dominated** at this depth — readout contributes
$16\times$ the two-qubit gate error — so readout mitigation buys a 50.6% accuracy
improvement while zero-noise extrapolation buys a much smaller second step and
lands at the shot-noise floor, where no further mitigation can help. And network
energy conservation degrades sharply with size: Pearson $r$ against the exact
statevector falls from 0.989 at $N=4$ to 0.925 at $N=8$ and is lost entirely by
$N=16$.

## Result integrity

Hardware records carry their own provenance — the IBM job identifier, the device
calibration snapshot taken at submission, transpiled circuit statistics, raw
expectation values, and the exact statevector reference they are scored against.

```bash
python check_numbers.py
```

101 checks that every numeric literal quoted in the manuscript matches its
archived value. One is expected to fail: an invariant asserting the QPU account
is not overspent against its monthly allowance. It is left failing rather than
relaxed — a guard is only useful if it is allowed to report bad news.

## Scope

This models classical dynamical systems on quantum hardware. It makes no claim
that pendulums or phasor networks are quantum systems, and no claim of quantum
advantage: at the sizes reached here a classical symplectic integrator is faster
and more accurate. What the circuit provides is structural — conservation that
holds by unitarity rather than by penalty, and dissipation that is a measurement
channel rather than a matrix chosen to be positive-semidefinite.

## Documentation

Nine chapters under [`docs/`](docs/README.md): theory, the single-DOF models, the
network models, training and metrics, the experiments, and next steps.

## Citing

Please cite the manuscript and the archived software release. A DOI will be
minted with the first tagged release; the badge above is a placeholder.

## Licence

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Manuscript sources
and figures are not licensed for redistribution here.
