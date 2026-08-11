# Q-pHNN Experiment Report — Damped Harmonic Oscillator [Dynamic Circuit (COBYLA)]

**Generated:** 2026-07-14 01:22 UTC  |  **Runtime:** 2.0s

---

## Table of Contents

1. [Overview](#overview)

2. [Environment](#environment)

3. [Hyperparameters](#hyperparameters)

4. [Training Results](#training-results)

5. [Damping Recovery](#damping-recovery)

6. [Physics Validation](#physics-validation)

7. [Figures](#figures)

8. [Reproduce](#reproduce)

---

## Overview

**Q-pHNN v1 (Dynamic Circuit)** uses a 2-qubit Statevector circuit with Measurement-Induced NonLinearity (MINL) to model dissipation:
- Conservative block: `Rz(θ_J)` on system qubit
- Dissipative coupling: `CRY(θ_R)` entangles system + ancilla
- MINL collapse: `bitstr, sv = sv.measure([ancilla])` — Born-rule projection
- Feedforward: `if bitstr == '1': sv = sv.evolve(Rx(θ_kick))`

---

## Environment

| Package | Version |

|---------|---------|

| Python  | 3.13.14 |

| Qiskit  | 2.5.0 |

| NumPy   | 2.5.1 |

| SciPy   | 1.18.0 |

| Engine  | Qiskit 2.x StatevectorEstimator + Statevector.measure() MINL |

---

## Hyperparameters

| Parameter | Value |

|-----------|-------|

| System | Damped Harmonic Oscillator |

| Variant | Dynamic Circuit (COBYLA) |

| n_traj_steps | 6 |

| dt_traj | 1.0 |

| optimizer | COBYLA (gradient-free) |

| max_iter | 80 |

| n_shots_train | 30 |

| true_gamma | 0.3 |

---

## Training Results

### Loss History

| Metric | Value |

|--------|-------|

| Final Train Loss | 0.387830 |

| Final Val Loss | N/A |

| Iterations | 32 |

| Converged | True |

| Wall Time | 1.3s |

### Learned Parameters

```
params_opt = [-0.0476  0.6244  0.474 ]
```

---

## Physics Validation

| Physics Metric | Value |

|----------------|-------|

| Trajectory RMSE q(t) (normalised) | 0.687221 |

| Trajectory RMSE p(t) (normalised) | 0.802155 |

| Energy Monotone Fraction | 100.00% |


> Energy monotone fraction > 60% indicates the dissipative channel is correctly removing energy from the system.

---

## Figures

![summary.png](summary.png)



![training_loss.png](training_loss.png)



---

## Reproduce

```bash

cd hnn-quantum/codes

python experiments/run_qphnn_damped.py

```


All figures and this report are saved to `codes/report/qphnn_damped_harmonic_oscillator/`.
