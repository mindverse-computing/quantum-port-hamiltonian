# Q-HNN Experiment Report — Nonlinear Pendulum

**Generated:** 2026-07-14 01:22 UTC  |  **Runtime:** 16.4s

---

## Table of Contents

1. [Overview](#overview)

2. [Environment](#environment)

3. [Hyperparameters](#hyperparameters)

4. [Training Results](#training-results)

5. [Physics Validation](#physics-validation)

6. [Figures](#figures)

7. [Reproduce](#reproduce)

---

## Overview

The **Quantum Hamiltonian Neural Network (Q-HNN)** learns the scalar energy manifold $H_\theta(q, p)$ of a conservative dynamical system using a 2-qubit parameterized quantum circuit. Symplectic gradients (Hamilton's equations) are extracted via the **Parameter-Shift Rule on data-encoding gates**:


$$\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q}$$


The Q-HNN is strictly energy-conserving by construction (all gates unitary).

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

| System | Nonlinear Pendulum |

| n_train | 160 |

| n_val | 40 |

| n_layers | 1 |

| max_iter | 200 |

| optimizer | BFGS (exact statevector) |

| q_range | (-1.571, 1.571) |

| p_range | (-1.00, 1.00) |

| rollout_steps | 300 |

| dt | 0.05 |

| n_weights | 6 |

---

## Training Results

### Loss History

| Metric | Value |

|--------|-------|

| Final Train Loss | 0.075391 |

| Best Train Loss | 0.075391 |

| Final Val Loss | N/A |

| Iterations | 8 |

| Converged | True |

| Wall Time | 14.2s |

### Vector Field MSE

| Split | q̇ MSE | ṗ MSE |

|-------|--------|--------|

| Train | 0.061855 | 0.013537 |

| Val   | 0.040237 | 0.012368 |

### Learned Parameters

```
θ_opt = [ 1.7231 -1.5824  1.1695  1.5935  1.3354  0.    ]
```

---

## Physics Validation

| Physics Metric | Value |

|----------------|-------|

| Energy Conservation Error std(H) | 0.010928 |

| Relative Energy Drift std(H)/|⟨H⟩| | 1.3526% |

| Trajectory RMSE q(t) (normalised) | 0.303894 |

| Trajectory RMSE p(t) (normalised) | 0.288660 |


> **Note**: Energy conservation error < 1% relative drift indicates the Q-HNN has learned a physically consistent energy manifold.

---

## Figures

![phase_portrait.png](phase_portrait.png)



![training_loss.png](training_loss.png)



![trajectory.png](trajectory.png)



![vector_field_val.png](vector_field_val.png)



---

## Reproduce

```bash

cd hnn-quantum/codes

python experiments/run_qhnn_pendulum.py

```


All figures and this report are saved to `codes/report/qhnn_nonlinear_pendulum/`.
