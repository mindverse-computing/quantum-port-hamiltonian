# Q-pHNN Experiment Report — Damped Harmonic Oscillator [Vector Field (BFGS + γ)]

**Generated:** 2026-07-14 01:24 UTC  |  **Runtime:** 93.1s

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

**Q-pHNN v2 (Vector Field)** learns the port-Hamiltonian vector field by combining a quantum circuit energy ansatz with a classical damping parameter:


$$\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q} - \gamma p$$


The quantum circuit learns $H_\theta(q,p)$; $\gamma$ is a trainable scalar.

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

| Variant | Vector Field (BFGS + γ) |

| n_train | 160 |

| n_val | 40 |

| n_layers | 1 |

| max_iter | 200 |

| optimizer | BFGS (exact statevector) |

| true_gamma | 0.3 |

| k | 1.0 |

| m | 1.0 |

| rollout_steps | 200 |

---

## Training Results

### Loss History

| Metric | Value |

|--------|-------|

| Final Train Loss | 0.342626 |

| Final Val Loss | N/A |

| Iterations | 13 |

| Converged | True |

| Wall Time | 91.2s |

### Vector Field MSE

| Split | q̇ MSE | ṗ MSE |

|-------|--------|--------|

| Train | 0.165576 | 0.176702 |

| Val   | 0.268002 | 0.260319 |

### Learned Parameters

```
params_opt = [ 3.3189 -0.0156 -0.025   0.2367  1.536   0.2636]
```

---

## Damping Recovery

| Parameter | Value |

|-----------|-------|

| True γ | 0.3000 |

| Learned γ | 0.2636 |

| Absolute Error \|Δγ\| | 0.0364 |

| Relative Error \|Δγ\|/γ | 12.13% |


> **Key result**: A relative error < 10% indicates successful separation of conservative (J) and dissipative (R) dynamics on the quantum circuit.

---

## Physics Validation

| Physics Metric | Value |

|----------------|-------|

| Trajectory RMSE q(t) (normalised) | 0.438238 |

| Trajectory RMSE p(t) (normalised) | 0.438818 |

| Energy Monotone Fraction | 72.36% |


> Energy monotone fraction > 60% indicates the dissipative channel is correctly removing energy from the system.

---

## Figures

![phase_portrait.png](phase_portrait.png)



![summary.png](summary.png)



![training_loss.png](training_loss.png)



![trajectory.png](trajectory.png)



![vector_field_val.png](vector_field_val.png)



---

## Reproduce

```bash

cd hnn-quantum/codes

python experiments/run_qphnn_damped.py

```


All figures and this report are saved to `codes/report/qphnn_damped_harmonic_oscillator/`.
