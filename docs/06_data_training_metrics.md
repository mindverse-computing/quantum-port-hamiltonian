# Chapter 6 — Data, Training & Metrics

How the models are trained and validated: data generators, the parameter-shift
trainers, and the metrics that certify energy conservation, dissipation, and
coupling recovery.

## 6.1 Data generators

All data is synthetic — the true equations of motion are integrated so the
ground-truth energy, damping, and (for networks) coupling matrix are known and can
be used to score the learned model.

| Module | Systems | Regime |
|--------|---------|--------|
| `non_dissipative/data_generator.py` | `NonlinearPendulum`, `HarmonicOscillator` | conservative |
| `dissipative/data_generator.py` | `DampedHarmonicOscillator` (γ=0.3), `VanDerPolOscillator` (μ=0.5) | dissipative |
| `network/data_generator.py` | conservative / dissipative Kuramoto networks | both |

The single-DOF generators return a vector-field dataset (states + true
derivatives). The network generator returns a `NetworkVectorFieldDataset` with the
coupling `K`, per-node damping `gamma`, and true Hamiltonian `H_true`, plus the
encoding scales.

## 6.2 Training by parameter-shift + BFGS

Every model is trained the same way: minimise the **vector-field MSE** between the
model's predicted `ẋ` and the true derivatives, using a gradient-based optimiser.

- **Single-DOF conservative** — `train_qhnn` (`non_dissipative/trainer.py`).
- **Single-DOF dissipative** — `train_dynamic_qphnn` (MINL variant) and
  `train_vector_field_qphnn` (damping variant) in `dissipative/trainer.py`.
- **Network** — `train_network_qphnn(model, train_data, test_data, max_iter=100)`
  in `network/trainer.py`: scipy `BFGS` on `model.compute_loss`, which is the MSE
  between `model.vector_field` and the dataset's `d_states`. For dissipative
  gamma-mode models it also reports the mean recovered damping `γ̄` during
  training.

The gradients driving the vector field are **exact parameter-shift** gradients
(Chapter 2), not finite differences. The outer optimisation over circuit weights
uses BFGS.

## 6.3 Metrics (`common/metrics.py`)

The metrics translate a trained circuit into physical validation numbers.

### Energy conservation / drift

`compute_energy_conservation(H_vals)` returns `(std(H), std(H)/|mean(H)|)`. The
second value is the **relative energy drift** — the headline conservative metric.
For a well-learned conservative model it is small (a few percent for the pendulum;
near-zero for the network J-channel where the two symplectic terms cancel).

### Energy monotonicity

`compute_energy_monotone_fraction(H_vals)` returns the fraction of steps with
$H(t{+}1) \le H(t)$. This is the **dissipative** diagnostic: a genuinely passive
channel has monotone-decreasing energy, so this fraction approaches 1. It is the
key evidence that MINL implements real dissipation.

### Trajectory RMSE and damping recovery

`compute_trajectory_rmse` scores rollout accuracy against the true trajectory. For
the dissipative single-DOF model, the recovered damping error $|\hat\gamma - \gamma|/\gamma$ measures
how well the learned model identifies the physical friction rate. For networks,
the per-node $\hat\gamma_i$ are compared against the true $\gamma_i$.

### The metric records

`EpochRecord`, `QHNNMetrics`, and `QpHNNMetrics` are dataclasses that accumulate
per-epoch loss and the validation numbers, so an experiment produces a structured,
serialisable record alongside its figures.

## 6.4 What to expect

| Metric | Conservative | Dissipative |
|--------|--------------|-------------|
| Relative energy drift | small (few %); ~0 for network J-channel | n/a |
| Energy monotone fraction | n/a | high (→1) |
| Passivity $\dot H \le 0$ | `Ḣ = 0` exactly (network) | $\dot H \le 0$ by construction |
| Damping recovery `|Δγ|/γ` | n/a | tens of % at demo budget |
| Vector-field MSE | small | small |

The physics rows (drift = 0, $\dot H \le 0$) are structural — they follow from the
unitary J-channel and the gradient-damping R-channel, not from training succeeding.
The recovery rows depend on training effort and scale.

The next chapter runs the actual experiments.
