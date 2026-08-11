# Symplectic gradients on IBM hardware — Q-pHNN plan steps 6-7

**Backend** `ibm_marrakesh` (156q Heron r2) · **job** `d9t67lhdsedc73aihho0` · **1 job, 1 PUB, 160 shifted
evaluations, 128 shots each** · **QPU charge 17.0 s** · wall 36 s
Result file: `codes/results/ibm_qhnn_gradients_hw.json` — every number below is read from it.

## What was measured

The manuscript's central mechanism is not the energy value but the **parameter-shift rule on the
data-encoding gates**, which is claimed to return Hamilton's equations:

    q̇ = ∂H/∂p  =  s·½[⟨ZZ⟩(q, p+π/2) − ⟨ZZ⟩(q, p−π/2)]
    ṗ = −∂H/∂q = −s·½[⟨ZZ⟩(q+π/2, p) − ⟨ZZ⟩(q−π/2, p)]

The convention is taken verbatim from `QuantumHNN.q_dot`/`.p_dot` and `common/parameter_shift.py`.
Before submission the array-bound implementation was checked against those methods on all 40 points:
**max |deviation| = 0e+00** (exact agreement).

Evaluation set: the manuscript's validation split, reproduced bit-for-bit —
`NonlinearPendulum(q_range=π/2, p_range=1).generate(200, seed=42).train_test_split(0.20, seed=42)`
→ **all 40 validation points**, at θ* = [1.723, -1.582, 1.17, 1.594], s* = 1.335.

**Batching.** All 4 × 40 = 160 shifted evaluations ship as a single PUB on a **single transpiled ISA
circuit** with a (160, 6) parameter array. This matters beyond cost: transpiling each shifted angle
set separately at `optimization_level=3` would fold particular angles away, so the ± members of a
shift pair would no longer be the same experiment and the difference would mix a gradient with a
transpiler artefact. One ISA circuit means each pair shares gate sequence and qubit layout, so
coherent errors partially cancel in the difference.

## Per-component accuracy versus exact statevector

| | MAE | RMSE | max abs | bias | median rel err | Pearson r |
|---|---|---|---|---|---|---|
| q̇ | 0.0675 | 0.0911 | 0.2594 | -0.0011 | 13.7% | 0.9821 |
| ṗ | 0.0760 | 0.0904 | 0.1758 | -0.0340 | 17.2% | 0.9917 |

Both components correlate with the exact field at r = 0.982 (q̇) and 0.992 (ṗ). Relative to the dynamic range the field
spans on this split (q̇ ∈ [-0.99, 0.91], ṗ ∈ [-1.15, 1.08]), the RMSE is
4.8% and 4.1% of range respectively.

*Mean relative error is 0.48 (q̇) and 0.33 (ṗ) and should not be quoted:
the pendulum vector field passes through zero inside the validation domain, so the ratio diverges.
Median relative error, MAE and RMSE are the interpretable summaries.*

## Is the hardware field within shot noise? — **Close, but no**

The estimator's reported standard errors run at a median **0.67×** the binomial
expectation √((1−⟨ZZ⟩²)/N) under gate twirling at 128 shots, and some collapse toward zero, which
makes per-point pulls unreliable. The shot floor was therefore recomputed independently from binomial
statistics and propagated through the shift rule as |s|/2·√(σ₊²+σ₋²).

| | observed RMSE | shot floor (binomial) | ratio | χ²/dof | excess over shot noise |
|---|---|---|---|---|---|
| q̇ | 0.0911 | 0.0772 | 1.18 | 1.37 | 0.0484 |
| ṗ | 0.0904 | 0.0729 | 1.24 | 1.55 | 0.0535 |

**χ²/dof of 1.37 and 1.55 (40 points each): the hardware vector field is near the shot-noise
floor but not statistically within it.** At 128 shots the residual is dominated by sampling — roughly
85% of the RMSE in quadrature — with a device contribution of 0.048 (q̇) and
0.054 (ṗ) on top.

## The two systematics

1. **Observable contraction.** Raw ⟨ZZ⟩ regresses on exact with slope
   **λ = 0.9829** — a ~1.7% depolarising-like shrinkage, consistent with the
   0.10% CZ error and 3.1% readout error on qubit 135. It propagates linearly through
   the shift rule (q̇ slope 0.9890, ṗ slope 0.9743) and is correctable by one scalar.

2. **A ṗ-only offset that contraction does not explain.** ṗ carries a bias of
   **-0.0340** (-2.53 empirical σ of the mean), while q̇ is unbiased
   (-0.0011, -0.08σ). A pure contraction predicts a bias of only +0.0015 for ṗ,
   and rescaling the hardware gradients by 1/λ leaves the offset essentially untouched
   (-0.0356). It appears as an **intercept** (-0.0355), not a slope.

   The asymmetry is structurally interesting: q̇ shifts `p_in` (an Ry on qubit 1, readout error
   0.17%) while ṗ shifts `q_in` (an Rx on qubit 0, readout error 3.1% — **18× worse**,
   and T2 = 91 µs against 131 µs). The component whose shift acts on the worse qubit is the
   biased one. With one job this is a consistent explanation, not a demonstrated cause — separating
   it needs a qubit-swap control run.

## Device state at submission

Qubits [135, 139] · logical depth 5 → ISA depth 15 · 2 two-qubit gates ·
success proxy 0.9980 · calibration read 2026-08-10T23:50:45.805343+00:00

| qubit | T1 (µs) | T2 (µs) | readout err |
|---|---|---|---|
| 135 | 340 | 91 | 0.0310 |
| 139 | 151 | 131 | 0.0017 |

CZ(135,139) error 0.00101 · settings: resilience_level 1,
dynamical decoupling on, gate + measurement twirling on, optimization_level 3.

## Budget — an overrun, reported plainly

| | |
|---|---|
| Jobs allocated to this track | 2 (1 planned + 1 contingency) |
| Jobs submitted | **1** |
| QPU allocated | ~10 s |
| **QPU charged** | **17.0 s** |
| Predicted before submission | 7.1 s |

The charge came in at 17.0 s against a 7.1 s prediction from the device rep-delay model
(2.0 s overhead + 250 µs × 20 480 shots, calibrated on the 3.0 s Bell job). The script carries a
pre-execution guard that cancels the job when IBM's own estimate exceeds a cap, but
`job.usage_estimation` raised `TypeError` on this backend, so the guard had no number to test and
did not fire.

**The contingency job was deliberately left unspent.** This track's job charged
17 s; the ledger is shared with sibling tracks, so its running total at the time of
this run reflects their jobs too and is not a per-track figure. The campaign total
at completion was 340 s spent of the 600 s monthly Open-plan allowance
(260 s remaining, 11 jobs) — see `results/ibm_qpu_ledger.json`, which is authoritative.

## What a second job would buy, if one is authorised later

- **Qubit-swap control** (same circuit, `q_in`/`p_in` mapped to the opposite physical qubits) — this
  is the one measurement that would confirm or kill the readout-asymmetry explanation of the ṗ offset.
- **Higher shots** (512–1024) to push the sampling floor below the device systematic and measure the
  0.054 excess directly rather than by quadrature subtraction.

## Bottom line

The parameter-shift rule on data-encoding gates **works on hardware**: 40-point vector fields
recovered at r = 0.982 (q̇) and 0.992 (ṗ) with RMSE ≈ 0.09 in both components, ~4–5% of the field's dynamic range, from
one 17 s job. The residual is mostly shot noise at 128 shots (χ²/dof 1.37 and 1.55), with a
~1.7% observable contraction and a ṗ-specific offset of -0.034 on top. The claim that survives
review is *"near the shot-noise floor with a small correctable contraction and one unexplained
component-asymmetric offset"* — not *"within shot noise"*.
