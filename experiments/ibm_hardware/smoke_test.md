# hnn-quantum — Environment & Baseline Smoke Test

**Env:** conda `hnn-quantum` — Python 3.11, qiskit 2.5.0, scipy 1.17.1, numpy 2.4.6, matplotlib.
Qiskit 2.x `StatevectorEstimator` / `Statevector.measure()` APIs confirmed available.

## Baseline reproduction (existing 1-DOF code)

| Experiment | Metric | Result | Manuscript |
|---|---|---|---|
| Q-HNN (nonlinear pendulum, conservative) | Energy drift (rel) | 3.16% | 3.16% |
| Q-HNN | Val q̇ MSE / ṗ MSE | 0.099 / 0.021 | — |
| Q-pHNN v2 (damped oscillator) | Learned γ (true 0.30) | 0.342 | — |
| Q-pHNN v2 | \|Δγ\|/γ | 13.91% | 13.9% |
| Q-pHNN v2 | Energy monotone fraction | 63.3% | — |

All imports (`non_dissipative`, `dissipative`, `common`) resolve; figure + report
writers produce PNGs and `report.md` under `codes/report/`.

**Conclusion:** 2-qubit baseline is healthy and reproduces published numbers.
Ready to build the N-node network extension on top.
