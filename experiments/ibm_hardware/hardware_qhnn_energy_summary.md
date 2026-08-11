# Q-HNN energy landscape on IBM hardware

Backend **ibm_marrakesh** (156-qubit Heron r2). Three EstimatorV2 jobs, **119 s** of QPU
time. Every number below is readable out of a results JSON and is re-checked by
`check_numbers.py` (39 checks, passing).

## What was run

The trained 2-qubit Q-HNN energy `H(q,p) = s*<ZZ>(q,p;theta*) + b` was evaluated over the
manuscript's training box, q in [-1.6, 1.6] x p in [-1.4, 1.4], at
theta* = [1.723, -1.582, 1.17, 1.594], s = 1.335, b = 0.0. One bound circuit per grid point;
the whole grid ships as PUBs in a single job. Binding was verified against
`model.energy(q, p, phi)` to |diff| = 0 before any submission.

| job | setting | points | shots | QPU s | job id |
|---|---|---|---|---|---|
| 1 | resilience 1, DD on, twirling on | 36 (full grid) | 4096 | 52 | `d9t66cvpemts73cug3q0` |
| 2 | resilience 0, DD off, twirling off | 12 (subset) | 4096 | 15 | `d9t6897tfhrs73dthqh0` |
| 3 | resilience 2 (ZNE), DD on, twirling on | 12 (subset) | 4096 | 52 | `d9t68h7pemts73cug6gg` |

Job 1 charged 52 s rather than the ~5 s the single-PUB Bell test implied: at 36 PUBs the
charge is **sampling-dominated, not per-job-overhead-dominated**. Rungs 0 and 2 were
therefore run on a 12-point subset chosen by energy quantile, so the deep well, the flat
region and the zero crossing are all represented. Job 1's error on those same 12 points
(MAE 0.0392) matches its full-grid error (MAE 0.0391), confirming the subset is
representative.

## Step 1 - energy landscape, full 36-point grid

Hardware vs exact statevector, from `ibm_qhnn_energy_hw.json`:

| metric | value |
|---|---|
| MAE | 0.0391 |
| RMSE | 0.0477 |
| median relative error | 0.1021 |
| max absolute error | 0.0978 |
| bias (mean signed) | +0.0181 |
| Pearson r | 0.9963 |

H spans [-1.120, 0.065] on this grid, so an MAE of
0.0391 is about 3% of the landscape's dynamic range. The
residual is a **positive bias** (+0.0181): hardware
systematically reports the well as shallower than it is, the expected signature of
depolarising noise pulling <ZZ> toward zero.

Mean relative error is 1.57 and is **not
reportable** - H crosses zero inside the training box, so near-zero denominators dominate
it. Use the median, MAE and RMSE.

## Step 2 - mitigation ladder (common 12-point subset)

| setting | resilience | DD | twirling | MAE | RMSE | median rel err | bias | QPU s |
|---|---|---|---|---|---|---|---|---|
| raw | 0 | off | off | 0.0794 | 0.1054 | 0.2039 | +0.0553 | 15 |
| readout mitigation | 1 | on | on | 0.0392 | 0.0459 | 0.1234 | +0.0285 | (in job 1) |
| + ZNE | 2 | on | on | 0.0277 | 0.0341 | 0.1845 | +0.0043 | 52 |

**The ladder is not flat.** MAE falls 51% from raw to resilience 1, and a
further 30% with ZNE on top - 65% overall. The clearest effect
is on **bias**, which collapses from +0.0553 to +0.0043, a factor of
13. Mitigation is removing the systematic shallowing of the well;
the residual scatter is far less affected.

Median relative error does **not** improve monotonically (0.204 -> 0.123
-> 0.184). That is the zero-crossing artefact again: ZNE's variance amplification
hurts most where |H| is smallest, so median relative error is the wrong statistic here and
is included only for completeness.

## Noise floor and depth budget

From the calibration snapshot taken at submission (`ibm_qhnn_energy_hw.json:calibration`):

| quantity | value |
|---|---|
| physical qubits used | 135, 139 |
| logical depth -> transpiled depth | 5 -> 15 |
| 2q gates per circuit | 2 (cz) |
| cz error, 135-139 | 1.01e-03 |
| median readout error | 0.0164 |
| median T1 / T2 | 245 us / 111 us |
| circuit success proxy | 0.9980 |

The error budget is **readout-dominated**. Two-qubit readout contributes
~0.0324 of error, while the two cz gates contribute ~0.00203 - a factor
of 16 apart. This explains the ladder's shape: resilience 1
(TREX readout mitigation) buys the large first step because readout is the dominant term,
while ZNE - which targets gate noise - buys a smaller second step because there are only
two cz gates to extrapolate over.

The post-ZNE MAE of 0.0277 sits at the **shot-noise floor**: the median per-point
sampling standard deviation on H at 4096 shots is 0.0245. Further mitigation cannot help
this observable at this depth; only more shots can.

## Honest limitations

- Rungs 0 and 2 rest on 12 points, not 36, because of the QPU budget. The rung-1 agreement
  between subset and full grid (0.0392 vs 0.0391) supports the subset but does not replace
  the full measurement.
- Each setting was run **once**. There are no repeat runs, so run-to-run calibration drift
  is unquantified and the ladder's step sizes carry an unmeasured systematic.
- The three rungs were submitted within ~5 minutes on one calibration snapshot; a ladder
  spread across a recalibration would likely look different.
- The local FakeMarrakesh dry run cannot validate mitigation: qiskit-ibm-runtime's local
  testing mode explicitly ignores `resilience_level`, DD and twirling. The dry run
  validated the JSON schema and the circuit binding only.

## Files

- `results/ibm_qhnn_energy_hw.json` - step 1, full 36-point grid (also ladder rung 1)
- `results/ibm_qhnn_ladder_r0_raw.json` - rung 0, raw
- `results/ibm_qhnn_ladder_r1_trex.json` - rung 1 restricted to the 12-point subset (no
  separate submission; a slice of job 1, enforced by `check_numbers.py`)
- `results/ibm_qhnn_ladder_r2_zne.json` - rung 2, ZNE
- `results/ibm_mitigation_ladder_table.csv`, `results/ibm_qhnn_energy_pointwise.csv`,
  `results/ibm_noise_floor_table.csv`
- `report/fig_hw_energy_landscape.png`, `report/fig_hw_mitigation_ladder.png`

### JSON schema

Each results file is one `RunRecord`: `name`, `backend`, `mode`, `shots`, `n_pubs`,
`job_id`, `qpu_seconds`, `resilience_level`, `dynamical_decoupling`, `twirling`,
`transpile{...}`, `calibration{...}`, `success_proxy`, `observables{evs, stds, H}`,
`reference{grid_q, grid_p, exact_zz, exact_H, theta_star, scale_s, offset_b, ...}`,
`error_analysis{zz, H, note}`, `wall_seconds`, `created`, `notes`. Ladder subset files
additionally carry `reference.subset_indices` into the full 36-point grid.
