# Hardware validation and quantum error analysis — source document

Consolidated record for the manuscript's hardware section (Section VIII). Every
number here is read from a results JSON in `codes/results/`; nothing is retyped
from memory. `check_numbers.py` enforces that the manuscript's literals agree
with these files.

## Provenance

All measurements on `ibm_marrakesh`, a 156-qubit Heron r2 processor.
12 jobs, 670 s of QPU time.

**Budget note.** The campaign total exceeds the 600 s monthly IBM Open-plan
allowance by 70 s. The cause is the PEA-amplified ZNE run
(`ibm_mitigation_efficacy.json`), which charged 330 s against a 90 s estimate:
PEA performs a noise-learning pass per entangling layer before amplifying, and
budgeting it as a multiple of gate-folding ZNE understates it substantially at
depth. Recorded rather than absorbed.

Each record carries the job id, the calibration snapshot taken at submission,
the transpile report, raw expectation values, and the exact reference. Devices
drift, so a snapshot taken later is not the one that produced the numbers.

## Result 1 — the learned energy surface transfers to hardware

36 phase-space points at the manuscript's trained parameters, scored against
exact statevector values of the same model.

| metric | value |
|---|---|
| MAE | 0.0391 |
| RMSE | 0.0477 |
| Pearson r | 0.9963 |
| signed bias | +0.0181 |

The residual is **not zero-mean**. A depolarising channel contracts any Pauli
expectation toward zero, so a negative correlator is measured as less negative
and the energy well is reported shallower than it is. This biases the level sets
and hence the gradients taken from them, rather than adding scatter that averages
away.

**Metric caution.** The energy crosses zero inside the training box, so relative
error acquires near-zero denominators: the mean relative error is 1.569 and is
not a meaningful summary. Median relative error (0.1021) and the
absolute measures are what should be quoted.

## Result 2 — the parameter-shift rule returns Hamilton's equations

Both symplectic components at 40 points, 128 shots per evaluation. All shifted
evaluations ride one PUB on one transpiled ISA circuit, so the members of each
shift pair share gate sequence and layout and coherent error partially cancels
in the difference.

| comparison | MAE | RMSE | Pearson r |
|---|---|---|---|
| q̇ vs exact statevector | 0.0675 | 0.0911 | 0.9821 |
| ṗ vs exact statevector | 0.0760 | 0.0904 | 0.9917 |
| q̇ vs true pendulum field | 0.1676 | 0.2214 | 0.9226 |
| ṗ vs true pendulum field | 0.1353 | 0.1582 | 0.9786 |

**These two references must not be conflated.** Error against the exact
statevector is the hardware error. The larger error against the true field is the
4-parameter ansatz's own approximation error, already present in simulation.
Only the first is a statement about the device.

**Is the residual sampling noise?** Nearly, not quite: χ²/dof = 1.37 (q̇) and
1.55 (ṗ) against the binomial floor, leaving a device contribution of about
0.048 and 0.054. Pull statistics agree — 52.5% and 35% of points
within 1σ against the 68% expected under pure shot noise. The underlying
correlator shows contraction slope 0.9829, a 1.7% systematic
shrinkage propagating through the shift rule.

## Result 3 — the mitigation ladder

Three settings on a common 12-point subset.

| rung | MAE | bias |
|---|---|---|
| resilience 0 (raw) | 0.0794 | +0.0553 |
| resilience 1 (TREX + DD + twirling) | 0.0392 | +0.0285 |
| resilience 2 (+ ZNE) | 0.0277 | +0.0043 |

Total reduction 65.2%. Mitigation acts primarily on **bias**, which
collapses 13×, not on scatter.

## Error budget — why the ladder has that shape

Decomposed from the calibration snapshot for the transpiled circuit (depth 15,
2 cz gates):

| source | contribution to H error |
|---|---|
| readout (2 qubits) | 0.03266 |
| two-qubit gates (2 cz) | 0.00203 |
| shot noise (4096 shots) | 0.0245 |

**Readout-dominated by a factor of 16.** This explains the ladder
directly: resilience 1 is TREX readout mitigation, which targets the dominant
term and buys the large first step; ZNE targets gate noise and has only two cz
gates to extrapolate over, so it buys a smaller second step.

The post-ZNE MAE of 0.0277 sits at the shot floor of 0.0245.
**Further mitigation cannot improve this observable at this depth — only more
shots can.** At shallow depth, spend budget on shots and TREX, not on ZNE.

## Result 4 — the dissipative channel, and why it is a negative result

The MINL channel was built as a native dynamic circuit (mid-circuit measurement,
`if_test` feedforward, ancilla reuse) and validated offline against the
statevector reference to within sampling error.

On hardware at N=3, retention relative to ideal after 4 steps:

| arm | retention | ISA depth |
|---|---|---|
| MINL (dissipative kick) | 0.034 | 361 |
| unitary control (γ=0) | 0.636 | 172 |
| depth-matched null (θ_k=0) | 0.009 | 369 |

The observed suppression far exceeds the noiseless prediction — but the
depth-matched null control, which keeps the measurement machinery and nulls only
the kick, decays at least as much. **The dissipation is not separable from
measurement back-action at these depths.** The channel executes natively; its
physical effect has not been demonstrated.

## Result 5 — mirror benchmarking sets the network ceiling

Mirror circuits (U followed by U†) make the return probability the fidelity
itself, needing no classical reference, so they run where simulation cannot.

Measured per-two-qubit-gate decay F = exp(−α n₂q):

| topology | α | effective error per gate |
|---|---|---|
| ring | 0.01141 | 0.0113 |
| chain | 0.01773 | 0.0176 |
| star | 0.00640 | 0.0064 |

Against a calibrated cz error of 0.00231, the measured effective error
is 2.8–7.7× larger. **The gate-count success proxy is optimistic by a wide
margin**, and the usable network size is single-digit N, not the 64–100 the
proxy predicted. A repeat wave bounds run-to-run drift at 4.0% mean relative.

## Result 6 — energy conservation, and where it fails

Conservative network energy against exact statevector references:

| N | MAE | mean rel. | Pearson r |
|---|---|---|---|
| 4 | 0.2824 | 4.64% | +0.9888 |
| 8 | 1.7556 | 17.53% | +0.9251 |
| 16 | 8.6383 | 36.21% | -0.2793 |

At N=16 the measured energies are **uncorrelated with truth**. The energy
observable is a sum of O(N) Pauli terms, so its variance grows with N while the
signal does not, and each term carries the dominant readout error. This and the
mirror benchmark are independent measurements that place the ceiling in the same
region.

## Result 7 — error mitigation has a depth ceiling

The N=8 and N=16 conservation points re-measured on **identical states**,
changing only the mitigation stack to resilience 2 + PEA amplifier + XY4 DD +
32 twirling randomisations. States were regenerated from the seeded recipe and
verified against stored exact energies to 0.0.

| N | MAE resilience 1 | MAE PEA | change | r before | r after |
|---|---|---|---|---|---|
| 8 | 1.7556 | 1.9522 | +11.2% | 0.9251 | 0.6803 |
| 16 | 8.6383 | 11.7325 | +35.8% | -0.2793 | -0.2826 |

**The stronger stack made both sizes worse.** ZNE extrapolates to zero noise
from runs at 1×, 3× and 5× amplification; at the mirror-measured decay rates the
N=16 circuits retain 0.05%–6.5% of signal at 3× and less at 5×, so the fit is
anchored on uninformative points and the extrapolation is unconstrained.

Two consequences. The N=16 failure is **not** an artifact of insufficient
mitigation, which strengthens the ceiling claim. And the ladder's own 65% gain
is scoped to depth 15; it does not extend to depth 636.

## Result 8 — coupled nonlinear pendulums without training

Exploiting the structural isomorphism (coefficients follow from K and g
directly) plus the identity cos(φᵢ−φⱼ) = ⟨ZᵢZⱼ⟩ + ⟨XᵢXⱼ⟩, the Hamiltonian is
measurable with no trained model, at depth 4.

| N | H MAE | mean rel. |
|---|---|---|
| 2 | 0.0228 | 2.53% |
| 4 | 0.0313 | 1.06% |
| 8 | 0.0386 | 1.39% |
| 12 | 0.0485 | 0.43% |

Accuracy is **flat in N** — the opposite of every trained-circuit result above,
because depth rather than qubit count is the binding constraint. This also
exposed a limitation of the manuscript's read-out: the published ⟨ZZ⟩ observable
omits the XX term required for shift-invariant coupling.

## Limitations

- Ladder rungs 0 and 2 rest on 12 points, not 36, because of the QPU budget
- Every configuration was executed once; run-to-run drift is bounded only by the
  4.0% mirror figure, and the mitigation comparison inherits that on both arms
- All three ladder rungs share one calibration snapshot (~5 min window)
- Local fake backends cannot validate mitigation — runtime local mode ignores
  `resilience_level`, DD and twirling
- Gradient run used 128 shots/eval; χ²/dof 1.37 and 1.55 means near but not
  within the shot floor
- The mitigation stack changed as a bundle, so the degradation is not attributed
  to a single component; N=4 was not re-measured, so the crossover depth is
  bracketed between 15 and 636, not located
- MINL dissipation is a negative result: executed natively, not separated from
  measurement back-action

## Files

Results: `ibm_qhnn_energy_hw.json`, `ibm_qhnn_gradients_hw.json`,
`ibm_qhnn_ladder_{r0_raw,r1_trex,r2_zne}.json`, `ibm_minl_hardware.json`,
`ibm_mirror_scaling.json`, `ibm_energy_conservation.json`,
`ibm_pendulum_chain_hardware.json`, `ibm_mitigation_efficacy.json`,
`ibm_master_results.json`, `ibm_qpu_ledger.json`

Guard: `check_numbers.py`
