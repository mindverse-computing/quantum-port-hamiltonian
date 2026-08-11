# Next plan — upgrading the Q-pHNN manuscript

Written 2026-08-11, at the end of the first hardware campaign. This is the
starting point for the next session: what the manuscript currently claims, what
is weak in it, and what to run — in priority order, with costs from a model
calibrated on twelve real jobs rather than estimates.

Read this together with `report/hardware_error_analysis.md` (the consolidated
error analysis, current as of the same date) and `results/ibm_qpu_ledger.json`
(authoritative spend).

---

## 0. State at hand-off

**Manuscript.** 57 pages, Section VIII "Hardware Validation" with eleven
subsections. Compiles clean: 0 errors, 0 undefined citations, 0 undefined
references, 0 overfull boxes, abstract 199 words. arXiv bundle
(`qphnn_arxiv.tar.gz`) verified to compile from `main.bbl` with no `.bib`
present.

**Hardware campaign.** 12 jobs on `ibm_marrakesh` (156-qubit Heron r2),
670 s of QPU.

**Budget: OVER by 70 s** against the 600 s monthly IBM Open-plan allowance.
`check_numbers.py` fails 1 of 98 checks on exactly this invariant, deliberately
left failing. The allowance resets monthly — confirm the reset before planning
any run below.

**Guard.** `check_numbers.py`, 98 checks. Every literal in the hardware sections
is tied to a results JSON. Adding a number to the manuscript means adding a row
here.

### What hardware has established

| Result | Section | Status |
|---|---|---|
| Trained energy surface transfers | VIII.B | MAE 0.0391, r 0.9963 |
| Parameter-shift gradients work | VIII.C | r 0.982 / 0.992 |
| Mitigation ladder | VIII.D | 65% reduction at depth 15 |
| Error budget readout-dominated | VIII.E | 16x, post-ZNE at shot floor |
| MINL executes natively | VIII.G | **negative**: not separable from back-action |
| Mirror fidelity scaling | VIII.H | usable N is single-digit, proxy optimistic 2.8-7.7x |
| Energy conservation vs N | VIII.I | works N=4, fails N=16 (r = -0.28) |
| Mitigation has a depth ceiling | VIII.J | **negative**: PEA makes N=8/16 worse |
| Coupled pendulums, training-free | VIII.K | flat accuracy to N=12, r >= 0.9997 |

### Calibrated cost model — use this, not intuition

From all 12 jobs, at 4096 shots:

| configuration | s/PUB | multiplier |
|---|---|---|
| resilience 0-1 | 1.3-1.5 | 1x |
| resilience 2, gate-folding ZNE | 4.3 | ~3.5x |
| **resilience 2 + PEA amplifier** | **18.3** | **~14x** |

The PEA row is the lesson of the overrun: it performs a noise-learning pass per
entangling layer before amplifying, and budgeting it as a multiple of plain ZNE
understated it by 3.7x. Low shot counts do **not** save money — the 128-shot
gradient job cost 17 s for a single PUB, because per-job overhead dominates
there. Batch aggressively; queue latency (32-711 s wall observed) is a bigger
practical constraint than QPU seconds.

---

## 1. Priority A — reproducibility (do this first)

**Every hardware configuration in the manuscript was executed once.** This is
the single most likely reviewer objection, and it currently appears as a stated
limitation in three places. The only reproducibility bound anywhere in the paper
is the 4.0% mean relative drift from the mirror repeat wave, and it is borrowed
by results that never measured their own.

### A1. Repeat the energy sweep — 36 PUBs, ~52 s

Re-run `ibm_qhnn_energy_hw.json` at identical parameters on a different day.
Report per-point drift and whether the +0.0181 bias reproduces. This converts
"executed once" into a measured stability figure for the paper's headline
hardware result.

*Why first:* cheapest possible conversion of a limitation into a measurement,
and it strengthens every downstream claim that rests on the energy surface.

### A2. Repeat the conservation sweep — 27 PUBs, ~43 s

Same for `ibm_energy_conservation.json`. The N=16 failure is a load-bearing
claim (it sets the ceiling); showing it reproduces makes it a result rather than
an anecdote.

### A3. Ladder rungs on the full 36-point grid — 24 PUBs, ~35 s

Rungs 0 and 2 currently rest on a 12-point subset because of budget. Running
them on the full grid removes a stated limitation and lets the 65% figure be
quoted without a subset caveat.

**Priority A total: ~130 s.** Everything here is resilience <= 2 without PEA.

---

## 2. Priority B — locate the mitigation crossover

Section VIII.J shows the ladder helping at depth 15 (+65%) and hurting at depth
636 (-36%). The crossover is **bracketed but not located**, which the text
admits. Locating it turns a negative result into a design rule.

### B1. Depth ladder at fixed N — ~30 PUBs, ~130 s at resilience 2

Take one topology (chain, whose measured alpha 0.01773 is the worst case) and
run the conservation observable at Trotter steps 1, 2, 3, 4 at N=4 and N=8, each
at resilience 1 and resilience 2 (gate-folding ZNE, **not** PEA — the amplifier
is what blew the budget and is not needed to find a crossover).

Deliverable: a curve of mitigation benefit against transpiled depth, crossing
zero at a measurable depth. That is a genuinely useful number for anyone running
structure-preserving circuits on this class of device, and it is the natural
figure to pair with VIII.J.

### B2. Component attribution — ~18 PUBs, ~78 s

VIII.J changed the mitigation stack as a *bundle* (PEA + XY4 + 32
randomisations), so the degradation is not attributed to a component. Re-run the
N=8 points varying one factor at a time: gate-folding ZNE alone, XY4 alone,
randomisations alone. States are already reproducible from the seeded recipe
(verified to 0.0 — see `ibm_mitigation_efficacy.json`).

*Skip if budget is tight.* B1 is the scientifically valuable half.

---

## 3. Priority C — close the open scientific threads

### C1. The XX read-out limitation (no QPU required)

The manuscript states that the published `<ZZ>` read-out omits the `<XX>` term
required for shift-invariant coupling, with the exact identity in Eq. (zzxx).
**It does not fix it.**

Next step is a code change, not a run: extend `QGNNEnergy._obs` to carry both
terms, retrain in simulation at N=3,6,9, and compare against the current
ZZ-only models on the paper's existing metrics. If the corrected read-out
improves energy conservation or the monotone fraction, that is a genuine
contribution — the framework identifying and repairing its own limitation.
If it does not, that is worth reporting too.

*Cost: zero QPU, a few hours of local compute at those sizes.*

### C2. Pendulum chain with gravity — training required

`network/pendulum_chain.py` is written and validated (g=0 reproduces Kuramoto
exactly; momentum conservation breaks exactly when g>0). What is missing is
trained models: **the training run was lost** because the save happened after
the final loop iteration rather than after each configuration.

Two things to fix before retrying:

1. Save after every configuration, not at the end.
2. Do not train on the session machine. Measured cost at N=4 is ~14 h with the
   current numerical-gradient BFGS; N=12 is ~68 h. This needs a cluster or an
   overnight run.

Note that the *analytic* pendulum measurement (VIII.K) already covers N=2 to 12
without training, so C2 buys the **learned** case, not the physics. Lower value
than it first appears.

### C3. MINL separability — a dose-response sweep, ~15 PUBs, ~20 s

**This supersedes an earlier recommendation to park the question.** Re-reading
the three arms of `ibm_minl_hardware.json` shows the campaign never ran the
experiment that would actually discriminate, and it is cheap.

What we have at N=3, retention relative to ideal after 4 steps:

| arm | measurement machinery | dissipative kick | ISA depth | retention |
|---|---|---|---|---|
| unitary control | no | no | 172 | 0.636 |
| null (θ_k = 0) | yes | no | 369 | 0.009 |
| MINL | yes | yes | 361 | 0.034 |

The null arm is the informative one. It carries the full mid-circuit
measure/reset/feedforward structure with the kick set to identity, and it
collapses to 0.009 while the measurement-free unitary arm holds 0.636. **The
measurement machinery alone destroys the signal**, and MINL sits in the same
collapsed regime (marginally above the null, not below it — the ordering
expected if the kick were adding resolvable damping is not observed).

Comparing MINL against the *unitary* control therefore confounds two variables
at once: presence of measurement, and presence of the kick. That comparison
cannot resolve dissipation no matter how many shots it gets.

**The discriminating design is a dose-response sweep.** Hold the measurement
schedule, circuit depth, and step count *exactly* fixed, and vary only the kick
angle θ_k over roughly 5 values from 0 to π/2, at 3 step counts. Every
back-action contribution is then common-mode across the sweep and divides out;
the only thing changing is dissipation strength.

- If retention falls monotonically with θ_k, the dissipative channel is
  resolved as a physical effect, and the slope is a measurement of it.
- If retention is flat in θ_k, dissipation is genuinely unresolvable at this
  depth, and VIII.G's negative result is upgraded from "not separated" to
  "measured to be below the noise floor, with this bound."

Both outcomes are publishable and the second is a stronger statement than the
paper currently makes. Cheap enough to run first, before any device change is
considered.

*Only if the sweep comes back flat* does this become a hardware question —
better mid-circuit measurement fidelity, or an encoding in which the two
contributions come apart.

---

## 4. Priority D — presentation upgrades (no QPU)

- **Figure 1 is missing.** There is no schematic that shows pH structure ->
  circuit topology -> measured hardware result in one view. Every other figure
  is a results panel. A reader currently has to assemble the argument themselves.
- **Section VIII is eleven subsections and 28 pages.** Consider splitting the
  narrative results (B, C, D, E) from the scaling/limits results (H, I, J, K),
  or promoting the negative results to their own subsection with a shared
  framing — they are the paper's most defensible content and are currently
  scattered.
- **Related work stops at the pre-hardware literature.** Search for 2025-2026
  work on structure-preserving circuits on real devices; the hardware section
  makes the paper contemporary and the citations should reflect that.
- **Data availability statement.** All twelve job IDs are in the ledger; state
  that results JSONs and the guard are available, which reviewers increasingly
  expect for hardware claims.

---

## 5. Recommended sequence for the next session

Assuming a fresh 600 s allowance:

1. **Confirm the allowance reset** and that `check_numbers.py` passes 98/98
   once the ledger rolls over. Do not submit anything before this.
2. **Priority A (~130 s)** — three repeat runs, back to back, batched into as
   few jobs as possible. Update the three "executed once" passages.
3. **C3 dose-response (~20 s)** — cheapest discriminating experiment in the
   whole plan, and it targets the paper's central claim. Run it early.
4. **B1 (~130 s)** — the depth ladder. New subsection plus one figure.
5. **C1 (no QPU)** — the XX read-out fix, in parallel with the above since it
   is local compute.
6. **Priority D** — presentation, using whatever remains of the session.

Leaves ~340 s of reserve. Given that the last session overran on a job estimated
at a quarter of its true cost, keep that reserve genuinely unspent until the
first three runs have landed and their charges are recorded.

### Standing rules that earned their place

- Dry-run every circuit on a fake backend before submitting. Two real bugs were
  caught this way at zero cost (an empty classical register at steps=0; an
  initial state that made both MINL arms rise instead of decay).
- Record what mitigation was *applied*, not requested. `hardware_harness.py`
  now raises rather than spending QPU when an option is rejected — three early
  records lack this provenance and cannot be reconstructed.
- Reproduce states from a seeded recipe and verify against stored exact values
  before re-measuring. This is what made VIII.J a controlled comparison.
- Every number in the manuscript gets a row in `check_numbers.py`.
- Report negative results as negative. The two strongest contributions in
  Section VIII (MINL non-separability, the mitigation depth ceiling) are both
  failures to observe an expected effect.
