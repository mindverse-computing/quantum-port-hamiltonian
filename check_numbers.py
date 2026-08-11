#!/usr/bin/env python3
"""
check_numbers.py
================
Guard: every literal quoted in the hardware-validation write-up must be
readable out of a results JSON.

The manuscript claim is that these are *measured* numbers from IBM hardware.
That claim is only as good as the link between the text and the run record, so
each entry in CLAIMS below names the file, the path inside it, and the value
the prose quotes.  Run this before any submission:

    python check_numbers.py

Exit status is non-zero if any claim drifts from its source, so this can sit in
CI.  A claim whose source file is missing is reported as a failure rather than
skipped -- a number with no traceable origin is exactly what this guards
against.

Adding a number to the manuscript means adding a row here.  If that feels like
friction, that is the point.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "experiments" / "ibm_hardware" / "results"

# (label, filename, dotted path, expected value, tolerance)
# Paths index dicts by key and lists by integer.
CLAIMS: list[tuple[str, str, str, object, float]] = [
    # --- step 1: energy landscape, full 36-point grid, resilience 1 -----------
    ("energy sweep backend",        "ibm_qhnn_energy_hw.json", "backend",      "ibm_marrakesh", 0),
    ("energy sweep job id",         "ibm_qhnn_energy_hw.json", "job_id",       "d9t66cvpemts73cug3q0", 0),
    ("energy sweep QPU seconds",    "ibm_qhnn_energy_hw.json", "qpu_seconds",  52.0, 0),
    ("energy sweep PUB count",      "ibm_qhnn_energy_hw.json", "n_pubs",       36, 0),
    ("energy sweep shots",          "ibm_qhnn_energy_hw.json", "shots",        4096, 0),
    ("energy sweep MAE",            "ibm_qhnn_energy_hw.json", "error_analysis.H.mae",            0.0391, 5e-4),
    ("energy sweep RMSE",           "ibm_qhnn_energy_hw.json", "error_analysis.H.rmse",           0.0477, 5e-4),
    ("energy sweep median rel err", "ibm_qhnn_energy_hw.json", "error_analysis.H.median_rel_err", 0.1021, 5e-4),
    ("energy sweep bias",           "ibm_qhnn_energy_hw.json", "error_analysis.H.bias",           0.0181, 5e-4),
    ("energy sweep max abs err",    "ibm_qhnn_energy_hw.json", "error_analysis.H.max_abs_err",    0.0978, 5e-4),
    ("energy sweep Pearson r",      "ibm_qhnn_energy_hw.json", "error_analysis.H.pearson_r",      0.9963, 5e-4),
    ("transpiled depth",            "ibm_qhnn_energy_hw.json", "transpile.isa_depth",             15, 0),
    ("logical depth",               "ibm_qhnn_energy_hw.json", "transpile.logical_depth",         5, 0),
    ("2q gates per circuit",        "ibm_qhnn_energy_hw.json", "transpile.isa_2q",                2, 0),

    # --- step 2: mitigation ladder, common 12-point subset -------------------
    ("ladder r0 job id",   "ibm_qhnn_ladder_r0_raw.json",  "job_id",      "d9t6897tfhrs73dthqh0", 0),
    ("ladder r0 QPU s",    "ibm_qhnn_ladder_r0_raw.json",  "qpu_seconds", 15.0, 0),
    ("ladder r0 MAE",      "ibm_qhnn_ladder_r0_raw.json",  "error_analysis.H.mae",  0.0794, 5e-4),
    ("ladder r0 RMSE",     "ibm_qhnn_ladder_r0_raw.json",  "error_analysis.H.rmse", 0.1054, 5e-4),
    ("ladder r0 bias",     "ibm_qhnn_ladder_r0_raw.json",  "error_analysis.H.bias", 0.0553, 5e-4),
    ("ladder r0 resilience", "ibm_qhnn_ladder_r0_raw.json", "resilience_level", 0, 0),

    ("ladder r1 MAE",      "ibm_qhnn_ladder_r1_trex.json", "error_analysis.H.mae",  0.0392, 5e-4),
    ("ladder r1 RMSE",     "ibm_qhnn_ladder_r1_trex.json", "error_analysis.H.rmse", 0.0459, 5e-4),
    ("ladder r1 bias",     "ibm_qhnn_ladder_r1_trex.json", "error_analysis.H.bias", 0.0285, 5e-4),
    ("ladder r1 resilience", "ibm_qhnn_ladder_r1_trex.json", "resilience_level", 1, 0),

    ("ladder r2 job id",   "ibm_qhnn_ladder_r2_zne.json",  "job_id",      "d9t68h7pemts73cug6gg", 0),
    ("ladder r2 QPU s",    "ibm_qhnn_ladder_r2_zne.json",  "qpu_seconds", 52.0, 0),
    ("ladder r2 MAE",      "ibm_qhnn_ladder_r2_zne.json",  "error_analysis.H.mae",  0.0277, 5e-4),
    ("ladder r2 RMSE",     "ibm_qhnn_ladder_r2_zne.json",  "error_analysis.H.rmse", 0.0341, 5e-4),
    ("ladder r2 bias",     "ibm_qhnn_ladder_r2_zne.json",  "error_analysis.H.bias", 0.0043, 5e-4),
    ("ladder r2 resilience", "ibm_qhnn_ladder_r2_zne.json", "resilience_level", 2, 0),

    # ladder rungs must share one point set for the comparison to mean anything
    ("ladder subset size r0", "ibm_qhnn_ladder_r0_raw.json",  "reference.n_points", 12, 0),
    ("ladder subset size r1", "ibm_qhnn_ladder_r1_trex.json", "reference.n_points", 12, 0),
    ("ladder subset size r2", "ibm_qhnn_ladder_r2_zne.json",  "reference.n_points", 12, 0),

    # --- additional literals quoted in manuscript/hardware_validation.tex ----
    ("mean rel err (quoted as not meaningful)", "ibm_qhnn_energy_hw.json",
     "error_analysis.H.mean_rel_err", 1.569, 5e-4),
    ("ladder r0 median rel err", "ibm_qhnn_ladder_r0_raw.json",
     "error_analysis.H.median_rel_err", 0.204, 5e-4),
    ("ladder r1 median rel err", "ibm_qhnn_ladder_r1_trex.json",
     "error_analysis.H.median_rel_err", 0.123, 5e-4),
    ("ladder r2 median rel err", "ibm_qhnn_ladder_r2_zne.json",
     "error_analysis.H.median_rel_err", 0.184, 5e-4),
    ("ladder r1 MAE on subset",  "ibm_qhnn_ladder_r1_trex.json",
     "error_analysis.H.mae", 0.0392, 5e-4),
    ("cz error 135-139", "ibm_qhnn_energy_hw.json",
     "calibration.two_qubit.cz_135_139", 1.013e-3, 1e-6),
    ("success proxy",    "ibm_qhnn_energy_hw.json", "success_proxy", 0.9980, 5e-4),
    ("theta* [0]",       "ibm_qhnn_energy_hw.json", "reference.theta_star.0",  1.723, 0),
    ("theta* [1]",       "ibm_qhnn_energy_hw.json", "reference.theta_star.1", -1.582, 0),
    ("theta* [2]",       "ibm_qhnn_energy_hw.json", "reference.theta_star.2",  1.170, 0),
    ("theta* [3]",       "ibm_qhnn_energy_hw.json", "reference.theta_star.3",  1.594, 0),
    ("energy scale s*",  "ibm_qhnn_energy_hw.json", "reference.scale_s",       1.335, 0),
    ("offset b*",        "ibm_qhnn_energy_hw.json", "reference.offset_b",      0.0,   0),

    # --- Sec VIII.C: symplectic gradients, both references -------------------
    ("grad job id",       "ibm_qhnn_gradients_hw.json", "job_id", "d9t67lhdsedc73aihho0", 0),
    ("grad QPU s",        "ibm_qhnn_gradients_hw.json", "qpu_seconds", 17.0, 0),
    ("grad shots",        "ibm_qhnn_gradients_hw.json", "shots", 128, 0),
    ("grad n points",     "ibm_qhnn_gradients_hw.json", "error_analysis.q_dot_vs_exact.n", 40, 0),
    ("qdot MAE vs exact", "ibm_qhnn_gradients_hw.json", "error_analysis.q_dot_vs_exact.mae",  0.0675, 5e-4),
    ("qdot RMSE vs exact","ibm_qhnn_gradients_hw.json", "error_analysis.q_dot_vs_exact.rmse", 0.0911, 5e-4),
    ("qdot r vs exact",   "ibm_qhnn_gradients_hw.json", "error_analysis.q_dot_vs_exact.pearson_r", 0.9821, 5e-4),
    ("pdot MAE vs exact", "ibm_qhnn_gradients_hw.json", "error_analysis.p_dot_vs_exact.mae",  0.0760, 5e-4),
    ("pdot RMSE vs exact","ibm_qhnn_gradients_hw.json", "error_analysis.p_dot_vs_exact.rmse", 0.0904, 5e-4),
    ("pdot r vs exact",   "ibm_qhnn_gradients_hw.json", "error_analysis.p_dot_vs_exact.pearson_r", 0.9917, 5e-4),
    ("qdot MAE vs truth", "ibm_qhnn_gradients_hw.json", "error_analysis.q_dot_vs_truth.mae",  0.1676, 5e-4),
    ("qdot RMSE vs truth","ibm_qhnn_gradients_hw.json", "error_analysis.q_dot_vs_truth.rmse", 0.2214, 5e-4),
    ("qdot r vs truth",   "ibm_qhnn_gradients_hw.json", "error_analysis.q_dot_vs_truth.pearson_r", 0.9226, 5e-4),
    ("pdot MAE vs truth", "ibm_qhnn_gradients_hw.json", "error_analysis.p_dot_vs_truth.mae",  0.1353, 5e-4),
    ("pdot RMSE vs truth","ibm_qhnn_gradients_hw.json", "error_analysis.p_dot_vs_truth.rmse", 0.1582, 5e-4),
    ("pdot r vs truth",   "ibm_qhnn_gradients_hw.json", "error_analysis.p_dot_vs_truth.pearson_r", 0.9786, 5e-4),
    ("qdot frac within 1sigma", "ibm_qhnn_gradients_hw.json", "error_analysis.q_dot_vs_exact.frac_within_1sigma", 0.525, 5e-4),
    ("pdot frac within 1sigma", "ibm_qhnn_gradients_hw.json", "error_analysis.p_dot_vs_exact.frac_within_1sigma", 0.35, 5e-4),
    ("ZZ contraction slope",     "ibm_qhnn_gradients_hw.json", "error_analysis.raw_zz_vs_exact.contraction_slope", 0.9829, 5e-4),

    # --- Sec VIII.J: mitigation depth ceiling --------------------------------
    ("mitceiling job id",  "ibm_mitigation_efficacy.json", "job_id", "d9tkas7pemts73cv1kgg", 0),
    ("mitceiling QPU s",   "ibm_mitigation_efficacy.json", "qpu_seconds", 330.0, 0),
    ("mitceiling n pubs",  "ibm_mitigation_efficacy.json", "n_pubs", 18, 0),
    ("mitceiling amplifier","ibm_mitigation_efficacy.json","mitigation_applied.amplifier", "pea", 0),
    ("mitceiling DD seq",  "ibm_mitigation_efficacy.json", "mitigation_applied.dynamical_decoupling", "XY4", 0),
    ("mitceiling randomizations", "ibm_mitigation_efficacy.json", "mitigation_applied.num_randomizations", 32, 0),
    ("N=8 MAE resilience 1",  "ibm_mitigation_efficacy.json", "comparison.8.mae_r1",  1.7556, 5e-4),
    ("N=8 MAE PEA",           "ibm_mitigation_efficacy.json", "comparison.8.mae_pea", 1.9522, 5e-4),
    ("N=8 r resilience 1",    "ibm_mitigation_efficacy.json", "comparison.8.r_r1",    0.9251, 5e-4),
    ("N=8 r PEA",             "ibm_mitigation_efficacy.json", "comparison.8.r_pea",   0.6803, 5e-4),
    ("N=16 MAE resilience 1", "ibm_mitigation_efficacy.json", "comparison.16.mae_r1",  8.6383, 5e-4),
    ("N=16 MAE PEA",          "ibm_mitigation_efficacy.json", "comparison.16.mae_pea",11.7325, 5e-4),
    ("N=16 r resilience 1",   "ibm_mitigation_efficacy.json", "comparison.16.r_r1",   -0.2793, 5e-4),
    ("N=16 r PEA",            "ibm_mitigation_efficacy.json", "comparison.16.r_pea",  -0.2826, 5e-4),

]

# Derived quantities quoted in the manuscript that are not stored verbatim.
# Each is recomputed from the JSON so the prose cannot drift from the source.
# (label, filename(s), callable(load) -> float, quoted value, tolerance)
DERIVED = [
    # --- Sec VIII.K: pendulum sweep, worst-case correlation bound ------------
    ("pendulum min observable r", "ibm_pendulum_chain_hardware.json",
     lambda L: min(v["pearson_r"] for v in
                   L("ibm_pendulum_chain_hardware.json")["observable_accuracy"].values()), 0.99977, 5e-5),
    # --- Sec VIII.H: measured alpha vs calibrated cz error -------------------
    ("mirror alpha/cz min ratio", "ibm_mirror_scaling.json",
     lambda L: min(v["alpha"] for k, v in L("ibm_mirror_scaling.json")["scaling_fits"].items()
                   if k != "_pooled") / L("ibm_mirror_scaling.json")["calibrated_cz_error"], 2.77, 0.02),
    ("mirror alpha/cz max ratio", "ibm_mirror_scaling.json",
     lambda L: max(v["alpha"] for k, v in L("ibm_mirror_scaling.json")["scaling_fits"].items()
                   if k != "_pooled") / L("ibm_mirror_scaling.json")["calibrated_cz_error"], 7.68, 0.02),
    ("H landscape minimum",  "ibm_qhnn_energy_hw.json",
     lambda L: min(L("ibm_qhnn_energy_hw.json")["reference"]["exact_H"]), -1.120, 5e-4),
    ("H landscape maximum",  "ibm_qhnn_energy_hw.json",
     lambda L: max(L("ibm_qhnn_energy_hw.json")["reference"]["exact_H"]),  0.065, 5e-4),
    ("MAE reduction r0->r1 (%)", "ladder",
     lambda L: 100 * (1 - L("ibm_qhnn_ladder_r1_trex.json")["error_analysis"]["H"]["mae"]
                      / L("ibm_qhnn_ladder_r0_raw.json")["error_analysis"]["H"]["mae"]), 51, 0.6),
    ("MAE reduction r1->r2 (%)", "ladder",
     lambda L: 100 * (1 - L("ibm_qhnn_ladder_r2_zne.json")["error_analysis"]["H"]["mae"]
                      / L("ibm_qhnn_ladder_r1_trex.json")["error_analysis"]["H"]["mae"]), 30, 0.6),
    ("MAE reduction r0->r2 (%)", "ladder",
     lambda L: 100 * (1 - L("ibm_qhnn_ladder_r2_zne.json")["error_analysis"]["H"]["mae"]
                      / L("ibm_qhnn_ladder_r0_raw.json")["error_analysis"]["H"]["mae"]), 65, 0.6),
    ("bias reduction factor r0/r2", "ladder",
     lambda L: (L("ibm_qhnn_ladder_r0_raw.json")["error_analysis"]["H"]["bias"]
                / L("ibm_qhnn_ladder_r2_zne.json")["error_analysis"]["H"]["bias"]), 13, 0.5),
    ("two-qubit readout error", "ibm_qhnn_energy_hw.json",
     lambda L: 1 - (1 - _median_readout(L("ibm_qhnn_energy_hw.json"))) ** 2, 0.0324, 5e-4),
    ("median single-qubit readout error", "ibm_qhnn_energy_hw.json",
     lambda L: _median_readout(L("ibm_qhnn_energy_hw.json")), 0.0164, 5e-4),
    ("readout / gate error ratio", "ibm_qhnn_energy_hw.json",
     lambda L: (1 - (1 - _median_readout(L("ibm_qhnn_energy_hw.json"))) ** 2)
               / (1 - (1 - L("ibm_qhnn_energy_hw.json")["calibration"]["two_qubit"]["cz_135_139"]) ** 2),
     16, 0.5),
    ("median T1 (us)", "ibm_qhnn_energy_hw.json",
     lambda L: _median_qubit(L("ibm_qhnn_energy_hw.json"), "t1_us"), 245, 0.5),
    ("median T2 (us)", "ibm_qhnn_energy_hw.json",
     lambda L: _median_qubit(L("ibm_qhnn_energy_hw.json"), "t2_us"), 111, 0.5),
    ("median shot-noise std on H", "ibm_qhnn_energy_hw.json",
     lambda L: _median(
         [s * L("ibm_qhnn_energy_hw.json")["reference"]["scale_s"]
          for s in L("ibm_qhnn_energy_hw.json")["observables"]["stds"] if s]), 0.0245, 5e-4),
    ("subset vs full-grid MAE gap", "ladder",
     lambda L: abs(L("ibm_qhnn_ladder_r1_trex.json")["error_analysis"]["H"]["mae"]
                   - L("ibm_qhnn_energy_hw.json")["error_analysis"]["H"]["mae"]), 0.0002, 5e-5),
]


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _median_readout(rec: dict) -> float:
    return _median([v["readout_error"] for v in rec["calibration"]["qubits"].values()
                    if v.get("readout_error") is not None])


def _median_qubit(rec: dict, key: str) -> float:
    return _median([v[key] for v in rec["calibration"]["qubits"].values() if v.get(key)])


# The ledger is shared by every track in the campaign, so its running totals
# move when a sibling submits.  Pinning them to a literal would make this guard
# fail on someone else's job, so cost is checked per-track below instead: the
# three jobs this track owns, and the invariant that the account is not
# overspent.
TRACK_JOBS = {
    "qhnn_energy_r1_trex": 52.0,
    "qhnn_energy_r0_raw":  15.0,
    "qhnn_energy_r2_zne":  52.0,
}
TRACK_QPU_SECONDS = 119.0

# Cross-file invariants: (label, callable(loader) -> bool, description)
def _invariants(load):
    def same_subset():
        a = load("ibm_qhnn_ladder_r0_raw.json")["reference"]["subset_indices"]
        b = load("ibm_qhnn_ladder_r1_trex.json")["reference"]["subset_indices"]
        c = load("ibm_qhnn_ladder_r2_zne.json")["reference"]["subset_indices"]
        return a == b == c

    def ladder_monotone():
        m = [load(f)["error_analysis"]["H"]["mae"] for f in
             ("ibm_qhnn_ladder_r0_raw.json", "ibm_qhnn_ladder_r1_trex.json",
              "ibm_qhnn_ladder_r2_zne.json")]
        return m[0] > m[1] > m[2]

    def r1_subset_matches_parent():
        """The r1 subset record must be a literal slice of the full-grid job."""
        full = load("ibm_qhnn_energy_hw.json")
        sub = load("ibm_qhnn_ladder_r1_trex.json")
        idx = sub["reference"]["subset_indices"]
        return [full["observables"]["H"][i] for i in idx] == sub["observables"]["H"]

    def hardware_mode():
        return all(load(f)["mode"] == "hardware" for f in
                   ("ibm_qhnn_energy_hw.json", "ibm_qhnn_ladder_r0_raw.json",
                    "ibm_qhnn_ladder_r2_zne.json"))

    def track_cost():
        """This track's three jobs are in the ledger at the costs quoted."""
        led = load("ibm_qpu_ledger.json")
        by_name = {j["name"]: j["qpu_seconds"] for j in led["jobs"]}
        if any(by_name.get(k) != v for k, v in TRACK_JOBS.items()):
            return False
        return abs(sum(TRACK_JOBS.values()) - TRACK_QPU_SECONDS) < 1e-9

    def budget_not_overspent():
        led = load("ibm_qpu_ledger.json")
        return 0 <= led["spent_seconds"] <= led["monthly_seconds"]

    return [
        ("this track's 3 jobs are in the ledger at the quoted cost", track_cost),
        ("account is not overspent against the monthly allowance", budget_not_overspent),
        ("all three ladder rungs use the same 12 grid points", same_subset),
        ("ladder MAE is strictly monotone decreasing", ladder_monotone),
        ("r1 subset record is a slice of the full-grid job", r1_subset_matches_parent),
        ("every quoted run is mode=hardware, not simulation", hardware_mode),
    ]


def dig(obj, path: str):
    for part in path.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def main() -> int:
    cache: dict[str, dict] = {}

    def load(fn: str) -> dict:
        if fn not in cache:
            cache[fn] = json.loads((RESULTS / fn).read_text())
        return cache[fn]

    fails: list[str] = []
    for label, fn, path, expected, tol in CLAIMS:
        try:
            got = dig(load(fn), path)
        except FileNotFoundError:
            fails.append(f"{label}: {fn} missing -- number has no traceable source")
            continue
        except (KeyError, IndexError, ValueError):
            fails.append(f"{label}: path '{path}' not present in {fn}")
            continue
        ok = (abs(float(got) - float(expected)) <= tol
              if isinstance(expected, (int, float)) and not isinstance(expected, bool)
              else got == expected)
        if not ok:
            fails.append(f"{label}: quoted {expected!r}, {fn}:{path} holds {got!r}")

    for label, _src, fn, expected, tol in DERIVED:
        try:
            got = fn(load)
        except Exception as exc:                                   # noqa: BLE001
            fails.append(f"{label}: could not recompute from JSON ({exc})")
            continue
        if abs(float(got) - float(expected)) > tol:
            fails.append(f"{label}: quoted {expected!r}, JSON gives {got!r}")

    for label, fn in _invariants(load):
        try:
            if not fn():
                fails.append(f"invariant violated: {label}")
        except Exception as exc:                                   # noqa: BLE001
            fails.append(f"invariant uncheckable ({label}): {exc}")

    n = len(CLAIMS) + len(DERIVED) + 6
    if fails:
        print(f"FAIL  {len(fails)} of {n} checks\n")
        for f in fails:
            print("  -", f)
        return 1
    print(f"OK    {n} checks passed; every quoted number traces to a results JSON")
    return 0


if __name__ == "__main__":
    sys.exit(main())
