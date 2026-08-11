"""
make_hw_figures.py
==================
Regenerate the eight hardware figures at publication size.

The defect this fixes: figures were authored at 7.3-11.7 inches wide and then
placed in a single-column ``figure`` environment (REVTeX column = 3.40 in), so
matplotlib's 8 pt labels were scaled down to 2.3-3.7 pt in the PDF.

Fix: author each figure at exactly the width it is placed at, so the scale
factor is 1.0 and the font ladder survives into the page.

  FULL_W = 7.05 in  ->  figure*  (spans both columns, \\textwidth)
  COL_W  = 3.40 in  ->  figure   (single column)

Every number is read from the results/ directory beside this script. No value is typed here.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parents[3] / "manuscript" / "figures"

FULL_W = 7.05          # REVTeX \textwidth in inches (figure*)
COL_W = 3.40           # REVTeX \columnwidth in inches (figure)

# Colour roles held fixed across every hardware figure (figure-style S4.1).
HW = "#b2182b"         # measured on hardware
EX = "#2166ac"         # exact / ideal reference
MIT = "#1a9850"        # mitigated
ALT = "#762a83"        # third series
GREY = "#8c8c8c"


def _load(name):
    return json.loads((R / f"ibm_{name}.json").read_text())


def _style():
    """Three-size ladder: base 8, annotation 7, ticks 6 (figure-style S5.2)."""
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.3, "lines.markersize": 4,
        "legend.frameon": False, "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })


def _letter(ax, s):
    ax.text(-0.22, 1.06, s, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left")


def _save(fig, stem):
    p = OUT / f"{stem}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
def fig_energy_landscape():
    d = _load("qhnn_energy_hw")
    s, b = d["reference"]["scale_s"], d["reference"]["offset_b"]
    hw = np.array([s * z + b for z in d["observables"]["evs"]])
    ex = np.array(d["reference"]["exact_H"])
    ea = d["error_analysis"]["H"]

    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 2.9))

    a = ax[0]
    lim = [min(ex.min(), hw.min()) - 0.05, max(ex.max(), hw.max()) + 0.05]
    a.plot(lim, lim, ls="--", lw=0.8, color=GREY, zorder=1)
    a.plot(ex, hw, ls="none", marker="o", ms=4, color=HW, alpha=0.85, zorder=2)
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("exact energy (statevector)")
    a.set_ylabel("measured energy (hardware)")
    a.set_title("Energy vs exact reference")
    a.text(0.04, 0.93, f"MAE {ea['mae']:.4f}\n$r$ = {ea['pearson_r']:.4f}\n$n$ = {len(ex)}",
           transform=a.transAxes, va="top", fontsize=7, color=GREY)
    _letter(a, "a")

    a = ax[1]
    resid = hw - ex
    a.axhline(0, ls="--", lw=0.8, color=GREY)
    a.plot(ex, resid, ls="none", marker="o", ms=4, color=HW, alpha=0.85)
    a.axhline(ea["bias"], ls=":", lw=1.0, color=EX)
    a.text(0.98, 0.06, f"mean bias {ea['bias']:+.4f}", transform=a.transAxes,
           ha="right", fontsize=7, color=EX)
    a.set_xlabel("exact energy (statevector)")
    a.set_ylabel("hardware $-$ exact")
    a.set_title("Residual is a systematic offset")
    a.margins(0.06)
    _letter(a, "b")

    fig.tight_layout(w_pad=1.8)
    return _save(fig, "ibm_qhnn_energy_landscape")


# --------------------------------------------------------------------------
def fig_gradients():
    d = _load("qhnn_gradients_hw")
    ref, obs, ea = d["reference"], d["observables"], d["error_analysis"]
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 2.9))
    for a, comp, key, lab in ((ax[0], "q_dot", "q_dot_vs_exact", r"$\dot q$"),
                              (ax[1], "p_dot", "p_dot_vs_exact", r"$\dot p$")):
        e = np.array(ref[f"{comp}_exact_statevector"]); h = np.array(obs[f"{comp}_hw"])
        sd = np.array(obs.get(f"{comp}_sigma", np.zeros_like(h)))
        lim = [min(e.min(), h.min()) - 0.1, max(e.max(), h.max()) + 0.1]
        a.plot(lim, lim, ls="--", lw=0.8, color=GREY, zorder=1)
        a.errorbar(e, h, yerr=sd, ls="none", marker="o", ms=3.6, color=HW,
                   ecolor=GREY, elinewidth=0.6, capsize=1.4, alpha=0.9, zorder=2)
        a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel(f"exact {lab} (statevector)")
        a.set_ylabel(f"measured {lab} (hardware)")
        m = ea[key]
        a.set_title(f"{lab} from the parameter-shift rule")
        a.text(0.04, 0.93, f"MAE {m['mae']:.4f}\n$r$ = {m['pearson_r']:.4f}",
               transform=a.transAxes, va="top", fontsize=7, color=GREY)
    _letter(ax[0], "a"); _letter(ax[1], "b")
    fig.tight_layout(w_pad=1.8)
    return _save(fig, "ibm_gradients_hw")


# --------------------------------------------------------------------------
def fig_ladder():
    rungs = [("Res. 0\nraw", "qhnn_ladder_r0_raw", GREY),
             ("Res. 1\nreadout mit.", "qhnn_ladder_r1_trex", EX),
             ("Res. 2\n+ ZNE", "qhnn_ladder_r2_zne", MIT)]
    maes, biases, labs, cols = [], [], [], []
    for lab, key, col in rungs:
        ea = _load(key)["error_analysis"]["H"]
        maes.append(ea["mae"]); biases.append(ea["bias"]); labs.append(lab); cols.append(col)

    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 3.2))
    x = np.arange(len(labs))

    a = ax[0]
    a.bar(x, maes, width=0.6, color=cols)
    for xi, v in zip(x, maes):
        a.text(xi, v + max(maes) * 0.02, f"{v:.4f}", ha="center", fontsize=7)
    a.set_xticks(x); a.set_xticklabels(labs, fontsize=7)
    a.set_ylabel("mean absolute error")
    a.set_title("Error by mitigation setting")
    a.margins(y=0.16)
    _letter(a, "a")

    a = ax[1]
    a.axhline(0, ls="--", lw=0.8, color=GREY)
    a.plot(x, biases, marker="o", ms=5, color=HW)
    for i, (xi, v) in enumerate(zip(x, biases)):
        a.annotate(f"{v:+.4f}", (xi, v), textcoords="offset points",
                   xytext=(10 if i == 0 else 0, 8),
                   ha="left" if i == 0 else "center", fontsize=7)
    a.set_xticks(x); a.set_xticklabels(labs, fontsize=7)
    a.set_ylabel("mean bias (hardware $-$ exact)")
    a.set_title("Bias by mitigation setting")
    a.margins(y=0.22)
    _letter(a, "b")

    fig.tight_layout(w_pad=1.8)
    return _save(fig, "ibm_mitigation_ladder")


# --------------------------------------------------------------------------
def fig_minl():
    d = _load("minl_hardware")["arms"]
    steps = _load("minl_hardware")["steps"]
    series = [("MINL (dissipative)", "minl_N3", HW, "o"),
              ("unitary control ($\\gamma=0$)", "unitary_ctrl_N3", EX, "s"),
              ("null control ($\\theta_k=0$)", "null_thetak0_N3", ALT, "^")]
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 2.9))

    a = ax[0]
    for lab, key, col, mk in series:
        a.plot(steps, d[key]["hw_over_ideal"], marker=mk, ms=4.2, color=col, label=lab)
    a.set_xticks(steps)
    a.set_xlabel("MINL step"); a.set_ylabel("retention (hardware / ideal)")
    a.set_title("Retention by MINL step")
    a.legend(loc="upper right", fontsize=6.5)
    a.margins(0.05)
    _letter(a, "a")

    a = ax[1]
    keys = [k for _, k, _, _ in series]
    finals = [d[k]["hw_over_ideal"][-1] for k in keys]
    depths = [d[k]["isa_depth"][-1] for k in keys]
    cols = [c for _, _, c, _ in series]
    xs = np.arange(len(keys))
    a.bar(xs, finals, width=0.6, color=cols)
    for xi, v, dp in zip(xs, finals, depths):
        a.text(xi, v + 0.02, f"{v:.3f}\ndepth {dp}", ha="center", fontsize=6.5)
    a.set_xticks(xs)
    a.set_xticklabels(["MINL", "unitary\ncontrol", "null\n$\\theta_k=0$"], fontsize=6.5)
    a.set_ylabel("retention after 4 steps")
    a.set_title("Retention after 4 steps")
    a.margins(y=0.28)
    _letter(a, "b")

    fig.tight_layout(w_pad=1.8)
    return _save(fig, "ibm_minl_hardware")


# --------------------------------------------------------------------------
def fig_mirror():
    d = _load("mirror_scaling")
    pts, fits = d["points"], d["scaling_fits"]
    cz = d["calibrated_cz_error"]
    COL = {"ring": HW, "chain": EX, "star": MIT}
    MK = {"ring": "o", "chain": "s", "star": "^"}
    FLOOR = 3e-4

    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 2.9))

    a = ax[0]
    for topo in ("ring", "chain", "star"):
        sel = sorted([p for p in pts if p["topology"] == topo], key=lambda r: r["N"])
        Ns = np.array([p["N"] for p in sel])
        F = np.array([p["fidelity_mean"] for p in sel])
        det = F > FLOOR
        a.plot(Ns[det], F[det], marker=MK[topo], ms=4.2, color=COL[topo], label=topo)
        if (~det).any():
            a.plot(Ns[~det], np.full((~det).sum(), FLOOR), marker="x", ms=4.5,
                   ls="none", color=COL[topo], alpha=0.75)
    a.axhline(FLOOR, ls=":", lw=0.8, color=GREY)
    a.text(0.98, 0.06, "$\\times$ = not detected (shot floor)", transform=a.transAxes,
           ha="right", fontsize=6.5, color=GREY)
    a.set_yscale("log"); a.set_xscale("log")
    a.set_xticks([4, 8, 16, 32, 64, 100])
    a.set_xticklabels(["4", "8", "16", "32", "64", "100"])
    a.minorticks_off()
    a.set_xlabel("network size $N$ (nodes)")
    a.set_ylabel("mirror fidelity")
    a.set_title("Fidelity vs network size")
    a.legend(loc="lower left", fontsize=6.5)
    _letter(a, "a")

    a = ax[1]
    for topo in ("ring", "chain", "star"):
        sel = [p for p in pts if p["topology"] == topo and p["fidelity_mean"] > FLOOR]
        n2 = np.array([p["two_qubit_gates"] for p in sel])
        F = np.array([p["fidelity_mean"] for p in sel])
        a.plot(n2, F, marker=MK[topo], ms=4.2, ls="none", color=COL[topo], label=topo)
        al = fits[topo]["alpha"]
        xs = np.linspace(n2.min(), n2.max(), 40)
        a.plot(xs, np.exp(-al * xs), lw=0.9, color=COL[topo], alpha=0.55)
    a.set_yscale("log")
    a.set_xlabel("transpiled two-qubit gates")
    a.set_ylabel("mirror fidelity")
    a.legend(loc="lower left", fontsize=6.5)
    a.set_title("Fidelity vs entangling gates")
    lo = min(fits[t]["alpha"] for t in ("ring", "chain", "star")) / cz
    hi = max(fits[t]["alpha"] for t in ("ring", "chain", "star")) / cz
    a.text(0.97, 0.93, f"measured / calibrated\n= {lo:.1f}$\\times$ to {hi:.1f}$\\times$",
           transform=a.transAxes, ha="right", va="top", fontsize=6.5, color=GREY)
    _letter(a, "b")

    fig.tight_layout(w_pad=1.8)
    return _save(fig, "ibm_mirror_scaling")


# --------------------------------------------------------------------------
def fig_conservation():
    d = _load("energy_conservation")
    pts, acc = d["points"], d["accuracy_by_size"]
    sizes = sorted(acc, key=lambda k: int(k))
    COLS = {"4": EX, "8": ALT, "16": HW}
    MKS = {"4": "o", "8": "s", "16": "^"}

    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, 2.7))

    a = ax[0]
    allv = []
    for N in sizes:
        sel = [p for p in pts if str(p["N"]) == N]
        e = np.array([p["exact_H"] for p in sel]); h = np.array([p["hardware_H"] for p in sel])
        allv += [e.min(), e.max(), h.min(), h.max()]
        a.plot(e, h, ls="none", marker=MKS[N], ms=4, color=COLS[N], label=f"$N={N}$")
    lim = [min(allv) * 1.05, max(allv) * 1.05]
    a.plot(lim, lim, ls="--", lw=0.8, color=GREY, zorder=0)
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("exact energy (statevector)")
    a.set_ylabel("measured energy (hardware)")
    a.set_title("Energy vs exact reference")
    a.legend(loc="upper left", fontsize=6.5)
    _letter(a, "a")

    a = ax[1]
    Ns = [int(N) for N in sizes]
    rs = [acc[N]["pearson_r"] for N in sizes]
    a.axhline(0, ls="--", lw=0.8, color=GREY)
    a.plot(Ns, rs, marker="o", ms=5, color=HW)
    for N, r in zip(Ns, rs):
        a.annotate(f"{r:+.3f}", (N, r), textcoords="offset points",
                   xytext=(0, 9 if r > 0 else -14), ha="center", fontsize=7)
    a.set_xscale("log"); a.set_xticks(Ns); a.set_xticklabels([str(n) for n in Ns]); a.minorticks_off()
    a.set_xlabel("network size $N$ (nodes)")
    a.set_ylabel("correlation with exact energy")
    a.set_title("Correlation collapses by $N=16$")
    a.margins(y=0.3)
    _letter(a, "b")

    # (c) energy drift along each trajectory, hardware vs discretised reference
    a = ax[2]
    cons = d["conservation"]
    TCOL = {"ring": HW, "chain": EX, "star": MIT}
    TMK = {"ring": "o", "chain": "s", "star": "^"}
    for topo in ("ring", "chain", "star"):
        sel = sorted([r for r in cons if r["topo"] == topo], key=lambda r: r["N"])
        Ns = [r["N"] for r in sel]
        a.plot(Ns, [100 * r["hw_drift"] for r in sel], marker=TMK[topo], ms=4.2,
               color=TCOL[topo], label=topo)
        a.plot(Ns, [100 * r["exact_drift"] for r in sel], marker=TMK[topo], ms=3.4,
               color=GREY, alpha=0.65, lw=0.9, ls=":")
    a.set_xscale("log")
    a.set_xticks([4, 8, 16]); a.set_xticklabels(["4", "8", "16"]); a.minorticks_off()
    a.set_xlabel("network size $N$ (nodes)")
    a.set_ylabel("energy drift along trajectory (%)")
    a.set_title("Drift grows with size")
    a.plot([], [], color=GREY, ls=":", lw=0.9, label="discretised ref.")
    a.legend(loc="upper left", fontsize=6.5)
    _letter(a, "c")

    fig.tight_layout(w_pad=1.8)
    return _save(fig, "ibm_energy_conservation")


# --------------------------------------------------------------------------
def fig_mitceiling():
    d = _load("mitigation_efficacy")
    cmp_ = d["comparison"]
    rows = d["rows"]
    sizes = sorted(cmp_, key=lambda k: int(k))

    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, 2.7))

    # (a) measured vs exact, resilience 1 (filled) vs resilience 2+PEA (open)
    a = ax[0]
    allv = []
    for N, col in zip(sizes, (EX, HW)):
        sel = [r for r in rows if str(r["N"]) == N]
        e = np.array([r["exact_H"] for r in sel])
        r1 = np.array([r["r1_hardware_H"] for r in sel])
        r2 = np.array([r["mit_H"] for r in sel])
        allv += [e.min(), e.max(), r1.min(), r1.max(), r2.min(), r2.max()]
        a.plot(e, r1, ls="none", marker="o", ms=4, color=col, label=f"$N={N}$, res. 1")
        a.plot(e, r2, ls="none", marker="o", ms=4, mfc="none", mew=0.9,
               color=col, label=f"$N={N}$, res. 2 + PEA")
    lim = [min(allv) * 1.06, max(allv) * 1.06]
    a.plot(lim, lim, ls="--", lw=0.8, color=GREY, zorder=0)
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("exact energy (statevector)")
    a.set_ylabel("measured energy (hardware)")
    a.set_title("Energy vs exact reference")
    a.legend(loc="upper left", fontsize=5.8)
    _letter(a, "a")

    # (b) MAE under each stack
    a = ax[1]
    x = np.arange(len(sizes)); w = 0.34
    r1 = [cmp_[N]["mae_r1"] for N in sizes]
    r2 = [cmp_[N]["mae_pea"] for N in sizes]
    a.bar(x - w / 2, r1, w, color=EX, label="resilience 1")
    a.bar(x + w / 2, r2, w, color=HW, label="resilience 2 + PEA")
    top = max(max(r1), max(r2))
    for xi, v in zip(x - w / 2, r1):
        a.text(xi, v + top * 0.02, f"{v:.2f}", ha="center", fontsize=6.5)
    for xi, v in zip(x + w / 2, r2):
        a.text(xi, v + top * 0.02, f"{v:.2f}", ha="center", fontsize=6.5)
    a.set_xticks(x); a.set_xticklabels([f"$N={N}$" for N in sizes])
    a.set_ylabel("mean absolute error")
    a.set_title("Error rises under stronger stack")
    a.legend(loc="upper left", fontsize=6.5)
    a.margins(y=0.20)
    _letter(a, "b")

    # (c) surviving signal at each ZNE amplification factor
    a = ax[2]
    surv = [r for r in d["amplified_survival"] if r["N"] == 16]
    TCOL = {"ring": HW, "chain": EX, "star": MIT}
    TMK = {"ring": "o", "chain": "s", "star": "^"}
    for rec in sorted(surv, key=lambda r: r["topology"]):
        t = rec["topology"]
        a.plot([1, 3, 5], [rec["F1"], rec["F3"], rec["F5"]], marker=TMK[t], ms=4.2,
               color=TCOL[t], label=t)
    a.axhspan(1e-6, 1e-2, color=GREY, alpha=0.16)
    a.text(0.96, 0.10, "shot-noise floor", transform=a.transAxes, ha="right",
           fontsize=6.5, color=GREY)
    a.set_yscale("log")
    a.set_xticks([1, 3, 5])
    a.set_xticklabels(["1$\\times$", "3$\\times$", "5$\\times$"])
    a.set_xlabel("noise amplification factor")
    a.set_ylabel("surviving signal fraction")
    a.set_title("Amplified signal hits the floor")
    a.legend(loc="lower left", fontsize=6.5)
    _letter(a, "c")

    fig.tight_layout(w_pad=1.8)
    return _save(fig, "ibm_mitigation_ceiling")


def fig_pendulum():
    d = _load("pendulum_chain_hardware")
    rec = d["reconstruction_rows"]
    oa, ha = d["observable_accuracy"], d["hamiltonian_accuracy"]
    sizes = sorted(oa, key=lambda k: int(k))
    COLS = {"2": EX, "4": ALT, "8": MIT, "12": HW}
    MKS = {"2": "o", "4": "s", "8": "^", "12": "D"}

    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, 2.7))

    a = ax[0]
    allv = []
    for N in sizes:
        sel = [r for r in rec if str(r["N"]) == N]
        t = np.array([r["H_true"] for r in sel]); h = np.array([r["H_hw"] for r in sel])
        allv += [t.min(), t.max(), h.min(), h.max()]
        a.plot(t, h, ls="none", marker=MKS[N], ms=4, color=COLS[N], label=f"$N={N}$")
    lim = [0, max(allv) * 1.06]
    a.plot(lim, lim, ls="--", lw=0.8, color=GREY, zorder=0)
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("classical Hamiltonian")
    a.set_ylabel("reconstructed (hardware)")
    a.set_title("Reconstructed vs classical")
    a.legend(loc="upper left", fontsize=6.5)
    _letter(a, "a")

    a = ax[1]
    Ns = [int(N) for N in sizes]
    err = [ha[N]["mean_rel_pct"] for N in sizes]
    a.plot(Ns, err, marker="o", ms=5, color=HW)
    for i, (N, v) in enumerate(zip(Ns, err)):
        a.annotate(f"{v:.2f}%", (N, v), textcoords="offset points",
                   xytext=(10 if i == 0 else 0, 9),
                   ha="left" if i == 0 else "center", fontsize=7)
    a.set_xticks(Ns); a.set_xticklabels([str(n) for n in Ns])
    a.set_xlabel("chain length $N$ (pendulums)")
    a.set_ylabel("mean relative error (%)")
    a.set_title("Relative error vs size")
    a.set_ylim(0, max(err) * 1.5)
    _letter(a, "b")

    a = ax[2]
    ss = d["shift_symmetry"]
    zz = [ss[N]["zz_only_range"] for N in sizes]
    both = [ss[N]["zz_plus_xx_range"] for N in sizes]
    a.plot(Ns, zz, marker="s", ms=4.4, color=HW, label=r"$\langle ZZ\rangle$ only")
    a.plot(Ns, both, marker="o", ms=4.4, color=EX, label=r"$\langle ZZ\rangle+\langle XX\rangle$")
    a.axhline(0, ls="--", lw=0.8, color=GREY)
    a.text(0.05, 0.55, "classical value:\nexactly invariant", transform=a.transAxes,
           fontsize=6.5, color=GREY)
    a.set_xticks(Ns); a.set_xticklabels([str(n) for n in Ns])
    a.set_xlabel("chain length $N$ (pendulums)")
    a.set_ylabel("variation under global phase shift")
    a.set_title("Shift-symmetry violation")
    a.legend(loc="upper left", fontsize=6.5)
    _letter(a, "c")

    fig.tight_layout(w_pad=1.8)
    return _save(fig, "ibm_pendulum_chain")


ALL = [fig_energy_landscape, fig_gradients, fig_ladder, fig_minl,
       fig_mirror, fig_conservation, fig_mitceiling, fig_pendulum]

if __name__ == "__main__":
    _style()
    for fn in ALL:
        p = fn()
        print(f"  {p.name}")
