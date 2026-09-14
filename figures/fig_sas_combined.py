#!/usr/bin/env python3
"""Figure 2. Mutator allele frequencies and selection-coefficient likelihoods under the South
Asian demographic history. Sized for a 6.5-inch text width.

  Panel A     Simulated frequency distribution of each known mutator (box = IQR,
              whiskers = 2.5th and 97.5th percentile), with the gnomAD v4.1.1
              South Asian frequency overlaid: a solid line at k/n and a shaded
              exact binomial 95% CI. Where the variant was not observed only the
              band is drawn, its top edge the one-sided 95% upper bound.
  Panels B-F  Relative log-likelihood of the selection coefficient of each
              mutator, in the same gene order as panel A. The x-axis is h s* for
              POLE and POLD1 and s* for MPG, XPC and MUTYH. MUTYH is the joint
              likelihood over Y179C, V234M and G368D.

Distributions include every simulated replicate; a replicate in which the
mutator was lost enters at a frequency of exactly 0. A log axis cannot show 0, so a
box whose 2.5th percentile, lower quartile or median is 0 is drawn at the axis
floor with a caret beneath it and named on stdout with the fraction of
replicates at zero. MUTYH variant labels sit above their boxes.

Reads   results/sas/simulations/          (workflow/const_phiG.smk, pop=sas)
        results/sas/simulations_varying_s/ (workflow/varying_selection.smk)
Writes  SAS_mutators_freq_and_profile_likelihoods.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (                       # noqa: E402
    RC as rc, GNOMAD_COLOR, BOX_FACECOLOR, BOX_ALPHA, CI_ALPHA,
    MUTYH_VARIANTS_ALL, clopper_pearson, freq_and_ci, binom_logpmf,
    summary_is_unconditional, FloorTracker, load_gnomad, load_demography,
    results_root, figures_root,
)

GNOMAD_COUNTS = load_gnomad()

import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, LogLocator, LogFormatterMathtext
from scipy.stats import beta, chi2
from scipy.special import logsumexp, gammaln, xlogy, xlog1py

SIM_DIR  = f"{results_root()}/sas/simulations"
VARY_DIR = f"{results_root()}/sas/simulations_varying_s"
OUT_DIR  = str(figures_root())

# ============================== style ======================================
PURPLE = "#5B3A8C"
LINE_LW, MLE_LW, THR_LW = 1.3, 1.4, 0.9
X_PAD_R = 1.7    # right-hand x-limit padding: the MLEs sit at the grid maximum

# Interior MLE -> two-sided 95% interval: the signed root R = sign(s_hat-s0)*sqrt(LR)
# is ~N(0,1), so |R| > 1.96, i.e. LR > chi2_1(0.95) = 3.841, i.e. l - l_max < -1.921.
THR_DLL          = -chi2.ppf(0.95, 1) / 2.0
# MLE pinned at s = 0 -> a two-sided interval is degenerate (its lower limit is the
# boundary regardless), so report a ONE-SIDED 95% upper bound: R < -1.645, i.e.
# LR > chi2_1(0.90) = 2.706, i.e. l - l_max < -1.353. One tail instead of two --
# still 95%, NOT a 90% interval.
THR_DLL_BOUNDARY = -chi2.ppf(0.90, 1) / 2.0

# ========================= panel A: data + metadata ========================
# ---- gnomAD v4.1.1 allele counts (non-UKBB excluded, i.e. UKBB included) ----
# k is the allele count and n the allele number, i.e. sampled chromosomes; k = 0
# means the variant was not seen in that sample. Every figure in this notebook
# reads this same table, and the likelihood panels use the same n, so the whole
# notebook quotes one set of sample sizes.



# frequency and its 95% CI per mutator, South Asian column
FREQS = {eid: freq_and_ci(*cnt["SAS"]) for eid, cnt in GNOMAD_COUNTS.items()}

SIM = {
    "POLE":        (f"{SIM_DIR}/POLE/POLE_sas_dem_const_phiG.npz",   "0.0188"),
    "POLD1":       (f"{SIM_DIR}/POLD1/POLD1_sas_dem_const_phiG.npz", "0.0092"),
    "MPG":         (f"{SIM_DIR}/MPG/MPG_sas_dem_const_phiG.npz",     "0.0102"),
    "XPC":         (f"{SIM_DIR}/XPC/XPC_sas_dem_const_phiG.npz",     "XPC_freq_s=0.0187"),
    "MUTYH_Y179C": (f"{SIM_DIR}/MUTYH/MUTYH_sas_dem_const_phiG.npz", "Y179C_freq_s=0.0020"),
    "MUTYH_V234M": (f"{SIM_DIR}/MUTYH/MUTYH_sas_dem_const_phiG.npz", "V234M_freq_s=0.0020"),
    "MUTYH_G368D": (f"{SIM_DIR}/MUTYH/MUTYH_sas_dem_const_phiG.npz", "G368D_freq_s=0.0020"),
}

# ========================== CONDITIONING SWITCHES ==========================
# The simulation summaries now hold EVERY replicate, with a frequency of exactly
# 0 recorded where the mutator was lost, so the conditioning is chosen HERE
# rather than being baked into the .npz. Both switches apply to the whole cell:
# panel A and panels B-F use the same rule, so the two halves of the figure
# always agree.
#
#   CONDITION_ON_SEGREGATING
#       True   restrict every distribution to q > 0 -- the published behaviour
#       False  every replicate, zeros included (the unconditional distribution)
#
#   MUTYH_CONDITIONING          (only bites when CONDITION_ON_SEGREGATING)
#       "independent"  each of the three variants on its OWN segregation,
#                      independent of whether its siblings segregate. This is
#                      the marginal the gnomAD comparison needs, and it is
#                      3.5-4.6x the sample of "all_three".
#       "all_three"    only the replicates in which ALL THREE variants
#                      segregate -- the old `--require all` run set.
#
# Reproduction note: before 2026-09 the two halves disagreed -- panel A used
# "independent" and panel F "all_three". "independent" reproduces the published
# panel A exactly; "all_three" reproduces the published panel F exactly.
#
# Display note: an unconditional distribution cannot be read as a boxplot on a
# log axis when most of its mass sits at zero. POLE is 79.2% zeros and POLD1
# 62.1%, so their quartiles are all 0 and the boxes collapse onto the axis floor.
CONDITION_ON_SEGREGATING = False            # fixed: this is the unconditioned cell
MUTYH_CONDITIONING       = "independent"    # no effect here -- it only chooses among
                                            # ways of conditioning, and this cell does
                                            # not condition

if MUTYH_CONDITIONING not in ("independent", "all_three"):
    raise ValueError(f"MUTYH_CONDITIONING must be 'independent' or 'all_three', "
                     f"got {MUTYH_CONDITIONING!r}")



def apply_conditioning(arr, siblings=None):
    """Apply the two switches to one mutator's per-replicate frequencies.

    `siblings` is the list of the co-simulated MUTYH variant arrays, aligned
    run-for-run with `arr`; pass it only for MUTYH. Every array here has one
    entry per simulated replicate, so a boolean mask over one is a mask over all.
    """
    if not CONDITION_ON_SEGREGATING:
        return arr
    if siblings is not None and MUTYH_CONDITIONING == "all_three":
        keep = np.ones(arr.size, dtype=bool)
        for sib in siblings:
            keep &= sib > 0
        return arr[keep]
    return arr[arr > 0]


def mutyh_siblings(data, key):
    """The three run-aligned MUTYH variant arrays that share `key`'s s value."""
    s_str = key.split("_freq_s=")[1]
    return [np.asarray(data[f"{v}_freq_s={s_str}"], dtype=float).ravel()
            for v in MUTYH_VARIANTS_ALL]


def freqs_for(eid):
    """Panel A: simulated frequencies for one mutator, conditioned per the switches."""
    path, key = SIM[eid]
    if not summary_is_unconditional(path):
        raise SystemExit(
            f"{path}\n  discarded the replicates in which the mutator was absent, so "
            f"the conditioning cannot be chosen here.\n  Re-run "
            f"Snakefile_SAS_dem_const_phiG_all_genes to regenerate it.")
    data = np.load(path)
    arr = np.asarray(data[key], dtype=float).ravel()
    sibs = mutyh_siblings(data, key) if eid.startswith("MUTYH_") else None
    return apply_conditioning(arr, sibs)

# ---- zeros on a log axis --------------------------------------------------
# An unconditional distribution has real mass at exactly 0 -- under the South Asian history 79% of POLE
# replicates, 62% of POLD1 and 58% of MUTYH G368D --
# and 0 has no place on a log axis. matplotlib puts the whole box at y = 0, i.e.
# off the bottom of the axes, so the box simply disappears. Anything at or below
# Y_FLOOR, the bottom of the y axis, is therefore drawn AT the floor, and every
# box whose 2.5th percentile, lower quartile or median lands there is marked
# with a downward caret and named underneath, so a box resting on the floor is
# never read as a measured frequency.
Y_FLOOR = 1e-8
floor_for_log = FloorTracker(Y_FLOOR)
report_floored = lambda: floor_for_log.report(22)


PLOT_ENTRIES = [
    {"id": "POLE",        "x_tick_label": r"$\it{POLE}$",  "variant_label": None,    "is_mutyh": False},
    {"id": "POLD1",       "x_tick_label": r"$\it{POLD1}$", "variant_label": None,    "is_mutyh": False},
    {"id": "MPG",         "x_tick_label": r"$\it{MPG}$",   "variant_label": None,    "is_mutyh": False},
    {"id": "XPC",         "x_tick_label": r"$\it{XPC}$",   "variant_label": None,    "is_mutyh": False},
    {"id": "MUTYH_Y179C", "x_tick_label": r"$\it{MUTYH}$", "variant_label": "Y179C", "is_mutyh": True},
    {"id": "MUTYH_V234M", "x_tick_label": None,            "variant_label": "V234M", "is_mutyh": True},
    {"id": "MUTYH_G368D", "x_tick_label": None,            "variant_label": "G368D", "is_mutyh": True},
]
SINGLE_BOX_WIDTH, MUTYH_BOX_WIDTH = 0.40, 0.18
MUTYH_LABEL_FONTSIZE = 7
SINGLE_LINE_HW,   MUTYH_LINE_HW   = 0.15, 0.10
INTER_GENE_SPACING, MUTYH_SPACING = 1.05, 0.42

positions, pos = [], 0.0
for entry in PLOT_ENTRIES:
    if entry["is_mutyh"] and entry["id"] != "MUTYH_Y179C":
        pos += MUTYH_SPACING
    elif positions:
        pos += INTER_GENE_SPACING
    positions.append(pos)

# ==================== panels B-F: likelihood machinery =====================
# order matches panel A, so the two halves of the figure read together
SINGLE = {
    "POLE":  dict(file=f"{VARY_DIR}/POLE/POLE_sas_varying_s.npz",   K=0,  N=91090, h=0.5, dominant=True),
    "POLD1": dict(file=f"{VARY_DIR}/POLD1/POLD1_sas_varying_s.npz", K=0,  N=89328, h=0.5, dominant=True),
    "MPG":   dict(file=f"{VARY_DIR}/MPG/MPG_sas_varying_s.npz",     K=35, N=91070, h=0.0, dominant=False),
    "XPC":   dict(file=f"{VARY_DIR}/XPC/XPC_sas_varying_s.npz",     K=2,  N=90690, h=0.0, dominant=False),
}
# MUTYH's replicates were retained jointly (a run counts only when all three
# variants segregate), which is the conditioning used here.
MUTYH_NPZ = f"{VARY_DIR}/MUTYH/MUTYH_sas_varying_s.npz"
MUTYH_VARIANTS = {"Y179C": dict(k=4,  n=91090),
                  "V234M": dict(k=0,  n=91086),   # grey cell -> not observed -> k = 0
                  "G368D": dict(k=13, n=91088)}


def grid_keys(path, variant=None):
    """{s: npz key}; handles both key conventions, without reading any array."""
    with np.load(path, allow_pickle=True) as data:
        names = list(data.files)
    out = {}
    for key in names:
        if key.startswith("age_") or "_age_s=" in key:
            continue
        if "_s=" in key:
            if variant is not None and not key.startswith(f"{variant}_"):
                continue
            s_val = float(key.split("_s=")[1])
        else:
            s_val = float(key)
        out[round(s_val, 12)] = key
    return out


def load_grid(path, variant=None):
    """{s: array of simulated frequencies}; handles both npz key conventions.

    The arrays are left in their stored float32 rather than widened to float64:
    the unconditional MUTYH grid is 42 x 3 x 4e6 values, which is 2 GB as
    float32 and 4 GB as float64. binom.logpmf promotes per s value anyway, so
    nothing downstream is affected.
    """
    keys = grid_keys(path, variant)
    with np.load(path, allow_pickle=True) as data:
        return {s_val: np.ravel(data[key]) for s_val, key in keys.items()}


def loglik(q, K, N):
    return logsumexp(binom_logpmf(K, N, np.clip(q, 0, 1))) - np.log(q.size)


def single_gene_curve(cfg):
    """Relative log-likelihood for one single-locus mutator, or None if the grid
    on disk cannot support the requested conditioning."""
    if not CONDITION_ON_SEGREGATING and not summary_is_unconditional(cfg["file"]):
        return None
    g = load_grid(cfg["file"])
    s = np.array(sorted(g))
    ll = []
    for v in s:
        ll.append(loglik(apply_conditioning(g[v]), cfg["K"], cfg["N"]))
    return s, np.array(ll)


def mutyh_joint_curve():
    """Factorized joint: the three variants are independent given s*, so the joint
    log-likelihood is the SUM of the three single-variant marginals.

    The three arrays at a given s are aligned run-for-run, which is what lets
    MUTYH_CONDITIONING == "all_three" take the same replicate subset from each.
    Returns None if the grid on disk cannot support the requested conditioning:
    a grid that kept only the replicates in which all three variants segregate
    already IS that subset, so it can serve "all_three" but neither "independent"
    nor the unconditional case.
    """
    unconditional_on_disk = summary_is_unconditional(MUTYH_NPZ)
    if not unconditional_on_disk:
        if not CONDITION_ON_SEGREGATING or MUTYH_CONDITIONING != "all_three":
            return None
    # one s value at a time: the whole unconditional grid is 42 x 3 x 4e6 values,
    # so holding all three variants at once costs 2 GB for no reason
    keys = {v: grid_keys(MUTYH_NPZ, variant=v) for v in MUTYH_VARIANTS_ALL}
    s_grid = np.array(sorted(set.intersection(*[set(keys[v]) for v in MUTYH_VARIANTS])))
    ll = np.empty(len(s_grid))
    with np.load(MUTYH_NPZ, allow_pickle=True) as data:
        for i, s in enumerate(s_grid):
            cols = {v: np.ravel(data[keys[v][s]]) for v in MUTYH_VARIANTS_ALL}
            sibs = [cols[v] for v in MUTYH_VARIANTS_ALL] if unconditional_on_disk else None
            tot = 0.0
            for v, c in MUTYH_VARIANTS.items():
                q = apply_conditioning(cols[v], sibs)
                tot += (-np.inf if q.size == 0
                        else logsumexp(binom_logpmf(c["k"], c["n"], np.clip(q, 0, 1)))
                             - np.log(q.size))
            ll[i] = tot
    return s_grid, ll


def draw_likelihood(ax, s_all, ll_all, h, dominant, name, letter,
                    note=None, show_ylabel=True):
    """One relative log-likelihood panel; returns the numbers it reports."""
    o = np.argsort(s_all)
    s_all, ll_all = s_all[o], ll_all[o]
    gmax = ll_all.max()
    imax_all = int(np.argmax(ll_all))
    mle_neutral = (s_all[imax_all] == 0.0)
    thr = THR_DLL_BOUNDARY if mle_neutral else THR_DLL   # see the note above

    s, ll = s_all[s_all > 0], ll_all[s_all > 0]
    scale = h if dominant else 1.0
    xp = scale * s
    x = np.log10(xp)
    dll = ll - gmax
    ipos = int(np.argmax(dll))
    center = scale * np.sqrt(s.min() * s.max())
    ci_empty = dll[ipos] < thr

    def cross(rng, nxt):
        for i in rng:
            a, b = dll[i], dll[nxt(i)]
            if (a - thr) * (b - thr) < 0:
                t = (thr - a) / (b - a)
                return 10 ** (x[i] + t * (x[nxt(i)] - x[i]))
        return None

    lo = None if ci_empty else cross(range(ipos, 0, -1), lambda i: i - 1)
    hi = None if ci_empty else cross(range(ipos, len(s) - 1), lambda i: i + 1)

    # ---- curve, threshold, CI band, MLE ----
    ax.plot(xp, dll, "-", lw=LINE_LW, color=PURPLE, zorder=4)
    ax.axhline(thr, ls="--", lw=THR_LW, color="0.45", zorder=3)
    if (lo is not None) and (hi is not None):     # shade only a bounded interval
        ax.axvspan(lo, hi, color=PURPLE, alpha=0.13, lw=0, zorder=1)
    if not mle_neutral:
        ax.axvline(scale * s_all[imax_all], color="maroon", lw=MLE_LW, zorder=5)

    # ---- bottom axis ----
    ax.set_xscale("log")
    # pad the right limit only: several genes have their MLE at the top of the
    # simulated grid, and without the padding the maroon line would sit on the
    # spine. The left limit stays exactly at the grid minimum.
    ax.set_xlim(xp.min(), xp.max() * X_PAD_R)
    ax.set_ylim(-2, 0.25)
    ax.set_yticks([0, -0.5, -1.0, -1.5, -2.0])
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    # the quantity differs by dominance class, so the label names it outright
    ax.set_xlabel(r"Selection coefficient ($h\,s^{*}$)" if dominant
                  else r"Selection coefficient ($s^{*}$)",
                  fontsize=8, labelpad=1.5)
    if show_ylabel:
        ax.set_ylabel(r"$\ell-\ell_{\max}$", fontsize=9, labelpad=2)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="both", which="major", labelsize=7, length=3)
    ax.tick_params(axis="both", which="minor", length=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- top axis: one tick, marking the empirically estimated value ----
    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    ax2.xaxis.set_minor_locator(NullLocator())
    ax2.set_xticks([center])
    sym = r"h\,\hat{s}^{*}" if dominant else r"\hat{s}^{*}"
    ax2.set_xticklabels([r"$%s$" % sym], fontsize=8)
    ax2.tick_params(axis="x", length=3.5, width=1.1, pad=1.5)
    for sp in ax2.spines.values():
        sp.set_visible(False)

    title = r"$\mathbf{%s.}$  %s" % (letter, name)
    if note:
        title += "  " + note
    ax2.set_title(title, loc="left", pad=10, fontsize=8.5, x=-0.02)

    # ---- report ----
    u = "h*s*" if dominant else "s*"
    plain = name.replace(r"$\it{", "").replace("}$", "")
    mle_str = "boundary s*=0 (neutral)" if mle_neutral else f"{scale*s_all[imax_all]:.4g}"
    lo_s = f"{lo:.4g}" if lo else ("0 (boundary)" if mle_neutral and not ci_empty
                                   else ("<grid min" if not ci_empty else "—"))
    hi_s = f"{hi:.4g}" if hi else (">grid max" if not ci_empty else "—")
    print(f"\n=== {letter}. {plain} ===  K/k as noted, h={h}")
    print(f"  MLE ({u})   : {mle_str}")
    print(f"  CI regime   : " + ("one-sided 95% upper bound (MLE at s=0)" if mle_neutral
                                 else "two-sided 95% interval (interior MLE)"))
    print(f"  95% CI ({u}): [{lo_s}, {hi_s}]")
    # Chernoff test of neutrality, H0: s = 0. Under H0 the MLE is pinned at the
    # boundary half the time, so LR ~ 0.5*chi2_0 + 0.5*chi2_1 and p = 0.5*P(chi2_1>=LR).
    # Separate question from the CI above; it does not affect it.
    if s_all[0] == 0.0:
        LR0 = 2.0 * (gmax - ll_all[0])
        p0 = 1.0 if LR0 <= 0 else 0.5 * chi2.sf(LR0, 1)
        print(f"  neutrality  : LR={LR0:.3f}, p={p0:.4g} "
              f"({'REJECT s=0' if p0 < 0.05 else 'cannot reject s=0'})")
    return dict(name=plain, mle=None if mle_neutral else scale * s_all[imax_all], lo=lo, hi=hi)


LIK_PANELS = [
    dict(key="POLE",  name=r"$\it{POLE}$",  h=0.5, dominant=True,  note=None),
    dict(key="POLD1", name=r"$\it{POLD1}$", h=0.5, dominant=True,  note=None),
    dict(key="MPG",   name=r"$\it{MPG}$",   h=0.0, dominant=False, note=None),
    dict(key="XPC",   name=r"$\it{XPC}$",   h=0.0, dominant=False, note=None),
    dict(key="MUTYH", name=r"$\it{MUTYH}$", h=0.0, dominant=False, note=None),
]

print(f"conditioning: CONDITION_ON_SEGREGATING={CONDITION_ON_SEGREGATING}, "
      f"MUTYH_CONDITIONING={MUTYH_CONDITIONING!r}")

curves = {k: single_gene_curve(c) for k, c in SINGLE.items()}
curves["MUTYH"] = mutyh_joint_curve() if os.path.exists(MUTYH_NPZ) else None

# A likelihood panel is left blank rather than quietly drawn under the wrong
# rule when its varying-s grid on disk cannot support the requested switches.
_grid_files = {k: c["file"] for k, c in SINGLE.items()}
_grid_files["MUTYH"] = MUTYH_NPZ
for _k, _f in _grid_files.items():
    if curves.get(_k) is None:
        why = ("not found" if not os.path.exists(_f)
               else "discarded the replicates in which the mutator was absent, so "
                    "it cannot serve the requested conditioning")
        print(f"NOTE: {_k} panel left blank -- {os.path.basename(_f)} {why}.\n"
              f"      Re-run Snakefile_SAS_dem_varying_selection_coefficient_all_genes, "
              f"then re-run this cell.")
    elif not summary_is_unconditional(_f):
        same = ("which IS the all-three subset, so panel F is unchanged"
                if _k == "MUTYH" else
                "identical to the re-summarised grid once q > 0 is applied")
        print(f"note: {_k} read a grid that kept only segregating replicates "
              f"({os.path.basename(_f)}), {same}.")

# ================================= figure ==================================
with matplotlib.rc_context(rc):
    fig = plt.figure(figsize=(6.5, 7.0))
    # two separate grids rather than one 3-row grid, so the gap between panel A
    # and the likelihood block can be set independently of the gap between rows
    gs_top = fig.add_gridspec(1, 1, left=0.085, right=0.965, top=0.950, bottom=0.690)
    # hspace has to clear a tall header on the second likelihood row: each panel
    # carries its bold letter, then the symbolic top-axis ticks, above the axes
    # itself -- and it sits under the "selection coefficient" xlabel of the row
    # above. 0.68 left E/F crowding that label.
    gs = fig.add_gridspec(2, 6, left=0.085, right=0.965, top=0.548, bottom=0.070,
                          hspace=0.95, wspace=0.52)

    # ------------------------------ Panel A --------------------------------
    ax = fig.add_subplot(gs_top[0, 0])
    for entry, p in zip(PLOT_ENTRIES, positions):
        eid = entry["id"]
        box_width = MUTYH_BOX_WIDTH if entry["is_mutyh"] else SINGLE_BOX_WIDTH
        lhw = MUTYH_LINE_HW if entry["is_mutyh"] else SINGLE_LINE_HW

        arr, floored = floor_for_log(freqs_for(eid), eid)
        if floored:
            ax.plot([p], [Y_FLOOR], marker="v", ms=3.5, color="black",
                    clip_on=False, zorder=6)

        bp = ax.boxplot(arr, positions=[p], widths=box_width,
                        whis=[2.5, 97.5], showfliers=False, showmeans=False,
                        patch_artist=True, manage_ticks=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(BOX_FACECOLOR)
            patch.set_alpha(BOX_ALPHA)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.6)
        for line in bp["whiskers"] + bp["caps"]:
            line.set_color("black")
            line.set_linewidth(0.8)

        val, observed, ci_lo, ci_hi = FREQS[eid]
        # the interval as a shaded band, spanning exactly the same width as the
        # line, with a solid line on top marking the observed frequency. A variant
        # that was never seen has no frequency to mark, so it carries the band
        # alone: it runs down to the axis floor and its top edge is the one-sided
        # 95% upper bound.
        ax.fill_between([p - lhw, p + lhw], max(ci_lo, Y_FLOOR), ci_hi,
                        color=GNOMAD_COLOR, alpha=CI_ALPHA, lw=0, zorder=1)
        if observed:
            ax.plot([p - lhw, p + lhw], [val, val], color=GNOMAD_COLOR,
                    linewidth=1.6, zorder=5)

    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e-2)   # same limits as the NFE/AFR figures, for comparability

    # MUTYH variant names sit ABOVE their own boxes here. Unconditionally about
    # half of each MUTYH distribution is exactly 0, so the boxes rest on the axis
    # floor and a label underneath would be written across them. The y position is
    # therefore in DATA coordinates, just clear of the highest thing drawn for the
    # three variants -- box whisker or confidence bound, whichever is higher.
    _mutyh_top = max(
        max(float(np.percentile(np.maximum(freqs_for(e["id"]), Y_FLOOR), 97.5)),
            FREQS[e["id"]][3])
        for e in PLOT_ENTRIES if e["is_mutyh"])
    for entry, p in zip(PLOT_ENTRIES, positions):
        if entry["variant_label"] is not None:
            ax.text(p, _mutyh_top * 2.2, entry["variant_label"],
                    ha="center", va="bottom", fontsize=MUTYH_LABEL_FONTSIZE)

    mutyh_centre = np.mean([p for e, p in zip(PLOT_ENTRIES, positions) if e["is_mutyh"]])
    tick_positions, tick_labels = [], []
    for entry, p in zip(PLOT_ENTRIES, positions):
        if entry["x_tick_label"] is not None:
            tick_positions.append(mutyh_centre if entry["is_mutyh"] else p)
            tick_labels.append(entry["x_tick_label"])
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.tick_params(axis="x", which="major", bottom=True, top=False, length=3, pad=3)
    ax.tick_params(axis="y", which="major", labelsize=7.5, length=3)
    ax.tick_params(axis="y", which="minor", length=1.8)
    ax.set_ylabel("Allele frequency", fontsize=8.5, labelpad=2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
    ax.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=GNOMAD_COLOR, lw=1.6, label="Observed frequency"),
        matplotlib.patches.Patch(facecolor=GNOMAD_COLOR, alpha=CI_ALPHA, lw=0,
                                 label="95% CI"),
        matplotlib.patches.Patch(facecolor=BOX_FACECOLOR, alpha=BOX_ALPHA,
                                 edgecolor="black", lw=0.8,
                                 label="simulated (IQR; whiskers 2.5–97.5%)"),
    ]
    leg = ax.legend(handles=legend_handles, loc="upper left", frameon=False,
                    borderpad=0.3, handlelength=1.8, fontsize=7, labelspacing=0.35)
    for txt, col in zip(leg.get_texts(), (GNOMAD_COLOR, GNOMAD_COLOR, "black")):
        txt.set_color(col)

    ax.set_title(r"$\mathbf{A.}$  Mutator frequencies",
                 loc="left", pad=5, fontsize=8.5, x=-0.02)

    # ----------------------------- Panels B-F ------------------------------
    slots = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6], gs[1, 0:2], gs[1, 2:4]]
    left_col = (0, 3)                     # panels B and E carry the shared y-label
    results = []
    for i, (slot, letter, cfg) in enumerate(zip(slots, "BCDEF", LIK_PANELS)):
        axl = fig.add_subplot(slot)
        if curves[cfg["key"]] is None:
            axl.set_xticks([]); axl.set_yticks([])
            for sp in axl.spines.values():
                sp.set_visible(False)
            axl.text(0.5, 0.5, "not yet available", transform=axl.transAxes,
                     ha="center", va="center", fontsize=7.5, color="0.55")
            axl.set_title(r"$\mathbf{%s.}$  %s" % (letter, cfg["name"]),
                          loc="left", pad=10, fontsize=8.5, x=-0.02)
            continue
        results.append(draw_likelihood(
            axl, *curves[cfg["key"]], cfg["h"], cfg["dominant"], cfg["name"],
            letter, note=cfg["note"], show_ylabel=(i in left_col)))

    # ------- shared legend for B-F, in the free sixth slot -------
    axleg = fig.add_subplot(gs[1, 4:6])
    axleg.axis("off")
    lik_handles = [
        Line2D([0], [0], color=PURPLE, lw=LINE_LW, label="relative log-likelihood"),
        Line2D([0], [0], color="maroon", lw=MLE_LW,
               # the line marks the maximum on whichever quantity that panel's
               # x axis carries: h s* on the dominant genes, s* on the recessive ones
               label=r"MLE of $h\,s^{*}/s^{*}$"),
        Line2D([0], [0], color="0.45", ls="--", lw=THR_LW, label="95% CI threshold"),
    ]
    axleg.legend(handles=lik_handles, loc="center left", frameon=False, fontsize=7.5,
                 labelspacing=0.9, handlelength=1.9, borderaxespad=0.2,
                 bbox_to_anchor=(0.0, 0.52))
    axleg.text(0.0, 0.90, "Panels B–F", transform=axleg.transAxes,
               fontsize=8, fontweight="bold", va="top", ha="left")

    report_floored()

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(f"{OUT_DIR}/SAS_mutators_freq_and_profile_likelihoods.pdf")
    plt.show()

print(f"\nsaved -> {OUT_DIR}/SAS_mutators_freq_and_profile_likelihoods.pdf")
