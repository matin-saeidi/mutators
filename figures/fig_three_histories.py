#!/usr/bin/env python3
"""Figure S8. Simulated mutator frequency distributions under three demographic histories, at
each mutator's constant effect size phi_G, with a mutation-selection balance
prediction alongside.

Four columns per mutator:
  NFE / AFR / SAS  boxplots of the simulated frequencies (box = IQR, whiskers =
                   2.5th and 97.5th percentile), each overlaid with the gnomAD
                   v4.1.1 frequency for that population
  MSDB / MSB       the balance prediction, which carries no empirical line.
                   Recessive mutators are labelled MSDB and drawn as the
                   Ne = 20,000 mutation-selection-drift balance (orange
                   triangle, Eq. S28); semi-dominant ones are labelled MSB and
                   drawn as the deterministic mutation-selection balance (green
                   circle, Eq. S27).

A variant not observed in a sample is drawn as a dotted line at the one-sided
95% upper confidence bound on its frequency.

Distributions include every simulated replicate; a replicate in which the
mutator was lost enters at a frequency of exactly 0.

Reads   results/{eur,afr,sas}/simulations/  (workflow/const_phiG.smk per pop)
Writes  mutator_freq_dist_NFE_AFR_SAS_with_MSB.pdf
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
from scipy.stats import beta
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = str(results_root())
SIM = {
    "NFE": f"{ROOT}/eur/simulations/{{g}}/{{g}}_eur_dem_const_phiG.npz",
    "AFR": f"{ROOT}/afr/simulations/{{g}}/{{g}}_afr_dem_const_phiG.npz",
    "SAS": f"{ROOT}/sas/simulations/{{g}}/{{g}}_sas_dem_const_phiG.npz",
}
POPS = ["NFE", "AFR", "SAS"]

# ---------------------------------------------------------------------------
# Which npz key holds each mutator's frequencies. s is unchanged across
# histories, so one key per mutator serves all three files.
# ---------------------------------------------------------------------------
MUTATORS = {
    # u        = mutation rate TO THE MUTATOR allele (for XPC the variant rate,
    #            not the gene-wide LoF rate that drove the simulation)
    # phi      = effect size; "het" means it is quoted per heterozygous parent
    # h_mut    = dominance of the mutator allele itself
    "MPG":         dict(gene="MPG",   key="0.0102",              label=r"$\it{MPG}$",
                        u=1.0353e-07, phi=4.2379e-08, phi_scope="hom", h_mut=0.0),
    "XPC":         dict(gene="XPC",   key="XPC_freq_s=0.0187",   label=r"$\it{XPC}$",
                        u=7.922075e-08, phi=7.7921e-08, phi_scope="hom", h_mut=0.0),
    "POLE":        dict(gene="POLE",  key="0.0188",              label=r"$\it{POLE}$",
                        u=3.7048e-09, phi=3.9161e-08, phi_scope="het", h_mut=0.5),
    "POLD1":       dict(gene="POLD1", key="0.0092",              label=r"$\it{POLD1}$",
                        u=7.0543e-09, phi=1.9211e-08, phi_scope="het", h_mut=0.5),
    "MUTYH_Y179C": dict(gene="MUTYH", key="Y179C_freq_s=0.0020", label=r"$\it{MUTYH}$ Y179C",
                        u=1.0049e-08, phi=8.3248e-09, phi_scope="hom", h_mut=0.0),
    "MUTYH_V234M": dict(gene="MUTYH", key="V234M_freq_s=0.0020", label=r"$\it{MUTYH}$ V234M",
                        u=1.1317e-08, phi=8.3248e-09, phi_scope="hom", h_mut=0.0),
    "MUTYH_G368D": dict(gene="MUTYH", key="G368D_freq_s=0.0020", label=r"$\it{MUTYH}$ G368D",
                        u=7.6633e-09, phi=8.3248e-09, phi_scope="hom", h_mut=0.0),
}
ORDER = ["MPG", "XPC", "POLE", "POLD1", "MUTYH_Y179C", "MUTYH_V234M", "MUTYH_G368D"]

# ---- gnomAD v4.1.1 allele counts (non-UKBB excluded, i.e. UKBB included) ----
# k is the allele count and n the allele number, i.e. sampled chromosomes; k = 0
# means the variant was not seen in that sample. Every figure in this notebook
# reads this same table, and the likelihood panels use the same n, so the whole
# notebook quotes one set of sample sizes.



# frequency and its 95% CI per mutator, South Asian column

# frequency and its 95% CI per mutator per population
GNOMAD = {m: {p: freq_and_ci(*cnt[p]) for p in cnt} for m, cnt in GNOMAD_COUNTS.items()}


# ---------------------------------------------------------------------------
# Mutation-selection balance predictions, computed here from the SAME phi and s
# that drove the simulations, so they cannot drift out of step with them.
#
#   recessive     (h_m = 0)   : u * sqrt(2*N*pi/s) at N = 20,000  [orange triangle]
#                               sqrt(u/s), deterministic          [green circle]
#   semi-dominant (h_m = 0.5) : u / (h_m * s), deterministic      [green circle]
#
# Each entry carries a `draw` flag. The recessive deterministic equilibrium is
# still computed -- it is the sqrt(u/s) value, kept so it stays available and in
# step with the simulations -- but it is NOT drawn: for a recessive mutator at
# N_e = 20,000 the relevant comparison is the drift-inclusive value. So each
# panel shows exactly one theory point: the triangle for recessive mutators, the
# circle for semi-dominant ones.
# ---------------------------------------------------------------------------
N_MSB  = 20000
F_SEL, G_SIZE, S_DEL, H_DEL = 0.08, 3e9, 0.001, 0.5


def mutator_s(cfg):
    """Selection coefficient of the mutator in a homozygote."""
    phi_hom = 2.0 * cfg["phi"] if cfg["phi_scope"] == "het" else cfg["phi"]
    return 2.0 * phi_hom * F_SEL * G_SIZE * S_DEL * H_DEL


def msb_predictions(cfg):
    """[(value, marker, colour, draw), ...] for the theory column."""
    s = mutator_s(cfg)
    if cfg["h_mut"] == 0.0:                       # recessive
        return [
            (cfg["u"] * np.sqrt(2 * N_MSB * np.pi / s), "^", MSB_NE_COLOR,  True),   # Eq. S28
            (np.sqrt(cfg["u"] / s),                     "o", MSB_DET_COLOR, False),  # Eq. S27, not drawn
        ]
    return [(cfg["u"] / (cfg["h_mut"] * s), "o", MSB_DET_COLOR, True)]       # Eq. S27


# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------
GNOMAD_COLOR = "#d62728"
MSB_NE_COLOR  = "orange"   # finite-population MSB, N_e = 20,000 (recessive only)
MSB_DET_COLOR = "green"    # deterministic equilibrium (both dominance classes)
BOX_FACE     = "lightgray"
BOX_ALPHA    = 0.7
BOX_W        = 0.6
MSB_POS      = len(POPS) + 1          # 4th column
YLIM         = (1e-8, 2e-1)


# ========================== CONDITIONING SWITCHES ==========================
# Identical in meaning to the switches at the top of the combined SAS figure,
# so the two figures can be set to agree.
#
#   CONDITION_ON_SEGREGATING
#       True   restrict every distribution to q > 0 -- the published behaviour
#       False  every replicate, zeros included (the unconditional distribution)
#
#   MUTYH_CONDITIONING          (only bites when CONDITION_ON_SEGREGATING)
#       "independent"  each variant on its OWN segregation
#       "all_three"    only replicates in which all three variants segregate
#
# A column is drawn only if the summary on disk can actually serve the chosen
# rule, and is left blank and listed underneath otherwise -- the figure never
# mixes conditionings across its three populations without saying so.
#
# Re-summarised trees (a summary whose .json says "conditioned_on_segregating":
# false) hold every replicate and can serve any setting. One written earlier
# discarded the absent replicates at summarise time and can serve only
# CONDITION_ON_SEGREGATING = True, and for MUTYH only "all_three".
#
# All three sets of runs now keep every replicate, so every setting works here.
# "independent" is the default so that this figure matches the combined South
# Asian figure. Note this changes the MUTYH boxes relative to the version of
# this figure produced before Sep 2026: back then the summaries for all three
# populations held only the replicates in which all three variants segregate,
# which is what "all_three" reproduces.
CONDITION_ON_SEGREGATING = False            # fixed: this is the unconditioned cell
MUTYH_CONDITIONING       = "independent"    # no effect here -- it only chooses among
                                            # ways of conditioning, and this cell does
                                            # not condition

if MUTYH_CONDITIONING not in ("independent", "all_three"):
    raise ValueError(f"MUTYH_CONDITIONING must be 'independent' or 'all_three', "
                     f"got {MUTYH_CONDITIONING!r}")



# ---- zeros on a log axis --------------------------------------------------
# An unconditional distribution has real mass at exactly 0, and 0 has no place
# on a log axis: matplotlib puts the whole box at y = 0, i.e. off the bottom of
# the axes, so the box simply disappears. Anything at or below Y_FLOOR, the
# bottom of the y axis, is therefore drawn AT the floor, and every box whose
# 2.5th percentile, lower quartile or median lands there is marked with a
# downward caret and named underneath, so a box resting on the floor is never
# read as a measured frequency.
Y_FLOOR = YLIM[0]
floor_for_log = FloorTracker(Y_FLOOR)
report_floored = lambda: floor_for_log.report(26)


# NFE is drawn conditional on the mutator segregating in BOTH versions of this
# figure. Its present-day Ne is ~6.5x the South Asian one, so far fewer replicates
# lose the allele there -- 0% of MPG and XPC replicates end at zero against 79%
# and 62% for POLE and POLD1 under the South Asian history -- and an
# unconditional NFE column would therefore look almost unchanged while the other
# two collapse onto the axis floor. Holding NFE conditional keeps the three
# columns comparable instead of turning a difference in how often the allele
# survives into an apparent difference in where it sits when it does.
ALWAYS_CONDITIONED = {"NFE"}


def load_freqs(mutator, pop):
    """Frequencies for one mutator under one history, conditional on segregating.

    Returns None -- rather than raising -- if that simulation is not on disk
    yet, so the figure can be drawn from whatever is finished. Columns with no
    data are left blank and listed underneath; re-run the cell once the missing
    runs land and they fill themselves in.
    """
    cfg = MUTATORS[mutator]
    path = SIM[pop].format(g=cfg["gene"])
    condition_here = CONDITION_ON_SEGREGATING or pop in ALWAYS_CONDITIONED
    if not os.path.exists(path):
        return None, "no file"
    try:
        with np.load(path) as z:
            key = cfg["key"]
            if key not in z.files:                 # single-variant alias
                alt = key.split("_", 1)[1] if "_" in key else None
                if alt and alt in z.files:
                    key = alt
                else:
                    return None, f"key {cfg['key']!r} absent"
            arr = np.asarray(z[key], dtype=float).ravel()
            is_mutyh = mutator.startswith("MUTYH_") and "_freq_s=" in key
            if not summary_is_unconditional(path):
                # this summary discarded the absent replicates at summarise
                # time, so it can only serve the conditioning it was written with
                if not condition_here:
                    return None, "summary keeps only segregating replicates; re-run its Snakefile"
                if is_mutyh and MUTYH_CONDITIONING != "all_three":
                    return None, ("summary keeps only replicates where all three "
                                  "segregate; re-run its Snakefile")
                return (arr, None) if arr.size else (None, "no segregating runs")
            # unconditional summary: one entry per replicate for every variant,
            # aligned run-for-run, so the co-segregating subset is a mask
            if not condition_here:
                return (arr, None) if arr.size else (None, "no runs")
            if is_mutyh and MUTYH_CONDITIONING == "all_three":
                s_str = key.split("_freq_s=")[1]
                keep = np.ones(arr.size, dtype=bool)
                for v in MUTYH_VARIANTS_ALL:
                    keep &= np.asarray(z[f"{v}_freq_s={s_str}"], dtype=float).ravel() > 0
                arr = arr[keep]
    except Exception as exc:                       # unreadable / half-written npz
        return None, f"unreadable ({type(exc).__name__})"
    arr = arr[arr > 0]
    return (arr, None) if arr.size else (None, "no segregating runs")


# Guard: the s implied by each mutator's phi must equal the s embedded in its
# npz key, i.e. the value actually fed to the simulator. This is what stopped
# being true when the effect sizes were updated and the theory values were not.
for _m, _c in MUTATORS.items():
    _key_s = _c["key"].split("_s=")[1] if "_s=" in _c["key"] else _c["key"]
    _calc_s = "{:.4f}".format(mutator_s(_c))
    assert _calc_s == _key_s, (
        f"{_m}: phi implies s={_calc_s} but the simulation key says s={_key_s}. "
        "The MSB prediction and the simulation would not describe the same allele.")

fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.49))
axes = axes.ravel()

missing = []          # (mutator, population, why) for anything not yet available

for idx, mutator in enumerate(ORDER):
    ax = axes[idx]

    # Column positions stay fixed whatever is present, so panels drawn at
    # different times remain directly comparable.
    have_pos, have_data = [], []
    for pos, pop in enumerate(POPS, start=1):
        arr, why = load_freqs(mutator, pop)
        if arr is None:
            missing.append((mutator, pop, why))
            ax.text(pos, 0.02, "n/a", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6, color="0.55")
            continue
        arr, floored = floor_for_log(arr, f"{mutator} [{pop}]")
        if floored:
            ax.plot([pos], [Y_FLOOR], marker="v", ms=3.5, color="black",
                    clip_on=False, zorder=6)
        have_pos.append(pos)
        have_data.append(arr)
        # gnomAD frequency for THAT population, over its own column only
        val, observed, ci_lo, ci_hi = GNOMAD[mutator][pop]
        # the 95% CI as a shaded band spanning exactly the width of the line, with
        # a solid line on top marking the observed frequency. A variant not seen in
        # that sample has no frequency to mark, so it carries the band alone: it
        # runs down to the axis floor and its top edge is the one-sided upper bound.
        ax.fill_between([pos - BOX_W / 2, pos + BOX_W / 2], max(ci_lo, Y_FLOOR), ci_hi,
                        color=GNOMAD_COLOR, alpha=CI_ALPHA, lw=0, zorder=1)
        if observed:
            ax.hlines(val, pos - BOX_W / 2, pos + BOX_W / 2, colors=GNOMAD_COLOR,
                      lw=1.5, linestyles="-", zorder=4)

    if have_data:
        bp = ax.boxplot(have_data, positions=have_pos, widths=BOX_W,
                        whis=[2.5, 97.5], showfliers=False, showmeans=False,
                        patch_artist=True, manage_ticks=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(BOX_FACE); patch.set_alpha(BOX_ALPHA)
            patch.set_edgecolor("black"); patch.set_linewidth(0.8)
        for med in bp["medians"]:
            med.set_color("black"); med.set_linewidth(1.6)
        for line in bp["whiskers"] + bp["caps"]:
            line.set_color("black"); line.set_linewidth(0.8)

    # MSB prediction (theory column)
    for _val, _marker, _mcolor, _draw in msb_predictions(MUTATORS[mutator]):
        if not _draw:
            continue
        ax.plot(MSB_POS, _val, marker=_marker, color=_mcolor, markersize=6,
                linestyle="none", zorder=5)

    ax.axvline(MSB_POS - 0.5, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)

    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.set_xlim(0.5, MSB_POS + 0.5)
    ax.set_xticks(range(1, MSB_POS + 1))
    # recessive mutators are compared against mutation-selection-DRIFT balance,
    # semi-dominant ones against deterministic mutation-selection balance
    theory_col = "MSDB" if MUTATORS[mutator]["h_mut"] == 0.0 else "MSB"
    ax.set_xticklabels(POPS + [theory_col], fontsize=7)
    ax.tick_params(axis="x", labelsize=7, length=3)
    ax.tick_params(axis="y", labelsize=7, length=3)
    if idx % 4 != 0:
        ax.tick_params(axis="y", labelleft=False)
    else:
        ax.set_ylabel("Frequency", fontsize=8.5)

    ax.text(-0.02, 1.10, f"{chr(65 + idx)}.", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="baseline", ha="left")
    ax.text(0.16, 1.10, MUTATORS[mutator]["label"], transform=ax.transAxes,
            fontsize=8.5, va="baseline", ha="left")

# ---- legend occupies the 8th slot ----
ax_leg = axes[len(ORDER)]
ax_leg.axis("off")
handles = [
    Line2D([0], [0], color=GNOMAD_COLOR, ls="-", lw=1.5,
           label="Observed frequency"),
    Patch(facecolor=GNOMAD_COLOR, alpha=CI_ALPHA, lw=0, label="95% CI"),
    # named by what each marker is, not by the expression that produces it; the
    # labels wrap so the four entries still fit the narrow eighth panel
    Line2D([0], [0], color=MSB_DET_COLOR, marker="o", ls="none", markersize=6,
           label="Equilibrium frequency\nunder MSB"),
    Line2D([0], [0], color=MSB_NE_COLOR, marker="^", ls="none", markersize=6,
           label="Equilibrium frequency\nunder MSDB"),
]
# 6.5 pt with a shorter handle: the two wrapped MSB/MSDB labels are wide, and at
# 7 pt the legend overflowed this narrow eighth panel by ~9 px
ax_leg.legend(handles=handles, loc="center", fontsize=7.5, frameon=False,
              handlelength=1.2, labelspacing=0.9, borderpad=0.2)

fig.tight_layout(pad=0.5, h_pad=1.8, w_pad=0.8)

# ---- report anything that was not available ----
if missing:
    print(f"{len(missing)} of {len(ORDER) * len(POPS)} simulation columns are not "
          f"available yet and were left blank:")
    for _m, _p, _why in missing:
        print(f"    {_p:4} {_m:12} ({_why})")
    print("Re-run this cell once those runs finish and they will fill in.")
else:
    print(f"All {len(POPS)} demographic histories present for all {len(ORDER)} mutators.")

pdf_dir = str(figures_root())
os.makedirs(pdf_dir, exist_ok=True)
fig.savefig(f"{pdf_dir}/mutator_freq_dist_NFE_AFR_SAS_with_MSB.pdf", bbox_inches="tight")
plt.show()