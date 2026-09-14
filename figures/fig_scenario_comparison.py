#!/usr/bin/env python3
"""Figures S7 and S9. South Asian mutator frequencies, comparing the baseline runs with one
alternative as a paired boxplot per mutator.

  --scenario shet      S7: s_het = 5e-4 (baseline) against s_het = 1e-4
  --scenario backmut   S9: back mutation on (baseline) against off

s_het is the heterozygous fitness cost of one deleterious mutation induced by
the mutator, and enters the mutator's own selection coefficient through
s = 2 phi_hom f G s_het. Demography, phi_G, h_mut, mutation rates and run counts
are the same in both members of a pair.

The two runs of the shet pair wrote s to different precision in their filenames,
so each condition carries its own npz key; the backmut pair shares one key.

The gnomAD frequency belongs to the variant rather than to either condition, so
it is drawn once per mutator and spans both boxes: solid where the allele was
observed, dotted where it was not and the value is the one-sided 95% upper
confidence bound.

Distributions include every simulated replicate; a replicate in which the
mutator was lost enters at a frequency of exactly 0.

Reads   results/sas/simulations/ plus simulations_shet_1e-4/ or
        simulations_no_back_mut/  (workflow/const_phiG*.smk)
Writes  SAS_mutator_freqs_shet_5e-4_vs_1e-4.pdf
        SAS_mutator_freqs_back_mut_on_vs_off.pdf
"""
import argparse
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

SCENARIOS = {
    "shet": dict(
        alt_dirname="simulations_shet_1e-4",
        alt_suffix="_shet_1e-4",
        alt_face="#ededed",
        base_label=r"$s_{het} = 5\times 10^{-4}$",
        alt_label=r"$s_{het} = 1\times 10^{-4}$",
        col_base="median 5e-4", col_alt="median 1e-4",
        outfile="SAS_mutator_freqs_shet_5e-4_vs_1e-4.pdf",
        workflow="const_phiG_shet_1e-4.smk",
        # the two runs formatted s differently into their .out filenames -- 4
        # decimals for the baseline, 8 for the re-run, since 5x smaller values
        # collapse under 4 -- so each condition carries its own npz key
        alt_keys={"POLE": "0.00375946", "POLD1": "0.00184426", "MPG": "0.00203419",
                  "XPC": "XPC_freq_s=0.00374021",
                  "MUTYH_Y179C": "Y179C_freq_s=0.00039959",
                  "MUTYH_V234M": "V234M_freq_s=0.00039959",
                  "MUTYH_G368D": "G368D_freq_s=0.00039959"},
    ),
    "backmut": dict(
        alt_dirname="simulations_no_back_mut",
        alt_suffix="_no_back_mut",
        alt_face="#FDAE6B",
        base_label="With back mutation",
        alt_label="No back mutation",
        col_base="median (back on)", col_alt="median (back off)",
        outfile="SAS_mutator_freqs_back_mut_on_vs_off.pdf",
        workflow="const_phiG_no_back_mut.smk",
        # both runs formatted s to 4 decimals, so one key serves for both
        alt_keys=None,
    ),
}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="shet",
                help="which comparison to draw against the baseline runs")
ARGS = ap.parse_args()
SC = SCENARIOS[ARGS.scenario]

SIM_DIR_BASE = f"{results_root()}/sas/simulations"
SIM_DIR_ALT  = f"{results_root()}/sas/{SC['alt_dirname']}"
OUT_DIR      = str(figures_root())

# ============================== style ======================================
LINE_LW = 1.8           # gnomAD frequency lines

# The two conditions, in the order they are drawn left-to-right within a mutator
CONDITIONS = [
    dict(tag="base", dirname=SIM_DIR_BASE, suffix="",
         face=BOX_FACECOLOR,     label=SC["base_label"]),
    dict(tag="alt",  dirname=SIM_DIR_ALT,  suffix=SC["alt_suffix"],
         face=SC["alt_face"],    label=SC["alt_label"]),
]

# ========================== CONDITIONING SWITCHES ==========================
# The same two switches as the combined South Asian figure, with the same
# meaning, so the figures can be set to agree.
#
#   CONDITION_ON_SEGREGATING
#       True   restrict every distribution to q > 0 -- the published behaviour
#       False  every replicate, zeros included (the unconditional distribution)
#
#   MUTYH_CONDITIONING          (only bites when CONDITION_ON_SEGREGATING)
#       "independent"  each of the three variants on its OWN segregation,
#                      whether or not its siblings segregate
#       "all_three"    only the replicates in which all three segregate
#
# Both settings need a summary that kept every simulated replicate, recording a
# frequency of 0 where a mutator is absent. A summary written before Sep 2026
# discarded those replicates at summarise time, so it can only serve
# CONDITION_ON_SEGREGATING = True, and for MUTYH only the conditioning it was
# written with. freqs_for() below says so explicitly rather than quietly
# returning the wrong run set; the fix is always to re-run that Snakefile,
# which is the summarise step only because the raw .out files are kept.
CONDITION_ON_SEGREGATING = False            # fixed: this is the unconditioned cell
MUTYH_CONDITIONING       = "independent"    # no effect here -- it only chooses among
                                            # ways of conditioning, and this cell does
                                            # not condition

if MUTYH_CONDITIONING not in ("independent", "all_three"):
    raise ValueError(f"MUTYH_CONDITIONING must be 'independent' or 'all_three', "
                     f"got {MUTYH_CONDITIONING!r}")



# ========================= data + metadata =================================
# ---- gnomAD v4.1.1 allele counts (non-UKBB excluded, i.e. UKBB included) ----
# k is the allele count and n the allele number, i.e. sampled chromosomes; k = 0
# means the variant was not seen in that sample. Every figure in this notebook
# reads this same table, and the likelihood panels use the same n, so the whole
# notebook quotes one set of sample sizes.



# frequency and its 95% CI per mutator, South Asian column

FREQS = {eid: freq_and_ci(*cnt["SAS"]) for eid, cnt in GNOMAD_COUNTS.items()}



# gene -> one npz key PER CONDITION: the two runs wrote s with different
# precision into their filenames. Directory and suffix differ too.
# Each MUTYH variant is shown conditional on ITS OWN segregation, independent of
# whether its siblings segregate: that is the marginal the gnomAD comparison
# needs, and it is 3.5-4.6x the sample of the co-segregating subset (317k-418k
# runs per variant against 91k). It is read off the UNCONDITIONAL summary --
# every replicate, 0 recorded where a variant is absent -- by filtering q > 0.
# Verified: the subset of that summary where all three are > 0 reproduces the old
# --require all npz byte-for-byte, so it is the same simulation, differently
# filtered.
# gene -> npz key per condition. The baseline keys are shared by both
# scenarios; the alternative's are supplied by the scenario, or are the
# baseline's again when both runs formatted s the same way.
_BASE_KEYS = {
    "POLE":        dict(gene="POLE",  base_key="0.0188"),
    "POLD1":       dict(gene="POLD1", base_key="0.0092"),
    "MPG":         dict(gene="MPG",   base_key="0.0102"),
    "XPC":         dict(gene="XPC",   base_key="XPC_freq_s=0.0187",   variant="XPC"),
    "MUTYH_Y179C": dict(gene="MUTYH", base_key="Y179C_freq_s=0.0020", variant="Y179C"),
    "MUTYH_V234M": dict(gene="MUTYH", base_key="V234M_freq_s=0.0020", variant="V234M"),
    "MUTYH_G368D": dict(gene="MUTYH", base_key="G368D_freq_s=0.0020", variant="G368D"),
}
SIM = {
    eid: dict(spec,
              alt_key=(SC["alt_keys"][eid] if SC["alt_keys"] else spec["base_key"]))
    for eid, spec in _BASE_KEYS.items()
}




def npz_path(eid, cond):
    """The summary to read for this mutator under this condition.

    Every set of runs these two conditions draw on now keeps each simulated
    replicate, recording a frequency of 0 where the mutator is absent, so the
    conditioning is applied by the switches above rather than at summarise time.
    freqs_for() checks that and says so if it ever stops being true.
    """
    g = SIM[eid]["gene"]
    return f"{cond['dirname']}/{g}/{g}_sas_dem_const_phiG{cond['suffix']}.npz"


def resolve_key(data, want, variant=None):
    """The stored key, tolerating a change in how s was formatted in the filename."""
    if want in data.files:
        return want
    freq = [k for k in data.files if "age" not in k]
    if variant is not None:
        cand = [k for k in freq if k.startswith(f"{variant}_freq_s=")]
    else:
        cand = []
        for k in freq:
            try:
                float(k)
            except ValueError:
                continue
            cand.append(k)
    if len(cand) == 1:
        print(f"  note: key {want!r} absent; using {cand[0]!r}")
        return cand[0]
    raise KeyError(f"cannot resolve key {want!r}; npz holds {sorted(freq)}")


def freqs_for(eid, cond):
    """Simulated frequencies for one mutator under one condition.

    Unconditional: every replicate, a lost mutator entering as exactly 0."""
    path = npz_path(eid, cond)
    data = np.load(path)
    key = resolve_key(data, SIM[eid][cond["tag"] + "_key"], SIM[eid].get("variant"))
    arr = np.asarray(data[key], dtype=float).ravel()
    is_mutyh = eid.startswith("MUTYH_")

    if not summary_is_unconditional(path):
        # this summary discarded its non-segregating replicates when it was
        # written, so it can only serve the conditioning it was written with
        if not CONDITION_ON_SEGREGATING:
            raise SystemExit(
                f"{path}\n  kept only the replicates in which the mutator was "
                f"present, so the unconditional\n  distribution cannot be recovered "
                f"from it. Re-run that Snakefile (summarise step only).")
        if is_mutyh and MUTYH_CONDITIONING != "all_three":
            raise SystemExit(
                f"{path}\n  kept only the replicates in which ALL THREE MUTYH "
                f"variants segregate, so\n  MUTYH_CONDITIONING = 'independent' cannot "
                f"be served from it. Re-run that\n  Snakefile (summarise step only), "
                f"or set MUTYH_CONDITIONING = 'all_three'.")
        return arr

    if not CONDITION_ON_SEGREGATING:
        return arr
    if is_mutyh and MUTYH_CONDITIONING == "all_three":
        # every variant has one entry per replicate, aligned run-for-run, so the
        # replicates where all three segregate are a mask over the three arrays
        s_str = key.split("_freq_s=")[1]
        keep = np.ones(arr.size, dtype=bool)
        for v in MUTYH_VARIANTS_ALL:
            keep &= np.asarray(data[f"{v}_freq_s={s_str}"], dtype=float).ravel() > 0
        arr = arr[keep]
    return arr[arr > 0]


PLOT_ENTRIES = [
    {"id": "POLE",        "x_tick_label": r"$\it{POLE}$",  "variant_label": None,    "is_mutyh": False},
    {"id": "POLD1",       "x_tick_label": r"$\it{POLD1}$", "variant_label": None,    "is_mutyh": False},
    {"id": "MPG",         "x_tick_label": r"$\it{MPG}$",   "variant_label": None,    "is_mutyh": False},
    {"id": "XPC",         "x_tick_label": r"$\it{XPC}$",   "variant_label": None,    "is_mutyh": False},
    {"id": "MUTYH_Y179C", "x_tick_label": r"$\it{MUTYH}$", "variant_label": "Y179C", "is_mutyh": True},
    {"id": "MUTYH_V234M", "x_tick_label": None,            "variant_label": "V234M", "is_mutyh": True},
    {"id": "MUTYH_G368D", "x_tick_label": None,            "variant_label": "G368D", "is_mutyh": True},
]

# ------------------------------ geometry -----------------------------------
# Each mutator is a PAIR of boxes straddling its position p, at p -/+ OFFSET.
SINGLE_BOX_WIDTH,  MUTYH_BOX_WIDTH  = 0.21, 0.14
INTER_GENE_SPACING, MUTYH_SPACING   = 1.2, 0.55
X_PAD = 0.5

# The gap BETWEEN the two boxes of a pair, in data units. One value for both
# box widths, so a single gene's pair is separated by exactly as much as a
# MUTYH variant's pair; the offsets follow from it.
PAIR_GAP = 0.05
SINGLE_BOX_OFFSET = (SINGLE_BOX_WIDTH + PAIR_GAP) / 2
MUTYH_BOX_OFFSET  = (MUTYH_BOX_WIDTH  + PAIR_GAP) / 2

# MUTYH variant names stay BELOW their pair rather than between the two boxes,
# where they would run into the frequency lines. y is in AXES fraction, so the
# labels keep a fixed clearance from the axis whatever the whiskers do; give
# them a per-variant y if they ever need staggering to avoid one another.
# At 8 pt they fit side by side at MUTYH_SPACING = 0.55, so all three sit at
# the same height. Raise one here if a future spacing change crowds them; the
# guard at the bottom of the cell checks the rendered boxes in 2-D.
MUTYH_LABEL_FONTSIZE = 7
LEGEND_FONTSIZE      = 8      # 8, not 9: the legend carries four entries
                              # since the gnomAD ones were added, and at 9 pt
                              # its fourth row overlapped POLE's upper whisker
MUTYH_LABEL_Y = {"Y179C": 0.03, "V234M": 0.03, "G368D": 0.03}

positions, pos = [], 0.0
for entry in PLOT_ENTRIES:
    if entry["is_mutyh"] and entry["id"] != "MUTYH_Y179C":
        pos += MUTYH_SPACING
    elif positions:
        pos += INTER_GENE_SPACING
    positions.append(pos)

# ---------------------- fail early, and say what is missing ----------------
missing = sorted({npz_path(e["id"], c) for e in PLOT_ENTRIES for c in CONDITIONS
                  if not os.path.exists(npz_path(e["id"], c))})
if missing:
    raise SystemExit(
        "these summary files do not exist yet:\n  " + "\n  ".join(missing) +
        f"\n\nRe-run workflow/{SC['workflow']}, then this script.")



# ---- zeros on a log axis --------------------------------------------------
# An unconditional distribution has real mass at exactly 0 -- POLE is 79% zeros unconditionally, POLD1 62% --
# and 0 has no place on a log axis. matplotlib puts the whole box at y = 0, i.e.
# off the bottom of the axes, so the box simply disappears. Anything at or below
# Y_FLOOR, the bottom of the y axis, is therefore drawn AT the floor, and every
# box whose 2.5th percentile, lower quartile or median lands there is marked
# with a downward caret and named underneath, so a box resting on the floor is
# never read as a measured frequency.
Y_FLOOR = 1e-8
floor_for_log = FloorTracker(Y_FLOOR)
report_floored = lambda: floor_for_log.report(22)


def draw_boxplot(ax, arr, pos, width, facecolor, label=""):
    arr, floored = floor_for_log(arr, label)
    if floored:
        ax.plot([pos], [Y_FLOOR], marker="v", ms=3.5, color="black",
                clip_on=False, zorder=6)
    bp = ax.boxplot(arr, positions=[pos], widths=width, whis=[2.5, 97.5],
                    showfliers=False, showmeans=False, patch_artist=True,
                    manage_ticks=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(facecolor)
        patch.set_alpha(BOX_ALPHA)
        patch.set_edgecolor("black")
        patch.set_linewidth(1)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2.0)
    for line in bp["whiskers"] + bp["caps"]:
        line.set_color("black")
        line.set_linewidth(1)
    return bp


# ================================= figure ==================================
fig, ax = plt.subplots(figsize=(6.5, 3.4))

for entry, p in zip(PLOT_ENTRIES, positions):
    eid = entry["id"]
    box_width  = MUTYH_BOX_WIDTH  if entry["is_mutyh"] else SINGLE_BOX_WIDTH
    box_offset = MUTYH_BOX_OFFSET if entry["is_mutyh"] else SINGLE_BOX_OFFSET

    for sign, cond in zip((-1, +1), CONDITIONS):
        draw_boxplot(ax, freqs_for(eid, cond), p + sign * box_offset,
                     box_width, cond["face"], label=f"{eid} [{cond['tag']}]")

    # one gnomAD line per mutator, spanning BOTH boxes
    val, observed, ci_lo, ci_hi = FREQS[eid]
    # the 95% CI as a shaded band spanning exactly the width of the line, with a
    # solid line on top marking the observed frequency. A variant that was never
    # seen has no frequency to mark, so it carries the band alone: it runs down to
    # the axis floor and its top edge is the one-sided 95% upper bound.
    _x0 = p - box_offset - box_width / 2
    _x1 = p + box_offset + box_width / 2
    ax.fill_between([_x0, _x1], max(ci_lo, Y_FLOOR), ci_hi,
                    color=GNOMAD_COLOR, alpha=CI_ALPHA, lw=0, zorder=1)
    if observed:
        ax.plot([_x0, _x1], [val, val], color=GNOMAD_COLOR,
                linewidth=LINE_LW, zorder=5)

ax.set_yscale("log")
ax.set_ylim(1e-8, 1e-2)   # same limits as panel A and the NFE/AFR figures

# MUTYH variant names sit ABOVE their boxes here. Unconditionally about half of
# each MUTYH distribution is exactly 0, so the boxes rest on the axis floor and a
# label underneath would be written across them. MUTYH_LABEL_Y is in axes
# fraction and is not used in this version; the y below is in DATA coordinates,
# just clear of the highest thing drawn for the three variants across BOTH
# conditions -- box whisker or confidence bound, whichever is higher.
_mutyh_top = max(
    max([float(np.percentile(np.maximum(freqs_for(e["id"], c), Y_FLOOR), 97.5))
         for c in CONDITIONS] + [FREQS[e["id"]][0]])
    for e in PLOT_ENTRIES if e["is_mutyh"])
variant_texts = []
for entry, p in zip(PLOT_ENTRIES, positions):
    if entry["variant_label"] is not None:
        variant_texts.append(ax.text(
            p, _mutyh_top * 2.2, entry["variant_label"],
            ha="center", va="bottom", fontsize=MUTYH_LABEL_FONTSIZE))

mutyh_centre = np.mean([p for e, p in zip(PLOT_ENTRIES, positions) if e["is_mutyh"]])
tick_positions, tick_labels = [], []
for entry, p in zip(PLOT_ENTRIES, positions):
    if entry["x_tick_label"] is not None:
        tick_positions.append(mutyh_centre if entry["is_mutyh"] else p)
        tick_labels.append(entry["x_tick_label"])

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=10)
ax.tick_params(axis="x", which="major", bottom=True, top=False, length=4, width=1.5)
ax.tick_params(axis="y", which="major", labelsize=10, length=4, width=1.2)
ax.tick_params(axis="y", which="minor", length=3, width=1.0)

ax.set_ylabel("Allele frequency", fontsize=10.5)
ax.set_xlabel("Gene", fontsize=10.5)
ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
ax.set_xlim(positions[0] - X_PAD, positions[-1] + X_PAD)
# all four spines kept

# The two gnomAD line styles first, then the two box conditions. The dotted line
# is not a frequency the allele was seen at -- the variant was NOT observed, so
# the value plotted is the one-sided 95% upper confidence bound on its frequency,
# which is where it could sit and still have been missed 5% of the time.
legend_handles = [
    Line2D([0], [0], color=GNOMAD_COLOR, lw=LINE_LW, label="Observed frequency"),
    Patch(facecolor=GNOMAD_COLOR, alpha=CI_ALPHA, lw=0, label="95% CI"),
] + [
    Patch(facecolor=c["face"], alpha=BOX_ALPHA, edgecolor="black", label=c["label"])
    for c in CONDITIONS
]
leg = ax.legend(handles=legend_handles, loc="upper left", frameon=False,
                borderpad=0.5, handlelength=1.5, fontsize=LEGEND_FONTSIZE,
                labelspacing=0.35)
for txt, col in zip(leg.get_texts(),
                    (GNOMAD_COLOR, GNOMAD_COLOR, "black", "black")):
    txt.set_color(col)

report_floored()

plt.tight_layout(pad=0.5)

# ---- guard: the MUTYH variant labels must not run into one another ---------
# Separation in 2-D, since the labels are staggered: two boxes are clear if
# EITHER axis separates them, so a horizontal overlap is fine when the y differ.
fig.canvas.draw()
boxes = [t.get_window_extent(renderer=fig.canvas.get_renderer()) for t in variant_texts]
seps = [max(max(b.x0 - a.x1, a.x0 - b.x1), max(b.y0 - a.y1, a.y0 - b.y1))
        for i, a in enumerate(boxes) for b in boxes[i + 1:]]
if seps and min(seps) < 2.0:
    print(f"WARNING: two MUTYH variant labels are {min(seps):.1f} px apart -- widen "
          f"MUTYH_SPACING, or stagger them further via MUTYH_LABEL_Y.")
else:
    print(f"MUTYH variant labels clear by {min(seps):.1f} px (>= 2 px required).")

os.makedirs(OUT_DIR, exist_ok=True)
out_pdf = f"{OUT_DIR}/{SC['outfile']}"
fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
plt.show()

# ---- what actually moved --------------------------------------------------
print(f"\nsaved -> {out_pdf}")
print(f"{'mutator':14s} {SC['col_base']:>18s} {SC['col_alt']:>18s} {'ratio':>8s}")
for entry in PLOT_ENTRIES:
    m = [np.median(freqs_for(entry["id"], c)) for c in CONDITIONS]
    # both medians can be exactly 0 in the unconditioned version, so the ratio
    # is not always defined
    ratio = f"{m[1] / m[0]:8.3f}" if m[0] > 0 else f"{'--':>8s}"
    print(f"{entry['id']:14s} {m[0]:18.4g} {m[1]:18.4g} {ratio}")
