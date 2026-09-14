#!/usr/bin/env python3
"""Figure S10. Mutator allele age distributions under the South Asian demographic history,
against a single neutral reference at the human per-site rate mu = 1.25e-08.

  Panel A  boxplots of the allele age per mutator
  Panel B  empirical CCDF, P(age > tau), of the same distributions

One age per simulation replicate: the age, in generations before the present, of
the oldest surviving lineage assigned to that mutator.

Unlike the frequency figures these distributions are conditioned, because a
replicate in which the variant never arose has no age. A replicate enters a
variant's distribution if at least one lineage in it was assigned to that
variant; this applies to the neutral reference too.

Every mutator is simulated through the compound-het lineage tracker. MPG, POLD1
and POLE are given their single-variant mutation rate, so every lineage is the
focal variant. XPC and MUTYH are simulated as the gene-wide LoF allele and each
lineage is assigned to a focal variant with probability mu_variant / mu_gene.

The neutral reference is not matched in mutation rate to any mutator. Ages are
censored at the 250,000-generation burn-in, marked in panel B.

Reads   results/sas/simulations_ages/  (workflow/mutator_ages.smk)
Writes  mutator_allele_ages_sas_single_neutral.pdf
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

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SAS_ROOT = f"{results_root()}/sas"
AGES_NPZ = f"{SAS_ROOT}/simulations_ages/all_mutators_sas_ages.npz"
OUT_PDF  = f"{figures_root()}/mutator_allele_ages_sas_single_neutral.pdf"

# npz keys are namespaced "<label>__<variant>_maxage_s=<s>"
KEYS = {
    "MPG":         "MPG__MPG_maxage_s=0.0102",
    "XPC":         "XPC__XPC_maxage_s=0.0187",
    "POLD1":       "POLD1__POLD1_maxage_s=0.0092",
    "POLE":        "POLE__POLE_maxage_s=0.0188",
    "MUTYH-Y179C": "MUTYH__Y179C_maxage_s=0.0020",
    "MUTYH-V234M": "MUTYH__V234M_maxage_s=0.0020",
    "MUTYH-G368D": "MUTYH__G368D_maxage_s=0.0020",
    "Neutral":     "human_neutral__neutral_maxage_s=0.0000",
}

BURN_IN = 250_000        # start generation of every run; ages censor here

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
BOX_FACECOLOR = "darkgray"
BOX_ALPHA     = 0.7
WHIS          = [2.5, 97.5]

COLORS = {
    "XPC":          "#2166ac",   # blue
    "POLD1":        "#1b7837",   # green
    "POLE":         "#b35806",   # brown/orange
    "MPG":          "#c51b7d",   # magenta
    "MUTYH-Y179C":  "#d73027",   # red
    "MUTYH-V234M":  "#f46d43",   # orange-red
    "MUTYH-G368D":  "#8073ac",   # purple
    "Neutral":      "#000000",   # black — neutral reference
}

FS_AX   = 9.5    # axis labels
FS_TK   = 8.5    # tick labels
FS_GENE = 9.0    # gene tick labels on ax1
FS_LG   = 8.0    # legend
FS_VL   = 8.5    # variant labels on the boxplot
FS_TT   = 10.0   # panel titles
FS_ANNO = 7.5    # burn-in annotation

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────
with np.load(AGES_NPZ) as z:
    missing = [k for k in KEYS.values() if k not in z.files]
    if missing:
        raise KeyError(f"missing npz keys: {missing}")
    ages = {lab: z[key].astype(np.int64) for lab, key in KEYS.items()}

print("Allele age = oldest surviving lineage per replicate (generations)")
for k, v in ages.items():
    print(f"  {k:<14} n={v.size:>7,}  median={np.median(v):>6.0f}  "
          f"IQR={np.percentile(v,25):.0f}-{np.percentile(v,75):.0f}  "
          f"2.5-97.5%={np.percentile(v,2.5):.0f}-{np.percentile(v,97.5):.0f}  "
          f"max={v.max():,}")

print("\nMEDIAN AGE PER BOX (generations), left to right as plotted:")
for k in KEYS:
    print(f"  {k:<14} {np.median(ages[k]):>8.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
FIG_W = 6.5
FIG_H = FIG_W / 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), constrained_layout=True)


def panel_title(ax, label, title, pad=12, x=-0.14):
    t = ax.set_title(rf'$\bf{{{label}}}$ {title}', loc='left', pad=pad,
                     fontsize=FS_TT, fontweight='normal')
    t.set_x(x)


# ═════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — boxplots
# ═════════════════════════════════════════════════════════════════════════════
# Single-variant genes get WIDE boxes; each MUTYH variant gets a NARROW one, so
# the three variants read as one gene-level cluster.
WIDE     = 0.36                   # every non-MUTYH box; all five are identical
NARROW   = WIDE / 1.5             # each MUTYH variant box; all three are identical
GAP      = 0.2                    # edge-to-edge gap WITHIN the MUTYH cluster
STEP     = NARROW + GAP
GENE_GAP = 1.0 - WIDE             # equal edge-to-edge gap BETWEEN gene groups


def _c2c(w_left, w_right):
    """Centre-to-centre distance leaving GENE_GAP of empty space between boxes."""
    return w_left / 2.0 + GENE_GAP + w_right / 2.0


MPG_POS     = 1.0
XPC_POS     = MPG_POS   + _c2c(WIDE, WIDE)
POLD1_POS   = XPC_POS   + _c2c(WIDE, WIDE)
POLE_POS    = POLD1_POS + _c2c(WIDE, WIDE)
Y179C_POS   = POLE_POS  + _c2c(WIDE, NARROW)
V234M_POS   = Y179C_POS + STEP
G368D_POS   = V234M_POS + STEP
MUTYH_CTR   = (Y179C_POS + G368D_POS) / 2.0
NEUTRAL_POS = G368D_POS + _c2c(NARROW, WIDE)

PLOT_ENTRIES = [
    {"id": "MPG",         "pos": MPG_POS,     "width": WIDE,   "variant_label": None},
    {"id": "XPC",         "pos": XPC_POS,     "width": WIDE,   "variant_label": None},
    {"id": "POLD1",       "pos": POLD1_POS,   "width": WIDE,   "variant_label": None},
    {"id": "POLE",        "pos": POLE_POS,    "width": WIDE,   "variant_label": None},
    {"id": "MUTYH-Y179C", "pos": Y179C_POS,   "width": NARROW, "variant_label": "Y179C"},
    {"id": "MUTYH-V234M", "pos": V234M_POS,   "width": NARROW, "variant_label": "V234M"},
    {"id": "MUTYH-G368D", "pos": G368D_POS,   "width": NARROW, "variant_label": "G368D"},
    {"id": "Neutral",     "pos": NEUTRAL_POS, "width": WIDE,   "variant_label": None},
]

bp = ax1.boxplot(
    [ages[e["id"]] for e in PLOT_ENTRIES],
    positions=[e["pos"]   for e in PLOT_ENTRIES],
    widths   =[e["width"] for e in PLOT_ENTRIES],
    whis=WHIS,
    patch_artist=True,
    showfliers=False,
    medianprops =dict(color="black", linewidth=1.2),
    boxprops    =dict(facecolor=BOX_FACECOLOR),
    whiskerprops=dict(linewidth=0.8),
    capprops    =dict(linewidth=0.8),
)
for patch, entry in zip(bp["boxes"], PLOT_ENTRIES):
    patch.set_facecolor(COLORS["Neutral"] if entry["id"] == "Neutral" else BOX_FACECOLOR)
    patch.set_alpha(BOX_ALPHA)

# Log y: medians run 15-81 while the 97.5% whiskers reach 622 and the lower
# whiskers sit at 1 generation, so a linear axis would flatten every box. The top
# is opened out to 3e3 so the 10^3 decade carries a tick label and the MUTYH
# variant annotations have clear air above their whiskers.
ax1.set_yscale("log")
ax1.set_ylim(0.7, 1.8e3)

# Anchor each variant label above the upper WHISKER CAP (the 97.5th percentile),
# not above the box, so it clears the whisker rather than colliding with it.
whisker_top = {e["id"]: float(np.percentile(ages[e["id"]], WHIS[1]))
               for e in PLOT_ENTRIES}
for entry in PLOT_ENTRIES:
    if entry["variant_label"] is not None:
        ax1.text(entry["pos"], whisker_top[entry["id"]] * 1.7,
                 entry["variant_label"], ha="center", va="bottom",
                 fontsize=FS_VL, rotation=90)

gene_ticks  = [MPG_POS, XPC_POS, POLD1_POS, POLE_POS, MUTYH_CTR, NEUTRAL_POS]
gene_labels = ["MPG",   "XPC",   "POLD1",   "POLE",   "MUTYH",   "Neutral"]
ax1.set_xticks(gene_ticks)
xticklabels = ax1.set_xticklabels(gene_labels, fontsize=FS_GENE, style="italic")
for lbl in xticklabels:                      # "Neutral" is not a gene name
    if lbl.get_text() == "Neutral":
        lbl.set_style("normal")
ax1.set_xlim(MPG_POS - 0.5, NEUTRAL_POS + 0.5)
ax1.tick_params(axis="x", which="major", length=4, width=0.9, direction="out",
                labelsize=FS_GENE)
ax1.tick_params(axis="y", labelsize=FS_TK, width=0.9)
ax1.set_ylabel("Allele age (generations)", fontsize=FS_AX)
ax1.set_xlabel("Genes", fontsize=FS_AX)
panel_title(ax1, 'A.', 'Mutator age distributions')

# ═════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — empirical CCDF
# ═════════════════════════════════════════════════════════════════════════════
def empirical_ccdf(arr):
    """P(X > x) evaluated at each DISTINCT observed age.

    Ages are integers with heavy ties, so collapsing to distinct values gives
    the identical step function at a fraction of the vertices -- a ~100k-point
    curve becomes ~1k, which keeps the vector PDF small. Verified against the
    all-points estimator: maximum difference 0.0.
    """
    x, counts = np.unique(arr, return_counts=True)
    return x, 1.0 - np.cumsum(counts) / arr.size


ccdf_handles = {}
for label, arr in ages.items():
    if arr.size == 0:
        print(f"  WARNING: no age data for {label}, skipping CCDF")
        continue
    x, y = empirical_ccdf(arr)
    keep = y > 0                       # drop the trailing zero so log-y works
    (line,) = ax2.plot(x[keep], y[keep], color=COLORS[label], linewidth=0.9,
                       label=label)
    ccdf_handles[label] = line

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"Allele age $\tau$ (generations)", fontsize=FS_AX)
ax2.set_ylabel(r'$\Pr$(allele is older than $\tau$ generations)', fontsize=FS_AX)
ax2.tick_params(axis="both", labelsize=FS_TK, width=0.9)

# Mark the burn-in. Nothing can be older, so the neutral curve's long flat tail
# runs into this wall rather than converging; lengthening the burn-in moves the
# tail out with it. Under selection no distribution comes near it.
ylim = ax2.get_ylim()
ax2.axvline(BURN_IN, color="gray", linestyle=":", linewidth=0.8, zorder=1)
ax2.text(BURN_IN * 0.85, ylim[0] * 2.0, "burn-in", fontsize=FS_ANNO, color="gray",
         rotation=90, ha="right", va="bottom")
ax2.set_ylim(ylim)

LOWERLEFT_LABELS  = ["MPG", "XPC", "POLE", "POLD1"]
UPPERRIGHT_LABELS = ["MUTYH-Y179C", "MUTYH-V234M", "MUTYH-G368D", "Neutral"]
leg_ll = ax2.legend([ccdf_handles[l] for l in LOWERLEFT_LABELS], LOWERLEFT_LABELS,
                    fontsize=FS_LG, frameon=False, handlelength=1.5, loc="lower left")
ax2.add_artist(leg_ll)
ax2.legend([ccdf_handles[l] for l in UPPERRIGHT_LABELS], UPPERRIGHT_LABELS,
           fontsize=FS_LG, frameon=False, handlelength=1.5, loc="upper right")
panel_title(ax2, 'B.', 'Simulated CCDF of mutator ages')

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
print(f"\nSaved → {OUT_PDF}")

plt.show()
