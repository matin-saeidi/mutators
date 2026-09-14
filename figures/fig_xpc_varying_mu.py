#!/usr/bin/env python3
"""Figure S5. Frequency distribution of the focal XPC LoF variant against the gene-wide XPC
LoF mutation rate.

  Panel A  South Asian demography, 100,000 replicates per mutation rate
  Panel B  constant Ne = 10^6, 5,000 replicates per mutation rate

21 log-spaced values of mu_gene, from the focal variant's own rate 7.922075e-08
up to 8.8710e-06, at phi = 7.7921e-08 (s = 0.018701).

The simulator tracks every LoF lineage in the gene, so what is simulated is the
gene-wide LoF allele at rate mu_gene; each surviving lineage is then assigned to
the focal variant with probability mu_variant / mu_gene, with mu_variant held
fixed at 7.922075e-08.

Both panels share one y axis and only the left carries the axis label. Boxes are
the median and interquartile range, whiskers the 2.5th and 97.5th percentiles;
outliers are not drawn.

Distributions include every simulated replicate; a replicate in which the
mutator was lost enters at a frequency of exactly 0.

Reads   results/sas/simulations_varying_mu/  (workflow/XPC_varying_gene_mu.smk)
Writes  XPC_varying_LoF_mut_rate_SAS.pdf
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

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.transforms import offset_copy

# -------------------------------
# Load data
# -------------------------------
VMU_DIR = (f"{results_root()}/sas"
           "/simulations_varying_mu/XPC")
NPZ_PATH_SAS     = f"{VMU_DIR}/XPC_sas_varying_mu.npz"
NPZ_PATH_Ne_1M   = f"{VMU_DIR}/XPC_const1M_varying_mu.npz"

# ========================== CONDITIONING SWITCH ===========================
# True  -> each box is conditional on the focal variant segregating in that
#          replicate (the published behaviour)
# False -> every simulated replicate, including those in which no lineage was
#          assigned to the focal variant, which enter with a frequency of 0
#
# False needs a summary that kept every replicate; prepare() below stops with a
# message naming the file and the Snakefile to re-run if it did not.
#
# Note a box of an unconditional distribution is unreadable on a log axis when
# much of its mass sits at zero -- here that is 0.3-0.4% of replicates under the
# South Asian history but 5.6-12.4% at constant Ne, so the constant-Ne panel is
# the one to watch.
CONDITION_ON_SEGREGATING = False            # fixed: this is the unconditioned cell


data_sas  = np.load(NPZ_PATH_SAS)
data_ne1m = np.load(NPZ_PATH_Ne_1M)


def prepare(data, path):
    """Sort npz keys by mu and return (mu_values, distributions).

    These npz files also carry per-lineage allele ages under 'age_s=<mu>' keys,
    which are not frequencies and must be dropped before sorting by float.
    """
    unconditional = summary_is_unconditional(path)
    if not CONDITION_ON_SEGREGATING and not unconditional:
        raise SystemExit(
            f"{path}\n  kept only the replicates in which the focal variant was "
            f"present, so the\n  unconditional distribution cannot be recovered from "
            f"it. Re-run\n  Snakefile_SAS_dem_XPC_varying_gene_mu.")
    mu_keys = sorted((k for k in data.files if not k.startswith("age_s=")),
                     key=lambda k: float(k))
    mu_values = np.array([float(k) for k in mu_keys])
    distributions = []
    for k in mu_keys:
        q = np.asarray(data[k], dtype=float).ravel()
        # a summary that kept every replicate needs the zeros removed here when
        # conditioning; one that did not has already had them removed
        distributions.append(q[q > 0] if (CONDITION_ON_SEGREGATING and unconditional) else q)
    return mu_values, distributions


def draw_panel(ax, mu_values, distributions, title, box_color="darkgray", ylim=None,
               show_ylabel=True):
    # Boxplot widths proportional to log-spacing of mu
    log_mu = np.log10(mu_values)
    spacing = np.min(np.diff(log_mu))
    box_width_log = spacing * 0.8
    widths = [mu * (10**box_width_log - 1) * 0.5 for mu in mu_values]

    drawn = []
    for mu, dist in zip(mu_values, distributions):
        clipped, floored = floor_for_log(dist, f"mu={mu:.3g} [{title}]")
        drawn.append(clipped)
        if floored:
            ax.plot([mu], [Y_FLOOR], marker="v", ms=3.5, color="black",
                    clip_on=False, zorder=6)

    ax.boxplot(
        drawn,
        positions=mu_values,
        widths=widths,
        whis=[2.5, 97.5],
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1),
        boxprops=dict(facecolor=box_color, edgecolor="black", linewidth=0.7),
        whiskerprops=dict(color="black", linewidth=0.7),
        capprops=dict(color="black", linewidth=0.7),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(mu_values.min() * 0.9, mu_values.max() * 1.1)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("XPC LoF mutation rate", fontsize=10)
    # both panels share one y axis, so only the left one is labelled
    if show_ylabel:
        ax.set_ylabel("Frequency", fontsize=10)

    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.tick_params(axis="both", which="minor", labelsize=8)

    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


# -------------------------------
# Figure setup (1 x 2 panels, LaTeX column width = 6.5 in)
# -------------------------------
fig, axes = plt.subplots(
    1, 2, figsize=(6.5, 3.45),
    gridspec_kw={"width_ratios": [1, 1]},
    sharey=True,
    constrained_layout=True,
)

mu_sas,  dist_sas  = prepare(data_sas, NPZ_PATH_SAS)
mu_ne1m, dist_ne1m = prepare(data_ne1m, NPZ_PATH_Ne_1M)

# One shared y axis, wide enough for the 2.5-97.5% whiskers of BOTH panels
# (A: 6.20e-07 to 8.81e-04;  B: 5.00e-07 to 6.10e-03), so the two histories
# can be read against each other directly.
YLIM = (3e-7, 1e-2)

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


# Panel A: South Asian demography
draw_panel(
    axes[0], mu_sas, dist_sas,
    title="South Asian demography",
    box_color="darkgray",
    ylim=YLIM,
)

# Panel B: Constant Ne = 1e6
draw_panel(
    axes[1], mu_ne1m, dist_ne1m,
    title=r"Constant $N_e = 10^6$",
    box_color="#E0B96E",
    ylim=YLIM,
    show_ylabel=False,
)

# --- Panel labels (bold A./B.) and titles (normal weight) ---
titles = ["South Asian demography", r"Constant $N_e = 10^6$"]

# Offsets are in POINTS from the axes' top-left corner, via offset_copy, so the
# padding does not drift if the figure size or panel count changes (an offset in
# axes fractions would).
#
#   LABEL_DX  slight hang to the left of the left spine. The original cell
#             hung 44 pt out to clear a y-axis label on BOTH panels; panel B
#             no longer has one, so 12 pt is enough to read as a hang.
#   LABEL_DY  baseline above the top spine. 22 pt reproduces the headroom of
#             the original cell's y = 1.12 in axes fractions.
#   TITLE_DX  LABEL_DX + width of a bold 10 pt "A." (11.7 pt) + a 4 pt gap
#
# va="baseline" for BOTH, so the label and the title sit on the same baseline.
# va="bottom" would align their bounding-box bottoms instead, and panel B's
# title contains "$N_e$", whose subscript descends below the baseline -- that
# alone would push the title 2.4 px higher than the "B." next to it.
LABEL_DX, LABEL_DY = -12, 12
TITLE_DX = LABEL_DX + 16

for ax, label, title in zip([axes[0], axes[1]], ["A.", "B."], titles):
    tr_label = offset_copy(ax.transAxes, fig=fig, x=LABEL_DX, y=LABEL_DY, units="points")
    tr_title = offset_copy(ax.transAxes, fig=fig, x=TITLE_DX, y=LABEL_DY, units="points")
    ax.text(0, 1, label, transform=tr_label,
            fontsize=10, fontweight="bold",
            va="baseline", ha="left")
    ax.text(0, 1, title, transform=tr_title,
            fontsize=10, fontweight="normal",
            va="baseline", ha="left")

pdf_dir = str(figures_root())
plt.savefig(f"{pdf_dir}/XPC_varying_LoF_mut_rate_SAS.pdf", format="pdf", bbox_inches="tight")
plt.show()

# -------------------------------
# What the figure shows, in numbers
# -------------------------------
for name, mus, dists in [("A  South Asian", mu_sas, dist_sas),
                         ("B  constant Ne=1e6", mu_ne1m, dist_ne1m)]:
    med_lo, med_hi = np.median(dists[0]), np.median(dists[-1])
    print(f"{name}:  n per box {dists[0].size:,} .. {dists[-1].size:,}")
    print(f"    median at mu={mus[0]:.3e}: {med_lo:.3e}")
    print(f"    median at mu={mus[-1]:.3e}: {med_hi:.3e}")
    print(f"    fold change across the {mus[-1]/mus[0]:.0f}x range in mu: {med_lo/med_hi:.2f}\n")