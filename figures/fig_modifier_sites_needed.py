#!/usr/bin/env python3
"""Figure S6. Number of modifier sites M needed for a cohort of n trios to contain
a mutator parent with probability 0.95 (Eq. S34), against phi*G, for
n = 22,000 and n = 220,000. The recessive curves are smoothed with a Gaussian
filter (sigma = 2.5 grid points).

Reads   results/proband_discovery/frequencies/  (workflow/proband_discovery.smk)
Writes  modifiers_for_mutator_parent_by_cohort_size.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SRC, figures_root, load_mutator_frequencies   # noqa: E402

sys.path.insert(0, str(SRC / "proband"))
from analytic import pr_mutator_parent_one_site                  # noqa: E402

import numpy as np                                               # noqa: E402
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.ticker import NullFormatter                      # noqa: E402
from scipy.ndimage import gaussian_filter1d                      # noqa: E402

F, S_HET = 0.08, 5e-4             # s* = 2 f phi*G s_het
N_VALUES = (22000, 220000)
TARGET = 0.95
SMOOTH = {"recessive": 2.5, "semi_dominant": 0.0}
H = {"recessive": 0, "semi_dominant": 0.5}

PANELS = [("A", "Recessive mutators",     "recessive",     "purple", (100, 10000)),
          ("B", "Semi-dominant mutators", "semi_dominant", "peru",   (1, 200))]
NSTYLE = [(22000,  dict(ls="-",  lw=2.4), 0.95,  11),        # n, line, alpha, label offset
          (220000, dict(ls="--", lw=2.2), 0.60, -15)]
LABEL_XFRAC = 0.30

fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.1), constrained_layout=True)

for i, (ax, (letter, title, dominance, color, ylim)) in enumerate(zip(axes, PANELS)):
    x, m95 = [], {n: [] for n in N_VALUES}
    for s, q in load_mutator_frequencies(dominance):
        x.append(s / (2 * F * S_HET))
        for n in N_VALUES:
            p1 = np.clip(pr_mutator_parent_one_site(q, n, H[dominance]), 1e-15, 1 - 1e-15)
            m95[n].append(np.log(1 - TARGET) / np.log(1 - p1))
    x = np.array(x)

    for n, style, alpha, dy in NSTYLE:
        y = np.array(m95[n])
        if SMOOTH[dominance] > 0:
            y = gaussian_filter1d(y, SMOOTH[dominance])
        ax.plot(x, y, color=color, alpha=alpha, **style)
        j = int(LABEL_XFRAC * len(x))
        ax.annotate(f"$n$ = {n:,}", xy=(x[j], y[j]), xytext=(0, dy),
                    textcoords="offset points", fontsize=9.5, color=color,
                    alpha=alpha, ha="center")
        print(f"{title:24s} n = {n:>7,}: "
              + "  ".join(f"phiG ~ {t}: M = {y[int(np.argmin(np.abs(x - t)))]:,.0f}"
                          for t in (10, 100, 1000)))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 1000)
    ax.set_ylim(*ylim)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Extra mutations genome-wide ($\phi G$)", fontsize=10)
    if i == 0:
        ax.set_ylabel(r"Number of modifier sites ($M$)", fontsize=10)
    ax.tick_params(which="major", labelsize=9.5, width=1, length=4.5)
    ax.tick_params(which="minor", width=0.8, length=2)
    ax.text(-0.20, 1.05, rf"$\bf{{{letter}.}}$ {title}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10.5, clip_on=False)

out = figures_root() / "modifiers_for_mutator_parent_by_cohort_size.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
