#!/usr/bin/env python3
"""Figure S12. Pr(mutator parent) in a cohort of 22,000 trios with one modifier
site, against phi*G: Eq. 16 (line) and simulations (circles, +/- 2 standard
errors, every fourth phi*G).

Reads   results/proband_discovery/frequencies/
        results/proband_discovery/trios/n_22000/no_age_sex/M_1/
                                                (workflow/proband_discovery.smk)
Writes  checkA_mutator_parent_M1_validation_simple.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (SRC, figures_root, load_mutator_frequencies,   # noqa: E402
                    load_trio_runs)

sys.path.insert(0, str(SRC / "proband"))
from analytic import pr_mutator_parent_one_site                    # noqa: E402

import numpy as np                                                 # noqa: E402
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402

N_TRIOS = 22000
EVERY = 4
PANELS = [("A", "Recessive mutators",     "recessive",     "purple", 0),
          ("B", "Semi-dominant mutators", "semi_dominant", "peru",   0.5)]

fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.175), sharey=True, constrained_layout=True)

for ax, (letter, title, dominance, color, h) in zip(axes, PANELS):
    runs = load_trio_runs("no_age_sex", 1, dominance, N_TRIOS)
    analytic = {s: pr_mutator_parent_one_site(q, N_TRIOS, h)
                for s, q in load_mutator_frequencies(dominance)}
    phiG = np.array([r["phi_G"] for r in runs])
    p_ana = np.array([analytic[float(r["key"])] for r in runs])
    p_sim = np.array([r["pr_mutator_parent"] for r in runs])
    se = np.sqrt(p_sim * (1 - p_sim) / np.array([r["reps"] for r in runs]))

    every = slice(None, None, EVERY)
    ax.plot(phiG, p_ana, color=color, lw=2.2, zorder=1)
    ax.errorbar(phiG[every], p_sim[every], yerr=2 * se[every], fmt="o", ms=7, mfc="none",
                mec=color, ecolor=color, elinewidth=1.0, capsize=2,
                alpha=0.55, zorder=2, linestyle="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 1000)
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel(r"Extra mutations genome-wide ($\phi G$)", fontsize=10)
    ax.tick_params(which="major", labelsize=9.5, width=1, length=4.5)
    ax.tick_params(which="minor", width=0.8, length=2)
    ax.text(-0.14, 1.05, rf"$\bf{{{letter}.}}$ {title}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10.5, clip_on=False)

    z = (p_sim - p_ana) / se
    print(f"{title:24s} (sim - analytic) / s.e.: sd {z.std():.2f}, max |z| {np.abs(z).max():.2f}, "
          f"fraction |z| > 2 {np.mean(np.abs(z) > 2):.2f}")

axes[0].set_ylabel(r"$\Pr(\mathrm{mutator\ parent}\mid M=1)$", fontsize=10)
axes[0].legend(handles=[
    Line2D([0], [0], color="black", lw=2.2, label="analytic"),
    Line2D([0], [0], color="black", lw=1.0, marker="o", ms=7, mfc="none",
           label=r"simulation $\pm2$ s.e.")],
    frameon=False, fontsize=8.5, loc="lower left")

out = figures_root() / "checkA_mutator_parent_M1_validation_simple.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
