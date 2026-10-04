#!/usr/bin/env python3
"""Figure 4. Pr(proband | mutator parent) (A, analytic), Pr(mutator parent) (B)
and Pr(proband) (C) in a study of 22,000 trios against phi*G, for M = 10 and
M = 100 modifier sites.

Reads   results/proband_discovery/trios/n_22000/age_sex/  (workflow/proband_discovery.smk)
Writes  trio_to_cohort_per_trio_threshold.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SRC, figures_root, load_trio_runs        # noqa: E402

sys.path.insert(0, str(SRC / "proband"))
from analytic import pr_proband_given_mutator_parent        # noqa: E402

import numpy as np                                          # noqa: E402
import matplotlib.pyplot as plt                             # noqa: E402
from matplotlib.lines import Line2D                         # noqa: E402

M_VALUES = [10, 100]
LS = {10: ":", 100: "--"}
DOMINANCE = {"recessive":     ("Recessive",     "purple", 1.0),     # label, colour, dosage
             "semi_dominant": ("Semi-dominant", "peru",   0.5)}
phiG_a = np.logspace(1, 3, 400)

TITLES = ["Pr(proband given\nmutator parent)", "Pr(mutator parent)", "Pr(proband)"]
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.9), sharey=True, constrained_layout=True)

for dominance, (label, color, dosage) in DOMINANCE.items():
    alpha = 0.85 if dominance == "semi_dominant" else 1.0
    axes[0].plot(phiG_a, pr_proband_given_mutator_parent(phiG_a, dosage),
                 color=color, lw=2.4, alpha=alpha)
    for M in M_VALUES:
        runs = load_trio_runs("age_sex", M, dominance)
        x = [r["phi_G"] for r in runs]
        kw = dict(color=color, lw=2.2, ls=LS[M], alpha=alpha)
        axes[1].plot(x, [r["pr_mutator_parent"] for r in runs], **kw)
        axes[2].plot(x, [r["pr_proband"] for r in runs], **kw)

for ax, title in zip(axes, TITLES):
    ax.set_xscale("log")
    ax.set_xlim(10, 1000)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Extra mutations\n" r"genome-wide ($\phi G$)", fontsize=9.5)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", r"$\frac{1}{2}$", "1"])
    ax.set_yticks([0.25, 0.75], minor=True)
    ax.tick_params(which="major", labelsize=9.5, width=1, length=4.5)
    ax.tick_params(which="minor", width=1.0, length=2)
    ax.set_title(title, fontsize=9.5, pad=6)
axes[0].set_ylabel("Probability", fontsize=10)

handles = ([Line2D([0], [0], color=c, lw=2.4, label=lab) for lab, c, _ in DOMINANCE.values()]
           + [Line2D([0], [0], color="0.35", lw=2.0, ls=LS[M], label=f"$M={M}$")
              for M in M_VALUES])
fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False,
           fontsize=8.5, handlelength=2.4, columnspacing=1.6)

# panel letters
fig.canvas.draw()
fig.set_layout_engine("none")
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
y_top = max(inv.transform((0, ax.title.get_window_extent(renderer).y1))[1] for ax in axes)
for letter, ax in zip("ABC", axes):
    x0 = inv.transform((ax.get_window_extent(renderer).x0, 0))[0]
    fig.text(x0 - 0.055, y_top + 0.015, rf"$\bf{{{letter}.}}$",
             ha="left", va="bottom", fontsize=10.5)

out = figures_root() / "trio_to_cohort_per_trio_threshold.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
