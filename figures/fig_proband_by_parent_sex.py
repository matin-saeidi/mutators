#!/usr/bin/env python3
"""Figure S11. Pr(proband) in a trio in which only the father or only the mother
expresses a mutator, against phi*G (analytic, src/proband/analytic.py).

Writes  proband_in_affected_trio_by_parent_sex.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SRC, figures_root                   # noqa: E402

sys.path.insert(0, str(SRC / "proband"))
from analytic import pr_proband_carrier                # noqa: E402

import numpy as np                                     # noqa: E402
import matplotlib.pyplot as plt                        # noqa: E402

phiG = np.logspace(1, 3, 400)
PANELS = [("A", "Recessive mutators",     1.0, "purple"),
          ("B", "Semi-dominant mutators", 0.5, "peru")]
STYLE = {"father": dict(ls="-",  lw=2.4),
         "mother": dict(ls="--", lw=2.4)}

fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.3), sharey=True, constrained_layout=True)

for ax, (letter, title, dosage, color) in zip(axes, PANELS):
    for carrier in ("father", "mother"):
        y = pr_proband_carrier(phiG, dosage, carrier)
        ax.plot(phiG, y, color=color, alpha=0.85, **STYLE[carrier])
        x_half = phiG[int(np.argmin(np.abs(y - 0.5)))]
        ax.annotate(carrier, xy=(x_half, 0.5),
                    xytext=(-6, 14) if carrier == "father" else (6, -16),
                    textcoords="offset points", fontsize=9.5, color=color,
                    ha="right" if carrier == "father" else "left")
        print(f"{title:24s} {carrier:6s}: Pr(proband) = 1/2 at phiG = {x_half:6.0f}")
    ax.set_xscale("log")
    ax.set_xlim(10, 1000)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"Extra mutations genome-wide ($\phi G$)", fontsize=10)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", r"$\frac{1}{2}$", "1"])
    ax.set_yticks([0.25, 0.75], minor=True)
    ax.tick_params(which="major", labelsize=10, width=1, length=4.5)
    ax.tick_params(which="minor", width=1.0, length=2)
    ax.text(-0.13, 1.04, rf"$\bf{{{letter}.}}$ {title}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10.5, clip_on=False)

axes[0].set_ylabel("Pr(proband given a mutator parent)", fontsize=10)

out = figures_root() / "proband_in_affected_trio_by_parent_sex.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
