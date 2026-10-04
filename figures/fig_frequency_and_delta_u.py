#!/usr/bin/env python3
"""Figure 1. Expected mutator frequency (A) and E[Delta u] / u_hat (B) against phi*G.

Circles: mutator simulations at 2 N s_het = 20, +/- 2 standard errors.
Lines: Eqs. 8 and 11 from the stationary distribution, recessive and semi-dominant.

Reads   results/mutator_simulation/2Ns_het_20/summary.json  (workflow/mutator_simulation.smk)
Writes  Expected_mutator_frequency_and_delta_u.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SRC, figures_root, load_mutator_summary   # noqa: E402

sys.path.insert(0, str(SRC))
from stationary_distribution import (selection_coefficient,   # noqa: E402
                                     stationary_distribution, expected_delta_u)

import numpy as np                                              # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402

# -- simulations ---------------------------------------------------------------
rows = load_mutator_summary(20)
p = rows[0]
N, G, F, S_HET, MU, U_HAT = p["N"], p["G"], p["f"], p["s_het"], p["mu"], p["u0"]

x       = np.array([r["phi_G"] for r in rows])
y_mean  = np.array([r["mean_q"] for r in rows])
ye_mean = 2 * np.array([r["se_mean_q"] for r in rows])
y_rate  = x / G * np.array([r["mean_q2"] for r in rows]) / U_HAT
ye_rate = 2 * x / G * np.array([r["se_mean_q2"] for r in rows]) / U_HAT

# -- stationary distribution -----------------------------------------------------
an_x = np.logspace(-2, 3, 150)
an_mean, an_rate = {0: [], 0.5: []}, {0: [], 0.5: []}
for phi_G in an_x:
    s = selection_coefficient(phi_G, F, S_HET)
    for h in (0, 0.5):
        q, prob = stationary_distribution(N, s, h, MU)
        an_mean[h].append(np.sum(q * prob))
        an_rate[h].append(expected_delta_u(q, prob, phi_G / G, h) / U_HAT)

# -- 2 N s* = 1 and 10 ------------------------------------------------------------
phiG_2Ns1  = 1.0  / (2 * N * 2 * F * S_HET)
phiG_2Ns10 = 10.0 / (2 * N * 2 * F * S_HET)
XLIM = (1e-2, 1e3)

# -- style -------------------------------------------------------------------
PURPLE, PERU = "purple", "peru"
FS_TICK, FS_LABEL, FS_PANEL, FS_TITLE, FS_ANNO = 9, 10, 10, 10, 9
MS, ELW, ECAP = 4.5, 1.2, 2.5


def add_regime_shading(ax, add_labels=False, text_y=None):
    ax.axvspan(XLIM[0], phiG_2Ns1, color="gray", alpha=0.01, zorder=0)
    ax.axvspan(phiG_2Ns1, phiG_2Ns10, color="gray", alpha=0.10, zorder=0)
    ax.axvspan(phiG_2Ns10, XLIM[1], color="gray", alpha=0.20, zorder=0)
    ax.axvline(phiG_2Ns1, color="black", ls=(0, (5, 4)), lw=0.9, zorder=3)
    ax.axvline(phiG_2Ns10, color="black", ls=(0, (5, 4)), lw=0.9, zorder=3)
    if add_labels:
        ax.text(phiG_2Ns1 * 0.74, text_y, r"$2N_e s^*=1$", color="black",
                fontsize=FS_ANNO, ha="center", va="bottom", rotation=90, clip_on=False)
        ax.text(phiG_2Ns10 * 0.74, text_y, r"$2N_e s^*=10$", color="black",
                fontsize=FS_ANNO, ha="center", va="bottom", rotation=90, clip_on=False)


fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))

# -- panel A -----------------------------------------------------------------
ax = axes[0]
add_regime_shading(ax, add_labels=True, text_y=3.5e-7)
ax.errorbar(x, y_mean, yerr=ye_mean, fmt="o", color=PURPLE,
            ms=MS, elinewidth=ELW, capsize=ECAP, zorder=5)
ax.plot(an_x, an_mean[0], "-", color=PURPLE, lw=2, alpha=0.85, zorder=4)
ax.plot(an_x, an_mean[0.5], "-", color=PERU, lw=2, alpha=0.85, zorder=4)
ax.text(8e2, 3.1e-5, "Recessive", color=PURPLE, fontsize=FS_ANNO,
        ha="right", va="bottom", rotation=-19)
ax.text(8e2, 1e-6, "Semi-dominant", color=PERU, fontsize=FS_ANNO,
        ha="right", va="bottom", rotation=-31)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*XLIM)
ax.set_ylim(3e-7, 1)
ax.set_xlabel(r"Extra mutations genome-wide ($\phi G$)", fontsize=FS_LABEL)
ax.set_ylabel("Frequency", fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.text(-0.18, 1.16, r"$\mathbf{A.}$", transform=ax.transAxes,
        fontsize=FS_PANEL, va="bottom", ha="left")
ax.set_title("Expected mutator frequency", fontsize=FS_TITLE)

# -- panel B -----------------------------------------------------------------
ax = axes[1]
add_regime_shading(ax)
ax.errorbar(x, y_rate, yerr=ye_rate, fmt="o", color=PURPLE,
            ms=MS, elinewidth=ELW, capsize=ECAP, zorder=5)
ax.plot(an_x, an_rate[0], "-", color=PURPLE, lw=2, alpha=0.85, zorder=4)
ax.plot(an_x, an_rate[0.5], "-", color=PERU, lw=2, alpha=0.85, zorder=4)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*XLIM)
ax.set_ylim(3e-7, 1e-2)
ax.set_xlabel(r"Extra mutations genome-wide ($\phi G$)", fontsize=FS_LABEL)
ax.set_ylabel(r"$\mathrm{E}[\Delta u]/\hat{u}$", fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.text(-0.18, 1.16, r"$\mathbf{B.}$", transform=ax.transAxes,
        fontsize=FS_PANEL, va="bottom", ha="left")
ax.set_title("Relative expected increase in\n mean mutation rate", fontsize=FS_TITLE)

plt.tight_layout()
out = figures_root() / "Expected_mutator_frequency_and_delta_u.pdf"
plt.savefig(out, format="pdf", bbox_inches="tight")
print(f"wrote {out}")
