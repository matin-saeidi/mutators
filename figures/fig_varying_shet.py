#!/usr/bin/env python3
"""Figure S2. Expected mutator frequency (A) and E[Delta u] / u_hat (B) against
phi*G, from the mutator simulations at 2 N s_het = 10, 20 and 40, +/- 2 standard
errors.

Reads   results/mutator_simulation/2Ns_het_{10,20,40}/summary.json
        (workflow/mutator_simulation.smk)
Writes  mutators_freq_delta_u_varying_2Nhs.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import figures_root, load_mutator_summary   # noqa: E402

import numpy as np                                       # noqa: E402
import matplotlib.pyplot as plt                          # noqa: E402

DATASETS = [(10, "#1B998B"), (20, "purple"), (40, "#D95F02")]

FS_TICK, FS_LABEL = 9, 10
MS, ELW, ECAP = 4.5, 1.2, 2.5
XLIM = (1e-2, 1e3)

fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4))

for two_n_shet, color in DATASETS:
    rows = load_mutator_summary(two_n_shet)
    G, u_hat = rows[0]["G"], rows[0]["u0"]
    x = np.array([r["phi_G"] for r in rows])
    y_mean = np.array([r["mean_q"] for r in rows])
    ye_mean = 2 * np.array([r["se_mean_q"] for r in rows])
    y_rate = x / G * np.array([r["mean_q2"] for r in rows]) / u_hat
    ye_rate = 2 * x / G * np.array([r["se_mean_q2"] for r in rows]) / u_hat

    axes[0].errorbar(x, y_mean, yerr=ye_mean, fmt="o", color=color, ms=MS,
                     elinewidth=ELW, capsize=ECAP, zorder=5,
                     label=rf"$2N_e s_{{het}} = {two_n_shet}$")
    axes[1].errorbar(x, y_rate, yerr=ye_rate, fmt="o", color=color, ms=MS,
                     elinewidth=ELW, capsize=ECAP, zorder=5)

ax = axes[0]
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*XLIM)
ax.set_ylim(3e-7, 1)
ax.set_xlabel(r"Extra mutations genome-wide ($\phi G$)", fontsize=FS_LABEL)
ax.set_ylabel("Frequency", fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.legend(frameon=False, fontsize=FS_TICK)
ax.text(-0.16, 1.06, r"$\mathbf{A.}$ Expected mutator frequency",
        transform=ax.transAxes, fontsize=11, va="bottom", ha="left")

ax = axes[1]
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*XLIM)
ax.set_ylim(3e-7, 1e-2)
ax.set_xlabel(r"Extra mutations genome-wide ($\phi G$)", fontsize=FS_LABEL)
ax.set_ylabel(r"$\mathrm{E}[\Delta u]/\hat{u}$", fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.text(-0.12, 1.06,
        r"$\mathbf{B.}$ Expected increase in" "\n" r"mutation rate relative to the mean",
        transform=ax.transAxes, fontsize=11, va="bottom", ha="left")

plt.tight_layout()
out = figures_root() / "mutators_freq_delta_u_varying_2Nhs.pdf"
plt.savefig(out, format="pdf", bbox_inches="tight")
print(f"wrote {out}")
